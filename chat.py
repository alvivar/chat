from abc import ABC, abstractmethod
from anthropic import Anthropic
from openai import OpenAI
from typing import Optional, Dict, Any, Iterator, List
import hashlib
import json
import os


class AIProvider(ABC):
    @abstractmethod
    def create_client(self, base_url: Optional[str], api_key: Optional[str]):
        pass

    @abstractmethod
    def create_completion(self, stream: bool, **kwargs: Any):
        pass

    @abstractmethod
    def iter_chunks(self, completion: Any) -> Iterator[str]:
        pass

    @abstractmethod
    def extract_response(self, completion: Any) -> str:
        pass


class OpenAIProvider(AIProvider):
    def create_client(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> OpenAI:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        return OpenAI(base_url=base_url, api_key=api_key)

    def create_completion(self, stream: bool, **kwargs: Any):
        messages = [{"role": "system", "content": kwargs["system"]}] + kwargs[
            "messages"
        ]
        return kwargs["client"].chat.completions.create(
            model=kwargs["model"],
            messages=messages,
            max_tokens=kwargs["max_tokens"],
            temperature=kwargs["temperature"],
            stream=stream,
        )

    def iter_chunks(self, completion: Any) -> Iterator[str]:
        for chunk in completion:
            if content := chunk.choices[0].delta.content:
                yield content

    def extract_response(self, completion: Any) -> str:
        return completion.choices[0].message.content


class AnthropicProvider(AIProvider):
    def create_client(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> Anthropic:
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        return Anthropic(api_key=api_key)

    def create_completion(self, stream: bool, **kwargs: Any):
        common_params = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "max_tokens": kwargs["max_tokens"],
            "temperature": kwargs["temperature"],
            "system": kwargs["system"],
        }
        return (
            kwargs["client"].messages.stream(**common_params)
            if stream
            else kwargs["client"].messages.create(**common_params)
        )

    def iter_chunks(self, completion: Any) -> Iterator[str]:
        with completion as stream:
            yield from stream.text_stream

    def extract_response(self, completion: Any) -> str:
        return completion.content[0].text


# TODO:
# - Add Groq
# - Add Anthropic AWS Bedrock
# - Add Google?


class Chat:
    PROVIDER_MAP = {
        "openai": {
            "provider": OpenAIProvider,
            "models": [
                "gpt-3.5-turbo",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-4o-mini",
                "gpt-4o",
            ],
        },
        "anthropic": {
            "provider": AnthropicProvider,
            "models": [
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-5-sonnet-20240620",
            ],
        },
    }

    def __init__(
        self,
        model: str,
        system: str = "",
        max_tokens: int = 512,
        temperature: float = 0.8,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        use_cache: bool = False,
    ):
        self.provider = self._get_provider(model, provider)
        self.client = self.provider.create_client(base_url, api_key)
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key

        self.messages: List[Dict[str, str]] = []
        self.use_cache = use_cache
        self.cache = self._initialize_cache() if use_cache else None

    def _get_provider(self, model: str, provider: Optional[str]) -> AIProvider:
        if provider:
            if provider not in self.PROVIDER_MAP:
                raise ValueError(
                    f"The provider '{provider}' is not supported. "
                    "Check the available providers and try again."
                )
            return self.PROVIDER_MAP[provider]["provider"]()

        for provider_info in self.PROVIDER_MAP.values():
            if model in provider_info["models"]:
                return provider_info["provider"]()

        raise ValueError(
            f"The model '{model}' isn't supported. "
            "If you're using a custom model, specify a compatible provider and base_url."
        )

    def __call__(
        self,
        user_message: str,
        stream: bool = False,
        ignore_cache: bool = False,
    ):
        self.messages.append({"role": "user", "content": user_message})

        if self.use_cache and not ignore_cache:
            if cached_response := self._get_cached_response():
                return cached_response

        return self._generate_new_response(stream, ignore_cache)

    def _get_cached_response(self) -> Optional[str]:
        cache_key = self._generate_cache_key()
        if self.cache and cache_key in self.cache:
            cached_response = self.cache[cache_key]
            self.messages.append({"role": "assistant", "content": cached_response})
            return cached_response
        return None

    def _generate_new_response(self, stream: bool, ignore_cache: bool):
        cache_key = (
            self._generate_cache_key() if self.use_cache and not ignore_cache else None
        )
        completion = self._create_completion(stream)

        if stream:
            return self._stream_response(completion, cache_key)

        response = self.provider.extract_response(completion)
        self._update_messages_and_cache(response, cache_key)

        return response

    def _create_completion(self, stream: bool):
        return self.provider.create_completion(
            stream,
            client=self.client,
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
        )

    def _stream_response(self, completion, cache_key: Optional[str]):
        full_response = []
        for chunk in self.provider.iter_chunks(completion):
            full_response.append(chunk)
            yield chunk
        full_response_str = "".join(full_response)
        self._update_messages_and_cache(full_response_str, cache_key)

    def _update_messages_and_cache(self, response: str, cache_key: Optional[str]):
        self.messages.append({"role": "assistant", "content": response})
        if cache_key is not None:
            self._cache_response(cache_key, response)

    def _generate_cache_key(self) -> str:
        requirements = [self.model, self.system] + self.messages
        return hashlib.md5(json.dumps(requirements).encode()).hexdigest()

    def _initialize_cache(self) -> Dict[str, str]:
        self.cache_file = os.path.join(os.path.dirname(__file__), "chat_cache.json")
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(
                f"There was an issue initializing the cache: {e}. "
                "We'll start with an empty cache."
            )
        return {}

    def _cache_response(self, cache_key: str, response: str):
        if self.cache is not None:
            self.cache[cache_key] = response
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)


def prompt(
    model,
    provider=None,
    base_url=None,
    max_tokens=1024,
    temperature=0.7,
    stream=False,
):
    def decorator(func):
        system_prompt = func.__doc__.strip() if func.__doc__ else None

        chat = Chat(
            model=model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            provider=provider,
            base_url=base_url,
        )

        def wrapper(*args, **kwargs):
            try:
                func_return = func(*args, **kwargs)
                return chat(func_return, stream=stream)
            except Exception as e:
                print(
                    f"Something went wrong in the prompt decorator: {str(e)}. "
                    "Let's try to fix this and give it another go!"
                )

        return wrapper

    return decorator


if __name__ == "__main__":
    """
    If you run this script, it will execute chat tests for OpenAI, Anthropic,
    and Hermes (from LM Studio) providers, as well as demonstrate the usage of
    the prompt decorator.
    """

    # Helper functions for testing.

    @prompt(model="gpt-4o-mini", max_tokens=256, temperature=0.6)
    def country_info_openai(country: str = "France"):
        """
        You are a helpful AI assistant. Provide concise and accurate responses to user queries.
        """

        return f"What is the capital of {country} and what is its most famous landmark?"

    @prompt(model="gpt-4o-mini", max_tokens=256, temperature=0.6, stream=True)
    def country_info_stream_openai(country: str = "France"):
        """
        You are a helpful AI assistant. Provide concise and accurate responses to user queries.
        """

        return f"What is the capital of {country} and what is its most famous landmark?"

    @prompt(
        model="hermes-3-llama-3.1-8b",
        provider="openai",
        base_url="http://localhost:1234/v1",
        max_tokens=256,
        temperature=0.6,
    )
    def country_info_hermes(country: str = "France"):
        """
        You are a helpful AI assistant. Provide concise and accurate responses to user queries.
        """

        return f"What is the capital of {country} and what is its most famous landmark?"

    @prompt(
        model="hermes-3-llama-3.1-8b",
        provider="openai",
        base_url="http://localhost:1234/v1",
        max_tokens=256,
        temperature=0.6,
        stream=True,
    )
    def country_info_stream_hermes(country: str = "France"):
        """
        You are a helpful AI assistant. Provide concise and accurate responses to user queries.
        """

        return f"What is the capital of {country} and what is its most famous landmark?"

    def stream(chat, text, debug=True):
        response = ""
        for chunk in chat(text, stream=True):
            response += chunk
            if debug:
                print(chunk, end="", flush=True)
        return response

    def chat_test(chat, name, ignore_cache=False):
        print(f"{name} Test:")

        response = chat(
            "What is the capital of the moon?",
            ignore_cache=ignore_cache,
        )
        print(response)

        print("\n----\n")
        print(f"{name} Streaming Test:")

        for chunk in chat(
            "Are you sure?",
            stream=True,
            ignore_cache=ignore_cache,
        ):
            print(chunk, end="", flush=True)

        print("\n\n----\n")

    # Set up!

    system_prompt = "Provide accurate and concise responses."
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # OpenAI

    openai_chat = Chat(
        "gpt-4o-mini",
        system=system_prompt,
        provider="openai",
        api_key=openai_api_key,
        use_cache=True,
    )

    chat_test(openai_chat, "OpenAI")

    # Anthropic

    anthropic_chat = Chat(
        "claude-3-haiku-20240307",
        system=system_prompt,
        provider="anthropic",
        api_key=anthropic_api_key,
        use_cache=True,
    )

    chat_test(anthropic_chat, "Anthropic")

    # Hermes (OpenAI compatible) from LM Studio

    hermes_chat = Chat(
        "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
        system=system_prompt,
        provider="openai",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        use_cache=True,
    )

    chat_test(hermes_chat, "Hermes")

    # Testing the prompt decorator.

    print("Prompt Decorator Test:")
    print(country_info_openai("Moon"))

    print("\n----\n")

    print("Prompt Decorator Streaming Test:")
    for chunk in country_info_stream_openai("Moon"):
        print(chunk, end="", flush=True)

    print("\n----\n")

    # Testing the prompt decorator with Hermes from LM Studio.

    print("Prompt Decorator with Hermes Test:")
    print(country_info_hermes("Sun"))

    print("\n----\n")

    print("Prompt Decorator with Hermes Streaming Test:")
    for chunk in country_info_stream_hermes("Sun"):
        print(chunk, end="", flush=True)


# TODO
# - Make sure exceptions are handled correctly
# - Test if cache is working with decorators
# - Test consecutive calls to decorator to ensure context is maintained
# - Implement and test decorator functionality with all supported providers (OpenAI, Anthropic, etc.)
