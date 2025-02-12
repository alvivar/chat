from abc import ABC, abstractmethod
from anthropic import Anthropic
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from typing import Optional, Dict, Any, Iterator, List
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
    REASONING_MODELS = {"o3-mini"}

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
        completion_params = {
            "model": kwargs["model"],
            "messages": messages,
            "stream": stream,
        }

        if kwargs["model"] in self.REASONING_MODELS:
            completion_params["reasoning_effort"] = kwargs.get(
                "reasoning_effort", "high"
            )
        else:
            completion_params["max_tokens"] = kwargs["max_tokens"]
            completion_params["temperature"] = kwargs["temperature"]

        return kwargs["client"].chat.completions.create(**completion_params)

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


class GoogleProvider(AIProvider):
    def create_client(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> genai.Client:
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        return genai.Client(api_key=api_key)

    def create_completion(self, stream: bool, **kwargs: Any):
        completion_params = {
            "model": kwargs["model"],
            "contents": [{"role": "system", "content": kwargs["system"]}]
            + kwargs["messages"],
            "temperature": kwargs["temperature"],
        }

        if stream:
            return kwargs["client"].models.generate_content_stream(**completion_params)
        return kwargs["client"].models.generate_content(**completion_params)

    def iter_chunks(self, completion: Any) -> Iterator[str]:
        for chunk in completion:
            if chunk.text:
                yield chunk.text

    def extract_response(self, completion: Any) -> str:
        return completion.text


class Chat:
    PROVIDER_MAP = {
        "openai": {
            "provider": OpenAIProvider,
            "models": {
                "o3-mini": "o3-mini",
                "4o": "gpt-4o",
                "4o-mini": "gpt-4o-mini",
            },
        },
        "anthropic": {
            "provider": AnthropicProvider,
            "models": {
                "sonnet": "claude-3-5-sonnet-20241022",
                "haiku": "claude-3-5-haiku-20241022",
            },
        },
        "google": {
            "provider": GoogleProvider,
            "models": {
                "gemini-pro": "gemini-1.5-pro-002",
                "gemini-pro-vision": "gemini-1.5-pro-vision-002",
            },
        },
    }

    def __init__(
        self,
        model: str,
        system: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.8,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: str = "high",
    ):
        self.provider = self._get_provider(model, provider)
        self.client = self.provider.create_client(base_url, api_key)
        self.model = self._resolve_model_name(model, provider)
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.messages: List[Dict[str, str]] = []

    def _get_provider(self, model: str, provider: Optional[str]) -> AIProvider:
        if provider:
            if provider not in self.PROVIDER_MAP:
                raise ValueError(
                    f"The provider '{provider}' is not supported. "
                    "Check the available providers and try again."
                )
            return self.PROVIDER_MAP[provider]["provider"]()

        for provider_info in self.PROVIDER_MAP.values():
            models = provider_info["models"]
            if model in models.keys() or model in models.values():
                return provider_info["provider"]()

        raise ValueError(
            f"The model '{model}' isn't supported. "
            "If you're using a custom model, specify a compatible provider and base_url."
        )

    def _resolve_model_name(self, model: str, provider: Optional[str]) -> str:
        if provider:
            provider_info = self.PROVIDER_MAP[provider]
        else:
            for p_info in self.PROVIDER_MAP.values():
                models = p_info["models"]
                if model in models.keys() or model in models.values():
                    provider_info = p_info
                    break
            else:
                return model

        return provider_info["models"].get(model, model)

    def __call__(self, user_message: str, stream: bool = False):
        self.messages.append({"role": "user", "content": user_message})
        return self._generate_new_response(stream)

    def _generate_new_response(self, stream: bool):
        completion = self._create_completion(stream)

        if stream:
            return self._stream_response(completion)

        response = self.provider.extract_response(completion)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _create_completion(self, stream: bool):
        completion_params = {
            "stream": stream,
            "client": self.client,
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": self.system,
            "reasoning_effort": self.reasoning_effort,
        }
        return self.provider.create_completion(**completion_params)

    def _stream_response(self, completion):
        full_response = []
        for chunk in self.provider.iter_chunks(completion):
            full_response.append(chunk)
            yield chunk
        full_response_str = "".join(full_response)
        self.messages.append({"role": "assistant", "content": full_response_str})


def prompt(
    model,
    provider=None,
    base_url=None,
    max_tokens=4096,
    temperature=0.8,
    reasoning_effort="high",
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
            reasoning_effort=reasoning_effort,
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
    If you run this script, it will execute chat tests for all supported models
    across providers, testing both normal and streaming responses.
    """

    # Helper function to test basic chat functionality for a given model
    def test_model(chat, model_name):
        print(f"\nTesting {model_name}:")
        # Test regular non-streaming response
        print("Normal response:")
        print(chat("What color is the sky?"))

        # Test streaming response
        print("\nStreaming response:")
        for chunk in chat("What color is grass?", stream=True):
            print(chunk, end="", flush=True)
        print("\n")

    # Helper function to test the @prompt decorator functionality
    def test_decorator(model, provider=None, base_url=None):
        # Test regular non-streaming prompt
        @prompt(
            model=model,
            provider=provider,
            base_url=base_url,
            max_tokens=256,
            temperature=0.6,
            reasoning_effort="medium",
        )
        def simple_question():
            """You are a helpful AI assistant. Provide concise and accurate responses."""
            return "What color is the sun?"

        # Test streaming prompt
        @prompt(
            model=model,
            provider=provider,
            base_url=base_url,
            max_tokens=256,
            temperature=0.6,
            reasoning_effort="medium",
            stream=True,
        )
        def simple_question_stream():
            """You are a helpful AI assistant. Provide concise and accurate responses."""
            return "What color is the moon?"

        # Execute and print results
        print(f"\nTesting {model} normal response:")
        print(simple_question())

        print(f"\nTesting {model} streaming response:")
        for chunk in simple_question_stream():
            print(chunk, end="", flush=True)
        print("\n")

    # Set up common configuration
    system_prompt = "Provide accurate and concise responses."
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    # Test OpenAI models
    openai_models = ["o3-mini", "4o", "4o-mini"]
    for model in openai_models:
        chat = Chat(
            model,
            system=system_prompt,
            provider="openai",
            api_key=openai_api_key,
        )
        test_model(chat, f"OpenAI {model}")

    # Test Anthropic models
    anthropic_models = ["haiku", "sonnet"]
    for model in anthropic_models:
        chat = Chat(
            model,
            system=system_prompt,
            provider="anthropic",
            api_key=anthropic_api_key,
        )
        test_model(chat, f"Anthropic {model}")

    # Test local models (e.g., running on LM Studio)
    local_models = ["hermes-3-llama-3.2-3b"]
    for model in local_models:
        chat = Chat(
            model,
            system=system_prompt,
            provider="openai",
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
        )
        test_model(chat, f"Local {model}")

    # Test the @prompt decorator with all model types
    print("\nTesting prompt decorator with different models:")

    # Test OpenAI models with decorator
    for model in openai_models:
        test_decorator(model)

    # Test Anthropic models with decorator
    for model in anthropic_models:
        test_decorator(model, provider="anthropic")

    # Test local models with decorator
    for model in local_models:
        test_decorator(model, provider="openai", base_url="http://localhost:1234/v1")
