import argparse
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator, List

from anthropic import Anthropic
from google import genai
from google.genai.types import GenerateContentConfig
from openai import OpenAI


DEFAULT_SYSTEM_PROMPT = """
You are a helpful, concise AI assistant.
Be engaging and creative.
Think critically and offer unique perspectives.
Don't use Markdown, you are writing in a text-based interface.
"""

DEFAULT_MODEL = ["gpt4.1"]
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

REASONING_MODELS = {"o3", "o4-mini"}
DEFAULT_REASONING_EFFORT = "high"


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
        return OpenAI(
            base_url=base_url,
            api_key=api_key
            or os.environ.get("OPENAI_API_KEY")
            or "None",  # "None" means the client uses the OpenAI library but is not connecting to OpenAI's API.
        )

    def create_completion(self, stream: bool, **kwargs: Any):
        completion_params = {
            "model": kwargs["model"],
            "messages": [{"role": "system", "content": kwargs["system"]}]
            + kwargs["messages"],
            "stream": stream,
        }

        if any(model in kwargs["model"] for model in self.REASONING_MODELS):
            completion_params["max_completion_tokens"] = kwargs.get("max_tokens", 4096)
            completion_params["reasoning_effort"] = kwargs.get(
                "reasoning_effort", "high"
            )
        else:
            completion_params.update(
                {
                    "max_tokens": kwargs["max_tokens"],
                    "temperature": kwargs["temperature"],
                }
            )

        return kwargs["client"].chat.completions.create(**completion_params)

    def iter_chunks(self, completion: Any) -> Iterator[str]:
        return (
            chunk.choices[0].delta.content
            for chunk in completion
            if chunk.choices[0].delta.content
        )

    def extract_response(self, completion: Any) -> str:
        return completion.choices[0].message.content


class AnthropicProvider(AIProvider):
    def create_client(
        self,
        base_url: Optional[str],
        api_key: Optional[str],
    ) -> Anthropic:
        return Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def create_completion(self, stream: bool, **kwargs: Any):
        params = {
            "model": kwargs["model"],
            "messages": kwargs["messages"],
            "max_tokens": kwargs["max_tokens"],
            "temperature": kwargs["temperature"],
            "system": kwargs["system"],
        }
        client = kwargs["client"].messages
        return client.stream(**params) if stream else client.create(**params)

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
        return genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def create_completion(self, stream: bool, **kwargs: Any):
        contents = []
        for msg in kwargs["messages"]:
            if msg.get("content"):
                contents.append(msg["content"])

        config = None
        system_instruction = kwargs.get("system")
        if (
            system_instruction
            or kwargs.get("temperature") is not None
            or kwargs.get("max_tokens")
        ):
            config_params = {}
            if system_instruction:
                config_params["system_instruction"] = system_instruction
            if kwargs.get("temperature") is not None:
                config_params["temperature"] = kwargs["temperature"]
            if kwargs.get("max_tokens"):
                config_params["max_output_tokens"] = kwargs["max_tokens"]
            config = GenerateContentConfig(**config_params)

        completion_params = {
            "model": kwargs["model"],
            "contents": contents,
        }
        if config:
            completion_params["config"] = config

        models = kwargs["client"].models
        return (
            models.generate_content_stream(**completion_params)
            if stream
            else models.generate_content(**completion_params)
        )

    def iter_chunks(self, completion: Any) -> Iterator[str]:
        return (chunk.text for chunk in completion if chunk.text)

    def extract_response(self, completion: Any) -> str:
        return completion.text


class Chat:
    PROVIDER_MAP = {
        "openai": {
            "provider": OpenAIProvider,
            "models": {
                "o4-mini": "o4-mini-2025-04-16",
                "o3": "o3-2025-04-16",
                "gpt4.1": "gpt-4.1-2025-04-14",
                "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
            },
        },
        "anthropic": {
            "provider": AnthropicProvider,
            "models": {
                "opus4": "claude-opus-4-20250514",
                "sonnet4": "claude-sonnet-4-20250514",
                "sonnet3.7": "claude-3-7-sonnet-20250219",
                "sonnet3.5": "claude-3-5-sonnet-20241022",
            },
        },
        "google": {
            "provider": GoogleProvider,
            "models": {
                "gemini-pro": "gemini-2.5-pro-preview-06-05",
                "gemini-flash": "gemini-2.5-flash-preview-05-20",
            },
        },
    }

    def __init__(
        self,
        model: str,
        system: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
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
    max_tokens=None,
    temperature=None,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    api_key=None,
    stream=False,
):
    def decorator(func):
        system_prompt = func.__doc__.strip() if func.__doc__ else ""
        chat_instance = None

        def wrapper(*args, **kwargs):
            nonlocal chat_instance
            if chat_instance is None:
                chat_instance = Chat(
                    model=model,
                    system=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    reasoning_effort=reasoning_effort,
                    provider=provider,
                    base_url=base_url,
                    api_key=api_key,
                )

            return chat_instance(func(*args, **kwargs), stream=stream)

        return wrapper

    return decorator


def main():
    parser = argparse.ArgumentParser(
        description="AI Chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Hello, how are you?"
  %(prog)s "Explain quantum computing" -m gpt4.1 sonnet4
  %(prog)s "Write a poem" --system "You are a creative poet"
  %(prog)s "Solve this math problem" --temperature 0.2 --max-tokens 1000 --no-stream
        """.strip(),
    )

    parser.add_argument("message", nargs="?", help="Message to send to the model(s)")
    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        default=DEFAULT_MODEL,
        help="Model(s) to use (default: gpt4.1). Can specify multiple models.",
    )
    parser.add_argument(
        "-s",
        "--system",
        default=DEFAULT_SYSTEM_PROMPT.strip(),
        help="System prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming responses"
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=["low", "medium", "high"],
        help=f"Reasoning effort for reasoning models (default: {DEFAULT_REASONING_EFFORT})",
    )

    args = parser.parse_args()

    if not args.message:
        parser.print_help()
        return

    all_supported_models = {
        model
        for provider_info in Chat.PROVIDER_MAP.values()
        for model in provider_info["models"]
    }

    unsupported_models = [
        model for model in args.models if model not in all_supported_models
    ]

    if unsupported_models:
        print(f"❌ Unsupported model(s): {', '.join(unsupported_models)}")
        print(f"Supported models: {', '.join(sorted(all_supported_models))}")
        return

    def create_chat(model):
        return Chat(
            model=model,
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
        )

    def process_response(chat, model):
        print(f"🤖 {model}:\n")
        if not args.no_stream:
            for chunk in chat(args.message, stream=True):
                print(chunk, end="", flush=True)
            print()
        else:
            response = chat(args.message)
            print(response)

    if len(args.models) == 1:
        chat = create_chat(args.models[0])
        process_response(chat, args.models[0])
    else:
        print(f"📨 Sending to {len(args.models)} models: {', '.join(args.models)}")

        for i, model in enumerate(args.models):
            if i > 0:
                print("\n" + "-" * 80)
            else:
                print("-" * 80)

            chat = create_chat(model)
            process_response(chat, model)


if __name__ == "__main__":
    main()
