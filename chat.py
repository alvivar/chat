from anthropic import Anthropic
from google import genai
from google.genai.types import GenerateContentConfig
from openai import OpenAI

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Iterator, List
import os
import time
import argparse
import sys


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
    REASONING_MODELS = {"o4-mini", "o3"}

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
        system_instruction = None

        for msg in kwargs["messages"]:
            role, content = msg["role"], msg["content"]
            if role == "system":
                system_instruction = content
            elif role in ("user", "assistant"):
                contents.append(content)

        completion_params = {
            "model": kwargs["model"],
            "contents": contents,
            "config": GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=kwargs["temperature"],
                max_output_tokens=kwargs["max_tokens"],
            ),
        }

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
                "4.1": "gpt-4.1-2025-04-14",
                "4.1-mini": "gpt-4.1-mini-2025-04-14",
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
    max_tokens=None,
    temperature=None,
    reasoning_effort="high",
    api_key=None,
    stream=False,
):
    def decorator(func):
        system_prompt = func.__doc__.strip() if func.__doc__ else None

        chat = Chat(
            model=model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
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


def test_model(chat, model_name, verbose=False):
    """Test a single model with both normal and streaming responses."""
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Testing {model_name}")
        print("=" * 50)
    else:
        print(f"\nTesting {model_name}:")

    try:
        # Test regular non-streaming response
        if verbose:
            print("→ Normal response test:")
        else:
            print("Normal response:")

        response = chat("What color is the sky?")
        print(response)

        # Test streaming response
        if verbose:
            print("\n→ Streaming response test:")
        else:
            print("\nStreaming response:")

        for chunk in chat("What color is grass?", stream=True):
            print(chunk, end="", flush=True)
        print("\n")

        if verbose:
            print(f"✅ {model_name} - All tests passed")

        return True

    except Exception as e:
        print(f"❌ {model_name} - Error: {str(e)}")
        return False


def test_decorator(model, provider=None, base_url=None, verbose=False):
    """Test the @prompt decorator functionality."""
    if verbose:
        print(f"\n→ Testing @prompt decorator for {model}")

    try:
        # Test regular non-streaming prompt
        @prompt(
            model=model,
            provider=provider,
            base_url=base_url,
            max_tokens=256,
            temperature=0.6,
            reasoning_effort="low",
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
            reasoning_effort="low",
            stream=True,
        )
        def simple_question_stream():
            """You are a helpful AI assistant. Provide concise and accurate responses."""
            return "What color is the moon?"

        # Execute and print results
        print(f"\n@prompt decorator - {model} normal response:")
        print(simple_question())

        print(f"\n@prompt decorator - {model} streaming response:")
        for chunk in simple_question_stream():
            print(chunk, end="", flush=True)
        print("\n")

        return True

    except Exception as e:
        print(f"❌ @prompt decorator for {model} - Error: {str(e)}")
        return False


def get_models_by_provider(provider_name):
    """Get all models for a specific provider."""
    if provider_name not in Chat.PROVIDER_MAP:
        return []
    return list(Chat.PROVIDER_MAP[provider_name]["models"].keys())


def test_provider_models(
    provider_name, system_prompt, api_keys, test_decorators=False, verbose=False
):
    """Test all models for a specific provider."""
    models = get_models_by_provider(provider_name)
    if not models:
        print(f"❌ Unknown provider: {provider_name}")
        return 0, 0

    api_key = api_keys.get(provider_name.upper() + "_API_KEY")
    if not api_key:
        print(
            f"⚠️  Warning: No API key found for {provider_name}. Set {provider_name.upper()}_API_KEY environment variable."
        )

    passed = 0
    total = 0

    for model in models:
        total += 1
        try:
            chat = Chat(
                model,
                system=system_prompt,
                provider=provider_name,
                api_key=api_key,
            )

            if test_model(chat, f"{provider_name.title()} {model}", verbose):
                passed += 1

            if test_decorators:
                test_decorator(model, provider=provider_name, verbose=verbose)

        except Exception as e:
            print(f"❌ Failed to initialize {provider_name} {model}: {str(e)}")

        time.sleep(1)  # Rate limiting

    return passed, total


def test_local_models(
    base_url, models, system_prompt, test_decorators=False, verbose=False
):
    """Test local models (e.g., LM Studio)."""
    passed = 0
    total = len(models)

    for model in models:
        try:
            chat = Chat(
                model,
                system=system_prompt,
                provider="openai",  # LM Studio is compatible with OpenAI API
                base_url=base_url,
            )

            if test_model(chat, f"Local {model}", verbose):
                passed += 1

            if test_decorators:
                test_decorator(
                    model, provider="openai", base_url=base_url, verbose=verbose
                )

        except Exception as e:
            print(f"❌ Failed to test local model {model}: {str(e)}")

        time.sleep(1)

    return passed, total


def main():
    parser = argparse.ArgumentParser(
        description="Test AI chat models across different providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py --list                             # List all available models
  python chat.py --all                              # Test all models
  python chat.py --provider openai                  # Test all OpenAI models
  python chat.py --provider anthropic --decorators  # Test Anthropic models + decorators (requires API key)
  python chat.py --model sonnet3.5                  # Test specific model
  python chat.py --model o4-mini --model 4.1        # Test multiple specific models
  python chat.py --local --base-url http://localhost:1234/v1 --model hermes-3-llama-3.2-3b  # Test local models
        """,
    )

    # Main options
    parser.add_argument("--all", action="store_true", help="Test all available models")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "google"],
        help="Test all models from a specific provider",
    )
    parser.add_argument("--model", action="append", help="Test specific model(s)")
    parser.add_argument("--local", action="store_true", help="Test local models")
    parser.add_argument(
        "--list", action="store_true", help="List all available models and exit"
    )

    # Configuration options
    parser.add_argument("--base-url", help="Base URL for local/custom API endpoints")
    parser.add_argument(
        "--decorators", action="store_true", help="Also test @prompt decorators"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--system",
        default="Provide accurate and concise responses.",
        help="System prompt to use",
    )

    args = parser.parse_args()

    # Handle --list option
    if args.list:
        print("Available models by provider:")
        for provider, info in Chat.PROVIDER_MAP.items():
            print(f"\n{provider.upper()}:")
            for short_name, full_name in info["models"].items():
                print(f"  {short_name:<15} -> {full_name}")
        return

    # Validate arguments
    if not any([args.all, args.provider, args.model, args.local]):
        parser.error("Must specify one of: --all, --provider, --model, or --local")

    if args.local and not args.model:
        parser.error("--local requires --model to specify which local models to test")

    if args.local and not args.base_url:
        args.base_url = "http://localhost:1234/v1"  # Default LM Studio URL

    # Get API keys
    api_keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GEMINI_API_KEY"),
    }

    print("🚀 Starting AI model tests...")
    if args.verbose:
        print(f"System prompt: {args.system}")
        print(f"Test decorators: {args.decorators}")

    total_passed = 0
    total_tests = 0

    try:
        if args.all:
            # Test all providers
            for provider in ["openai", "anthropic", "google"]:
                passed, total = test_provider_models(
                    provider, args.system, api_keys, args.decorators, args.verbose
                )
                total_passed += passed
                total_tests += total

        elif args.provider:
            # Test specific provider
            passed, total = test_provider_models(
                args.provider, args.system, api_keys, args.decorators, args.verbose
            )
            total_passed += passed
            total_tests += total

        elif args.local:
            # Test local models
            passed, total = test_local_models(
                args.base_url, args.model, args.system, args.decorators, args.verbose
            )
            total_passed += passed
            total_tests += total

        elif args.model:
            # Test specific models
            for model in args.model:
                total_tests += 1
                try:
                    # Determine provider automatically or use local
                    if args.local:
                        chat = Chat(
                            model,
                            system=args.system,
                            provider="openai",
                            base_url=args.base_url,
                        )
                        model_name = f"Local {model}"
                    else:
                        chat = Chat(model, system=args.system)
                        model_name = model

                    if test_model(chat, model_name, args.verbose):
                        total_passed += 1

                    if args.decorators:
                        if args.local:
                            test_decorator(
                                model,
                                provider="openai",
                                base_url=args.base_url,
                                verbose=args.verbose,
                            )
                        else:
                            test_decorator(model, verbose=args.verbose)

                except Exception as e:
                    print(f"❌ Failed to test model {model}: {str(e)}")

                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"📊 Test Summary: {total_passed}/{total_tests} models passed")
    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(f"Success rate: {success_rate:.1f}%")

    if total_passed < total_tests:
        print("\n💡 Tips:")
        print("- Make sure API keys are set as environment variables")
        print("- Check your internet connection")
        print("- Some models may have usage limits or require special access")


if __name__ == "__main__":
    main()
