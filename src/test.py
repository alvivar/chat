import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from chat import Chat, prompt


def test_model(chat: Chat, model_name: str, verbose: bool = False) -> bool:
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Testing {model_name}")
        print("=" * 50)
    else:
        print(f"\nTesting {model_name}:")

    try:
        print("→ Normal response test:" if verbose else "Normal response:")
        response = chat("What color is the sky?")
        print(response)

        print("\n→ Streaming response test:" if verbose else "\nStreaming response:")
        for chunk in chat("What color is grass?", stream=True):
            print(chunk, end="", flush=True)
        print("\n")

        if verbose:
            print(f"✅ {model_name} - All tests passed")

        return True

    except Exception as e:
        print(f"❌ {model_name} - Error: {str(e)}")
        return False


def test_decorator(
    model: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    if verbose:
        print(f"\n→ Testing @prompt decorator for {model}")

    try:
        decorator_kwargs = {
            "model": model,
            "provider": provider,
            "base_url": base_url,
            "max_tokens": 2048,
            "temperature": 1,
            "reasoning_effort": "low",
        }

        @prompt(**decorator_kwargs)
        def simple_question():
            """You are a helpful AI assistant. Provide concise and accurate responses."""
            return "What color is the sun?"

        @prompt(**decorator_kwargs, stream=True)
        def simple_question_stream():
            """You are a helpful AI assistant. Provide concise and accurate responses."""
            return "What color is the moon?"

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


def get_models_by_provider(provider_name: str) -> List[str]:
    return list(Chat.PROVIDER_MAP.get(provider_name, {}).get("models", {}).keys())


def create_chat_instance(
    model: str,
    system_prompt: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Chat:
    kwargs = {"model": model, "system": system_prompt}
    if provider:
        kwargs["provider"] = provider
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return Chat(**kwargs)


def test_provider_models(
    provider_name: str,
    system_prompt: str,
    api_keys: Dict[str, str],
    test_decorators: bool = False,
    verbose: bool = False,
) -> Tuple[int, int]:
    models = get_models_by_provider(provider_name)
    if not models:
        print(f"❌ Unknown provider: {provider_name}")
        return 0, 0

    api_key_name = f"{provider_name.upper()}_API_KEY"
    api_key = api_keys.get(api_key_name)
    if not api_key:
        print(
            f"⚠️  Warning: No API key found for {provider_name}. Set {api_key_name} environment variable."
        )

    passed = 0
    total = len(models)

    for model in models:
        try:
            chat = create_chat_instance(
                model, system_prompt, provider_name, api_key=api_key
            )

            if test_model(chat, f"{provider_name.title()} {model}", verbose):
                passed += 1

            if test_decorators:
                test_decorator(model, provider=provider_name, verbose=verbose)

        except Exception as e:
            print(f"❌ Failed to initialize {provider_name} {model}: {str(e)}")

        time.sleep(1)

    return passed, total


def test_local_models(
    base_url: str,
    models: List[str],
    system_prompt: str,
    test_decorators: bool = False,
    verbose: bool = False,
) -> Tuple[int, int]:
    passed = 0
    total = len(models)

    for model in models:
        try:
            chat = create_chat_instance(model, system_prompt, "openai", base_url)

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


def test_specific_models(
    models: List[str],
    system_prompt: str,
    is_local: bool = False,
    base_url: Optional[str] = None,
    test_decorators: bool = False,
    verbose: bool = False,
) -> Tuple[int, int]:
    passed = 0
    total = len(models)

    for model in models:
        try:
            if is_local:
                chat = create_chat_instance(model, system_prompt, "openai", base_url)
                model_name = f"Local {model}"
            else:
                chat = create_chat_instance(model, system_prompt)
                model_name = model

            if test_model(chat, model_name, verbose):
                passed += 1

            if test_decorators:
                if is_local:
                    test_decorator(
                        model, provider="openai", base_url=base_url, verbose=verbose
                    )
                else:
                    test_decorator(model, verbose=verbose)

        except Exception as e:
            print(f"❌ Failed to test model {model}: {str(e)}")

        time.sleep(1)

    return passed, total


def get_api_keys() -> Dict[str, str]:
    return {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GEMINI_API_KEY"),
    }


def print_summary(total_passed: int, total_tests: int) -> None:
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


def run_decorator_tests(
    provider: Optional[str],
    models: Optional[List[str]],
    api_keys: Dict[str, str],
    verbose: bool,
    local: bool = False,
    base_url: Optional[str] = None,
) -> Tuple[int, int]:
    total_passed = 0
    total_tests = 0

    if provider:
        provider_models = get_models_by_provider(provider)
        api_key_name = f"{provider.upper()}_API_KEY"
        api_key = api_keys.get(api_key_name)
        if not api_key:
            print(
                f"⚠️  Warning: No API key found for {provider}. Set {api_key_name} environment variable."
            )

        for model in provider_models:
            try:
                if test_decorator(model, provider=provider, verbose=verbose):
                    total_passed += 1
                total_tests += 1
                time.sleep(1)
            except Exception as e:
                print(f"❌ Failed to test decorator for {provider} {model}: {str(e)}")
                total_tests += 1

    elif models:
        for model in models:
            try:
                if local:
                    success = test_decorator(
                        model,
                        provider="openai",
                        base_url=base_url,
                        verbose=verbose,
                    )
                else:
                    success = test_decorator(model, verbose=verbose)

                if success:
                    total_passed += 1
                total_tests += 1
                time.sleep(1)
            except Exception as e:
                print(f"❌ Failed to test decorator for {model}: {str(e)}")
                total_tests += 1

    return total_passed, total_tests


def main():
    parser = argparse.ArgumentParser(
        description="Test AI chat models across different providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --list                             # List all available models
  python test.py --all                              # Test all models
  python test.py --provider openai                  # Test all OpenAI models
  python test.py --provider anthropic --decorators  # Test Anthropic models + decorators (requires API key)
  python test.py --model sonnet3.5                  # Test specific model
  python test.py --model o4-mini --model 4.1        # Test multiple specific models
  python test.py --local --base-url http://localhost:1234/v1 --model hermes-3-llama-3.2-3b  # Test local models
  python test.py --decorators-only --provider openai  # Test only decorators for OpenAI
  python test.py --decorators-only --model sonnet3.5  # Test only decorators for specific model
        """,
    )

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

    parser.add_argument("--base-url", help="Base URL for local/custom API endpoints")
    parser.add_argument(
        "--decorators", action="store_true", help="Also test @prompt decorators"
    )
    parser.add_argument(
        "--decorators-only",
        action="store_true",
        help="Test only @prompt decorators (no regular chat tests)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--system",
        default="Provide accurate and concise responses.",
        help="System prompt to use",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models by provider:")
        for provider, info in Chat.PROVIDER_MAP.items():
            print(f"\n{provider.upper()}:")
            for short_name, full_name in info["models"].items():
                print(f"  {short_name:<15} -> {full_name}")
        return

    if not any([args.all, args.provider, args.model, args.local, args.decorators_only]):
        parser.error(
            "Must specify one of: --all, --provider, --model, --local, or --decorators-only"
        )

    if args.local and not args.model:
        parser.error("--local requires --model to specify which local models to test")

    if args.local and not args.base_url:
        args.base_url = "http://localhost:1234/v1"

    if args.decorators_only and not any([args.provider, args.model]):
        parser.error(
            "--decorators-only requires --provider or --model to specify which models to test"
        )

    api_keys = get_api_keys()

    print("🚀 Starting AI model tests...")
    if args.verbose:
        print(f"System prompt: {args.system}")
        print(f"Test decorators: {args.decorators or args.decorators_only}")
        if args.decorators_only:
            print("Mode: Decorators only")

    total_passed = 0
    total_tests = 0

    try:
        if args.decorators_only:
            total_passed, total_tests = run_decorator_tests(
                args.provider,
                args.model,
                api_keys,
                args.verbose,
                args.local,
                args.base_url,
            )
        elif args.all:
            for provider in ["openai", "anthropic", "google"]:
                passed, total = test_provider_models(
                    provider, args.system, api_keys, args.decorators, args.verbose
                )
                total_passed += passed
                total_tests += total
        elif args.provider:
            passed, total = test_provider_models(
                args.provider, args.system, api_keys, args.decorators, args.verbose
            )
            total_passed += passed
            total_tests += total
        elif args.local:
            passed, total = test_local_models(
                args.base_url, args.model, args.system, args.decorators, args.verbose
            )
            total_passed += passed
            total_tests += total
        elif args.model:
            passed, total = test_specific_models(
                args.model,
                args.system,
                args.local,
                args.base_url,
                args.decorators,
                args.verbose,
            )
            total_passed += passed
            total_tests += total

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)

    print_summary(total_passed, total_tests)


if __name__ == "__main__":
    main()
