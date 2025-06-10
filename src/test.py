import argparse
import os
import sys
import time
from chat import Chat, prompt


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
  python test.py --list                             # List all available models
  python test.py --all                              # Test all models
  python test.py --provider openai                  # Test all OpenAI models
  python test.py --provider anthropic --decorators  # Test Anthropic models + decorators (requires API key)
  python test.py --model sonnet3.5                  # Test specific model
  python test.py --model o4-mini --model 4.1        # Test multiple specific models
  python test.py --local --base-url http://localhost:1234/v1 --model hermes-3-llama-3.2-3b  # Test local models
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
