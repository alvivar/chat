# chat.py

A simple, unified API for interacting with AI language models from multiple providers (OpenAI, Anthropic, and Google). It offers:

-   A `Chat` class for managing conversations with AI models.
-   A `@prompt` decorator to simplify prompt creation.
-   A command-line interface for quick AI interactions.

## Quick Start

### Installation

```bash
pip install openai anthropic google-genai
```

### Set up API keys

```bash
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
export GEMINI_API_KEY='your-key'  # Note: uses GEMINI_API_KEY, not GOOGLE_API_KEY
```

### Basic Usage

```python
from src.chat import Chat

# Simple conversation
chat = Chat(model="sonnet3.5")
response = chat("What's the difference between a list and a tuple in Python?")
print(response)

# With streaming
for chunk in chat("Explain quantum computing", stream=True):
    print(chunk, end="", flush=True)
```

### Command Line

```bash
# Quick question
python src/chat.py "What is the capital of France?"

# Compare multiple models
python src/chat.py "Write a haiku about code" -m gpt4.1 sonnet3.5 gemini-pro

# Stream responses
python src/chat.py "Tell me a story" --stream
```

## Core Features

### Chat Class

The `Chat` class maintains conversation context and supports all major AI providers:

```python
from src.chat import Chat

chat = Chat(
    model="sonnet3.5",  # Short names work: gpt4.1, o4-mini, gemini-pro, etc.
    system="You are a helpful assistant.",
    max_tokens=4096,
    temperature=0.8,
    reasoning_effort="high"  # For reasoning models like o4-mini
)

# Maintains context across messages
response1 = chat("What's the capital of France?")
response2 = chat("What's the weather like there?")  # Knows we're talking about Paris
```

### @prompt Decorator

Streamline AI interactions by transforming functions into prompts:

```python
from src.chat import prompt

@prompt(model="gpt4.1", provider="openai")
def greet_user(name):
    """
    You are a friendly assistant.
    """
    return f"Say hello to {name}."

response = greet_user("Alice")
print(response)

# With streaming
@prompt(model="sonnet3.5", stream=True)
def generate_story(topic):
    """
    You are a creative storyteller.
    """
    return f"Tell a short story about {topic}."

for chunk in generate_story("a brave knight"):
    print(chunk, end="", flush=True)
```

## Supported Models

### OpenAI

-   `"o4-mini"` (o4-mini-2025-04-16) - Reasoning model
-   `"o3"` (o3-2025-04-16) - Reasoning model
-   `"gpt4.1"` (gpt-4.1-2025-04-14)
-   `"gpt4.1-mini"` (gpt-4.1-mini-2025-04-14)

### Anthropic

-   `"opus4"` (claude-opus-4-20250514)
-   `"sonnet4"` (claude-sonnet-4-20250514)
-   `"sonnet3.7"` (claude-3-7-sonnet-20250219)
-   `"sonnet3.5"` (claude-3-5-sonnet-20241022)

### Google

-   `"gemini-pro"` (gemini-2.5-pro-preview-06-05)
-   `"gemini-flash"` (gemini-2.5-flash-preview-05-20)

## Command-Line Interface

### Basic Commands

```bash
# Simple question with default model (gpt4.1)
python src/chat.py "What is the capital of France?"

# Use a specific model
python src/chat.py "Explain quantum computing" -m sonnet4

# Use multiple models for comparison
python src/chat.py "Write a haiku about code" -m gpt4.1 sonnet3.5 gemini-pro

# Stream responses
python src/chat.py "Tell me a story" --stream
```

### Options

-   `-m, --models`: Specify one or more models to use
-   `-s, --system`: Set the system prompt
-   `--max-tokens`: Maximum tokens for response (default: 4096)
-   `--temperature`: Controls randomness (default: 1)
-   `--stream`: Enable streaming responses
-   `--reasoning-effort`: Set reasoning effort for reasoning models (low/medium/high)

### Examples

```bash
# Custom system prompt
python src/chat.py "Solve this problem" --system "You are a mathematics tutor"

# Adjust model parameters
python src/chat.py "Be creative" --temperature 1.2 --max-tokens 500

# Use reasoning effort for supported models
python src/chat.py "Think step by step" -m o4-mini --reasoning-effort high
```

## Advanced Usage

### Local Models (LM Studio, etc.)

```python
chat = Chat(
    model="hermes-3-llama-3.2-3b",
    system="You are a helpful AI assistant.",
    provider="openai",  # LM Studio is compatible with openai provider
    base_url="http://localhost:1234/v1",
)
```

### Full Configuration Options

```python
chat = Chat(
    model="sonnet3.5",
    system="You are a helpful assistant.",
    provider="anthropic",  # "openai", "anthropic", or "google"
    max_tokens=4096,
    temperature=0.8,
    base_url=None,  # Custom API base URL
    api_key=None,   # API key (or use environment variables)
    reasoning_effort="high"  # For reasoning models
)
```

### Custom Providers

Create your own provider by subclassing the `AIProvider` class and implementing:

-   `create_client`: Set up the API client
-   `create_completion`: Generate completions with the model
-   `iter_chunks`: Extract text from streaming responses
-   `extract_response`: Extract text from non-streaming responses

## Testing

Test models across different providers with the included test suite:

### Quick Testing

```bash
# List all available models
python src/test.py --list

# Test all models
python src/test.py --all

# Test specific provider
python src/test.py --provider anthropic

# Test specific models
python src/test.py --model sonnet3.5 --model gpt4.1
```

### Test Options

-   `--all`: Test all available models
-   `--provider`: Test all models from a specific provider (openai, anthropic, google)
-   `--model`: Test specific model(s) - can be used multiple times
-   `--local`: Test local models with custom base URL
-   `--decorators`: Also test @prompt decorators
-   `--decorators-only`: Test only @prompt decorators (no regular chat tests)
-   `--verbose, -v`: Enable detailed output
-   `--base-url`: Custom base URL for local/custom API endpoints
-   `--system`: Custom system prompt for testing

### Local Model Testing

```bash
# Test local models through LM Studio
python src/test.py --local --model hermes-3-llama-3.2-3b
python src/test.py --local --base-url http://localhost:1234/v1 --model your-local-model
```

### Decorator Testing

```bash
# Test decorators alongside regular chat functionality
python src/test.py --provider anthropic --decorators

# Test only decorators for specific models
python src/test.py --decorators-only --model sonnet3.5 --model gpt4.1

# Test decorators for all models from a provider
python src/test.py --decorators-only --provider openai
```

## Examples

### Multi-Model Comparison

```bash
# Compare responses across different providers
python src/chat.py "Explain the concept of recursion" -m gpt4.1 sonnet3.5 gemini-pro
```

Output format:
📨 Sending to 3 models: gpt4.1, sonnet3.5, gemini-pro

---

🤖 gpt4.1:
[response from GPT-4.1]

---

🤖 sonnet3.5:
[response from Claude Sonnet 3.5]

---

🤖 gemini-pro:
[response from Gemini Pro]
