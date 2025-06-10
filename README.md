# chat.py

A simple, unified API for interacting with AI language models from multiple providers (OpenAI, Anthropic, and Google). It offers:

-   A `Chat` class for managing conversations with AI models.
-   A `@prompt` decorator to simplify prompt creation.
-   A command-line interface for quick AI interactions.

## Using the `Chat` Class

The `Chat` class allows you to interact with AI models in a conversational manner.

### Initialization

```python
from chat import Chat

chat = Chat(
    model="sonnet3.5",  # or "gpt4.1", "gemini-pro", "o4-mini", "sonnet4", "gemini-flash"
    system="You are a helpful assistant.",
    provider="anthropic",  # "openai" or "google"
    max_tokens=4096,
    temperature=0.8,
    reasoning_effort="high"  # Optional, for reasoning models
)
```

-   **Parameters:**
    -   `model`: The AI model to use (supported models include):
        -   OpenAI: `"o4-mini"` (o4-mini-2025-04-16), `"o3"` (o3-2025-04-16), `"gpt4.1"` (gpt-4.1-2025-04-14), `"gpt4.1-mini"` (gpt-4.1-mini-2025-04-14)
        -   Anthropic: `"opus4"` (claude-opus-4-20250514), `"sonnet4"` (claude-sonnet-4-20250514), `"sonnet3.7"` (claude-3-7-sonnet-20250219), `"sonnet3.5"` (claude-3-5-sonnet-20241022)
        -   Google: `"gemini-pro"` (gemini-2.5-pro-preview-06-05), `"gemini-flash"` (gemini-2.5-flash-preview-05-20)
    -   `system`: The system prompt that defines the assistant's behavior.
    -   `provider`: The AI provider (`"openai"`, `"anthropic"`, or `"google"`).
    -   `max_tokens`: Maximum number of tokens for the response.
    -   `temperature`: Controls the randomness of the output.
    -   `base_url`: (Optional) Custom API base URL.
    -   `api_key`: (Optional) API key for the provider.
    -   `reasoning_effort`: (Optional) Controls reasoning depth for supported models ("low", "medium", "high").

### Sending Messages

```python
response = chat("What's the difference between a list and a tuple in Python?")
print(response)
```

### Streaming Responses

For large responses, you might prefer to stream the output:

```python
for chunk in chat("Explain quantum computing in simple terms.", stream=True):
    print(chunk, end="", flush=True)
```

## Using the `@prompt` Decorator

The `@prompt` decorator streamlines AI interactions by transforming a function's output into a prompt. The function's docstring (`"""`) serves as the system prompt, while its return value becomes the user prompt sent to the AI model.

### Basic Usage

```python
from chat import prompt

@prompt(model="gpt4.1", provider="openai")
def greet_user(name):
    """
    You are a friendly assistant.
    """
    return f"Say hello to {name}."

response = greet_user("Alice")
print(response)
```

### With Streaming

```python
@prompt(model="gpt4.1", provider="openai", stream=True)
def generate_story(topic):
    """
    You are a creative storyteller.
    """
    return f"Tell a short story about {topic}."

for chunk in generate_story("a brave knight"):
    print(chunk, end="", flush=True)
```

### Parameters

-   **Decorator Parameters:**

    -   `model`: The AI model to use.
    -   `provider`: The AI provider.
    -   `max_tokens`: (Optional) Maximum tokens for the response.
    -   `temperature`: (Optional) Controls randomness.
    -   `stream`: (Optional) If `True`, streams the response.
    -   `base_url`: (Optional) Custom API base URL.
    -   `reasoning_effort`: (Optional) Controls reasoning depth for supported models.

-   **Function Parameters:**
    -   Use function arguments to customize prompts dynamically.

## Command-Line Interface

The chat.py module includes a powerful command-line interface for quick AI interactions:

### Basic Usage

```bash
# Simple question with default model (gpt4.1)
python chat.py "What is the capital of France?"

# Use a specific model
python chat.py "Explain quantum computing" -m sonnet4

# Use multiple models for comparison
python chat.py "Write a haiku about code" -m gpt4.1 sonnet3.5 gemini-pro

# Stream responses
python chat.py "Tell me a story" --stream

# Custom system prompt
python chat.py "Solve this problem" --system "You are a mathematics tutor"

# Adjust model parameters
python chat.py "Be creative" --temperature 1.2 --max-tokens 500

# Use reasoning effort for supported models
python chat.py "Think step by step" -m o4-mini --reasoning-effort high
```

### Command-Line Options

-   `-m, --models`: Specify one or more models to use
-   `-s, --system`: Set the system prompt
-   `--max-tokens`: Maximum tokens for response (default: 4096)
-   `--temperature`: Controls randomness (default: 1)
-   `--stream`: Enable streaming responses
-   `--reasoning-effort`: Set reasoning effort for reasoning models (low/medium/high)

## Advanced Usage

### Custom Models and Providers

You can use custom or self-hosted models by specifying the `base_url` and `provider`:

This example uses LM Studio (which uses the same provider as OpenAI) and the hermes-3-llama-3.2-3b model.

```python
chat = Chat(
    model="hermes-3-llama-3.2-3b",
    system="You are a helpful AI assistant.",
    provider="openai",  # LM Studio is compatible with openai provider
    base_url="http://localhost:1234/v1",
)
```

It's also super easy to create your own provider by subclassing the `AIProvider` class and implementing the required abstract methods:

-   `create_client`: Set up the API client
-   `create_completion`: Generate completions with the model
-   `iter_chunks`: Extract text from streaming responses
-   `extract_response`: Extract text from non-streaming responses

### Maintaining Context

The `Chat` class maintains the conversation context:

```python
response = chat("What's the capital of France?")
print(response)  # Paris

response = chat("What's the weather like there?")
print(response)  # Assistant discusses weather in Paris

response = chat("Tell me about its famous landmarks")
print(response)  # Assistant describes Eiffel Tower, Louvre, etc.
```

## Testing

The project includes a comprehensive testing suite (`test.py`) that allows you to test models across different providers:

### Basic Testing

```bash
# List all available models
python test.py --list

# Test all models (requires API keys)
python test.py --all

# Test models from a specific provider
python test.py --provider openai
python test.py --provider anthropic --decorators

# Test specific models
python test.py --model sonnet3.5
python test.py --model gpt4.1 --model o4-mini

# Test local models (e.g., LM Studio)
python test.py --local --model hermes-3-llama-3.2-3b
python test.py --local --base-url http://localhost:1234/v1 --model your-local-model
```

### Test Options

-   `--all`: Test all available models
-   `--provider`: Test all models from a specific provider
-   `--model`: Test specific model(s)
-   `--local`: Test local models with custom base URL
-   `--list`: List all available models
-   `--decorators`: Also test @prompt decorators
-   `--verbose`: Enable verbose output
-   `--system`: Custom system prompt for testing

## Getting Started

-   Python 3.7+
-   Install required packages: `pip install openai anthropic google-genai`
-   Set up API keys as environment variables:
    ```bash
    export OPENAI_API_KEY='your-key'
    export ANTHROPIC_API_KEY='your-key'
    export GEMINI_API_KEY='your-key'
    ```
    Or pass keys directly when initializing `Chat`

## Notes

-   **Providers Supported:** Currently supports OpenAI, Anthropic, and Google models:
    -   OpenAI: O4-mini, O3, GPT-4.1, and GPT-4.1-mini
    -   Anthropic: Claude Opus 4, Sonnet 4, Sonnet 3.7, and Sonnet 3.5
    -   Google: Gemini 2.5 Pro and Gemini 2.5 Flash
-   **Context:** The `Chat` class maintains context across messages. The `@prompt` decorator does not maintain context between calls (yet).
-   **Reasoning Effort:** Reasoning models (O4-mini, O3) support adjustable reasoning effort levels ("high", "medium", "low").
-   **Model Resolution:** The library automatically resolves short model names (like "sonnet3.5") to full model identifiers (like "claude-3-5-sonnet-20241022").

## Examples

Here are some practical examples of using the library:

### Multi-Model Comparison

```bash
# Compare responses across different providers
python chat.py "Explain the concept of recursion" -m gpt4.1 sonnet3.5 gemini-pro
```

### Creative Writing with Streaming

```python
from chat import Chat

chat = Chat(
    model="sonnet3.5",
    system="You are a creative writer with a poetic style.",
    temperature=1.2
)

print("Story begins:")
for chunk in chat("Write a short story about a time traveler", stream=True):
    print(chunk, end="", flush=True)
```

### Code Review Assistant

```python
from chat import prompt

@prompt(
    model="gpt4.1",
    system="You are an expert code reviewer. Provide constructive feedback.",
    temperature=0.3
)
def review_code(code):
    return f"Please review this code:\n\n{code}"

feedback = review_code("""
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
""")
print(feedback)
```

### Local Model Integration

```python
# Using a local model through LM Studio
local_chat = Chat(
    model="llama-3.2-3b-instruct",
    system="You are a helpful coding assistant.",
    provider="openai",
    base_url="http://localhost:1234/v1"
)

response = local_chat("How do I implement a binary search?")
print(response)
```
