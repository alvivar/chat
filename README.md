# chat.py

A simple, unified API for interacting with AI language models from multiple providers (OpenAI, Anthropic, and Google). It offers:

-   A `Chat` class for managing conversations with AI models.
-   A `@prompt` decorator to simplify prompt creation.

## Using the `Chat` Class

The `Chat` class allows you to interact with AI models in a conversational manner.

### Initialization

```python
from chat import Chat

chat = Chat(
    model="sonnet",  # or "o3-mini", "gemini-pro", "4o", "haiku", "gemini-flash", "4o-mini"
    system="You are a helpful assistant.",
    provider="anthropic",  # "openai" or "google"
    max_tokens=4096,
    temperature=0.8,
    reasoning_effort="high"  # Optional, for reasoning models
)
```

-   **Parameters:**
    -   `model`: The AI model to use (supported models include):
        -   Anthropic: `"sonnet"` (claude-3-5-sonnet-20241022), `"haiku"` (claude-3-5-haiku-20241022)
        -   OpenAI: `"o3-mini"`, `"4o"`, `"4o-mini"`
        -   Google: `"gemini-pro"` (gemini-2.0-pro-exp-02-05), `"gemini-flash"` (gemini-2.0-flash-001)
    -   `system`: The system prompt that defines the assistant's behavior.
    -   `provider`: The AI provider (`"openai"`, `"anthropic"`, or `"google"`).
    -   `max_tokens`: Maximum number of tokens for the response.
    -   `temperature`: Controls the randomness of the output.
    -   `base_url`: (Optional) Custom API base URL.
    -   `api_key`: (Optional) API key for the provider.
    -   `reasoning_effort`: (Optional) Controls reasoning depth for supported models.

### Sending Messages

```python
# Single message
response = chat("Hello, how can I improve my coding skills?")
print(response)

# Conversation continues
response = chat("Can you recommend resources for learning Python?")
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

@prompt(model="gpt-4", provider="openai")
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
@prompt(model="gpt-4", provider="openai", stream=True)
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

## Advanced Usage

### Custom Models and Providers

You can use custom or self-hosted models by specifying the `base_url` and `provider`:

```python
chat = Chat(
    model="custom-model",
    system="You are a helpful AI assistant.",
    provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="your-api-key"
)
```

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

## Prerequisites

-   Python 3.7 or higher.
-   API keys for the AI providers you plan to use (e.g., OpenAI, Anthropic, Google).
-   Installation of necessary packages:

    ```bash
    pip install openai anthropic google-genai
    ```

## Setting Up API Keys

Ensure your API keys are set as environment variables:

```bash
export OPENAI_API_KEY='your-openai-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
export GOOGLE_API_KEY='your-google-api-key'
```

Alternatively, pass the API key directly when initializing the `Chat` class.

## Notes

-   **Providers Supported:** Currently supports OpenAI, Anthropic, and Google models:
    -   Anthropic: Claude 3.5 Sonnet and Haiku
    -   OpenAI: O3-mini, 4O, and 4O-mini
    -   Google: Gemini Pro and Gemini Flash
-   **Context:** The `Chat` class maintains context across messages. The `@prompt` decorator does not maintain context between calls (yet).
-   **Reasoning Effort:** Some models support adjustable reasoning effort levels ("high", "medium", "low").

## Testing and Examples

To test the functionality, you can run the script provided in the `if __name__ == "__main__":` block. It includes basic tests for OpenAI, Anthropic, and custom providers, as well as examples using the `@prompt` decorator.
