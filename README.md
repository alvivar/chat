# Usage

This library provides a unified interface to interact with various AI language models across different providers like OpenAI and Anthropic. It offers:

-   A `Chat` class for managing conversations with AI models.
-   A `@prompt` decorator to simplify prompt creation.
-   Caching capabilities to store and reuse responses.

## Prerequisites

-   Python 3.7 or higher.
-   API keys for the AI providers you plan to use (e.g., OpenAI, Anthropic).
-   Installation of necessary packages:
    ```bash
    pip install openai anthropic
    ```

## Setting Up API Keys

Ensure your API keys are set as environment variables:

```bash
export OPENAI_API_KEY='your-openai-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

Alternatively, pass the API key directly when initializing the `Chat` class.

## Using the `Chat` Class

The `Chat` class allows you to interact with AI models in a conversational manner.

### Initialization

```python
from chat import Chat

chat = Chat(
    model="gpt-4",
    system="You are a helpful assistant.",
    provider="openai",  # or "anthropic"
    max_tokens=512,
    temperature=0.8,
    use_cache=True  # Enables caching of responses
)
```

-   **Parameters:**
    -   `model`: The AI model to use (e.g., `"gpt-4"`, `"claude-3-haiku-20240307"`).
    -   `system`: The system prompt that defines the assistant's behavior.
    -   `provider`: The AI provider (`"openai"` or `"anthropic"`).
    -   `max_tokens`: Maximum number of tokens for the response.
    -   `temperature`: Controls the randomness of the output.
    -   `base_url`: (Optional) Custom API base URL.
    -   `api_key`: (Optional) API key for the provider.
    -   `use_cache`: (Optional) Whether to cache responses.

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

### Ignoring Cache

To bypass the cache for a specific query:

```python
response = chat("What is the weather today?", ignore_cache=True)
print(response)
```

## Using the `@prompt` Decorator

The `@prompt` decorator simplifies the process of sending prompts to the AI model by converting a function's return value into a prompt.

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

-   **Function Parameters:**
    -   Use function arguments to customize prompts dynamically.

## Advanced Usage

### Custom Models and Providers

You can use custom or self-hosted models by specifying the `base_url` and `provider`:

```python
chat = Chat(
    model="custom-model",
    system="You are a helpful assistant.",
    provider="openai",
    base_url="http://localhost:1234/v1",
    api_key="your-api-key"
)
```

### Maintaining Context

The `Chat` class maintains the conversation context:

```python
response = chat("What's the capital of France?")
print(response)  # Assistant provides the capital.

response = chat("What is its population?")
print(response)  # Assistant uses previous context to answer.
```

### Error Handling

Wrap your calls in try-except blocks to handle exceptions gracefully:

```python
try:
    response = chat("Tell me a joke.")
    print(response)
except Exception as e:
    print(f"Error: {e}")
```

## Complete Examples

### Using OpenAI's GPT-4 Model

```python
from chat import Chat

chat = Chat(
    model="gpt-4",
    system="You are a knowledgeable assistant.",
    provider="openai"
)

response = chat("Who won the Nobel Prize in Physics in 2020?")
print(response)
```

### Using Anthropic's Claude Model

```python
from chat import Chat

chat = Chat(
    model="claude-3-haiku-20240307",
    system="You are a poetic assistant.",
    provider="anthropic"
)

response = chat("Write a haiku about the ocean.")
print(response)
```

### Using the `@prompt` Decorator with Parameters

```python
from chat import prompt

@prompt(model="gpt-4", provider="openai")
def summarize_article(url):
    """
    You are a summarization assistant.
    """
    return f"Summarize the main points of the article at {url}."

response = summarize_article("https://example.com/article")
print(response)
```

### Streaming Large Responses

```python
from chat import Chat

chat = Chat(
    model="gpt-4",
    system="You are an assistant who provides detailed explanations.",
    provider="openai"
)

for chunk in chat("Explain the theory of general relativity.", stream=True):
    print(chunk, end="", flush=True)
```

## Notes

-   **Caching:** Responses are cached in `chat_cache.json` to reduce API calls.
-   **Providers Supported:** Currently supports OpenAI and Anthropic. Support for more providers is planned.
-   **Context:** The `Chat` class maintains context across messages. The `@prompt` decorator does not maintain context between calls.

## Testing and Examples

To test the functionality, you can run the script provided in the `if __name__ == "__main__":` block. It includes tests for OpenAI, Anthropic, and custom providers, as well as examples using the `@prompt` decorator.
