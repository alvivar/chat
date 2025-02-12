# TODO

-   Add support for Google provider
-   Support for prompt caching (Anthropic, OpenAI)
-   Check that consecutive calls to the decorator maintain the context as expected
-   Add support for Groq provider
-   Ensure proper handling of exceptions throughout the code

## Done

-   [x] Implement and test the decorator functionality across all supported providers:
    -   [x] OpenAI
        -   [x] Support for o3-mini, 4o, and 4o-mini models
        -   [x] Reasoning effort parameter for o3-mini
        -   [x] Custom base URL support for self-hosted models
    -   [x] Anthropic
        -   [x] Support for Claude 3.5 Sonnet and Haiku models
        -   [x] Streaming responses with text_stream
        -   [x] System prompts integration
-   [x] Core decorator features:
    -   [x] System prompt from function docstring
    -   [x] Configurable parameters (max_tokens, temperature, etc.)
    -   [x] Streaming support
    -   [x] Error handling and recovery
    -   [x] Custom provider and base URL support
