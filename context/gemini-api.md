**Context: Text Generation with the Gemini API**

---

**Capabilities Overview**
The Gemini API enables text generation from multiple modalities (text, images, audio, video). All models in the Gemini family support text output. SDKs are available in Python, JavaScript, Go, REST, and Apps Script.

---

**Basic Text Generation Example (Python):**

```python
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["How does AI work?"]
)
print(response.text)
```

---

**System Instructions and Configuration**
Behavior can be controlled using system instructions via `GenerateContentConfig`.

_Example:_

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(system_instruction="You are a cat. Your name is Neko."),
    contents="Hello there"
)
print(response.text)
```

_Parameters such as output length and temperature can also be configured:_

```python
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"],
    config=types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1
    )
)
print(response.text)
```

For a complete list of configurable parameters, refer to the API documentation.

---

**Multimodal Inputs**
Text prompts can be combined with images or other media files.

_Example with image input:_

```python
from PIL import Image
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

image = Image.open("/path/to/organ.png")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, "Tell me about this instrument"]
)
print(response.text)
```

Supports document, video, and audio input/understanding.

---

**Streaming Responses**
To receive outputs incrementally as they're generated, use streaming:

```python
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

response = client.models.generate_content_stream(
    model="gemini-2.0-flash",
    contents=["Explain how AI works"]
)
for chunk in response:
    print(chunk.text, end="")
```

---

**Multi-turn Conversations (Chat)**
SDKs support chat sessions, allowing the maintenance of conversation history. The full conversation is sent with each turn.

_Example:_

```python
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")
chat = client.chats.create(model="gemini-2.0-flash")

response = chat.send_message("I have 2 dogs in my house.")
print(response.text)

response = chat.send_message("How many paws are in my house?")
print(response.text)

for message in chat.get_history():
    print(f'role - {message.role}: {message.parts[0].text}')
```

Streaming can also be used in multi-turn conversations.

---

**Best Practices and Prompting Tips**

-   Zero-shot prompts suffice for basic generation.
-   Use `system_instruction` for guiding behavior.
-   Provide few-shot examples for tailored outputs.
-   Consider fine-tuning for advanced cases.
-   For structured outputs (e.g. JSON), consult the structured output guide.

---

**References**

-   Prompt engineering guide
-   Structured output guide
-   API reference for model/parameter details

---

**Note:** For model details and capability comparisons, see the Gemini Models page.
