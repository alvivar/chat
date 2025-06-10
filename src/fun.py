from src.chat import prompt
import asyncio
import concurrent.futures
import os


cloud = {"stream": True}

local = {
    "model": "deepseek-r1-distill-qwen-7b",
    "provider": "openai",
    "base_url": "http://localhost:1234/v1",
    "stream": True,
}


# Lower temperature for consistent world-building
@prompt(**cloud, model="o3-mini", temperature=0.6)
def define_system_rules():
    """Define the core rules and mechanics of a fictional system or world."""
    return """Create 3-5 fundamental rules that govern this fictional world/system. The rules should be:
1. Internally consistent and logically connected
2. Specific enough to create interesting constraints
3. General enough to allow creative possibilities
4. Free of contradictions or paradoxes"""


# Balanced temperature for creative but grounded characters
@prompt(**cloud, model="o3-mini", temperature=0.7)
def create_character_profiles(rules):
    """Generate character profiles that exist within the defined system."""
    return f"""Based on these system rules:
{rules}

Create 2-3 detailed character profiles including:
- Distinct personality traits and motivations
- How they specifically interact with the system rules
- Their goals and conflicts within the world
- Relationships or connections to other potential characters"""


# Moderate temperature for well-defined abilities
@prompt(**cloud, model="o3-mini", temperature=0.65)
def define_character_abilities(rules):
    """Define possible actions and abilities for characters within the system."""
    return f"""Given these system rules:
{rules}

Define 4-6 possible abilities/actions that:
1. Naturally emerge from or connect to the system rules
2. Have clear limitations and costs
3. Create interesting strategic choices
4. Could be used in multiple ways"""


# Higher temperature for dynamic interactions
@prompt(**cloud, model="o3-mini", temperature=0.8)
def create_character_interactions(characters, abilities):
    """Create scenarios showing character interactions using their abilities."""
    return f"""Using these characters:
{characters}

And these abilities:
{abilities}

Create 2-3 interaction scenarios that:
1. Demonstrate creative use of abilities
2. Reveal character motivations and relationships
3. Create dramatic tension or conflict
4. Remain consistent with the established rules"""


def _get_translation_prompt(lang, text):
    return f"""Translate this text into {lang} with the following artistic considerations:

{text}

Guidelines for a masterful translation:
- Craft flowing, poetic language that captures the original's spirit
- Preserve literary devices, metaphors and imagery
- Maintain the author's unique voice and stylistic flourishes
- Keep character names and key terms authentic
- Adapt cultural references thoughtfully and gracefully
- Ensure the translation reads as polished, published literature
- Balance faithfulness to source with artistic expression"""


# Higher temperature for creative, literary translations
@prompt(**cloud, model="sonnet", temperature=0.7)
def sonnet(lang, text):
    """Transform the given text into elegant, literary prose in the target language while preserving the essence and artistry of the original."""
    return _get_translation_prompt(lang, text)


@prompt(**cloud, model="4o", temperature=0.7)
def gpt4o(lang, text):
    """Transform the given text into elegant, literary prose in the target language while preserving the essence and artistry of the original."""
    return _get_translation_prompt(lang, text)


@prompt(**cloud, model="gemini-pro", temperature=0.7)
def gemini(lang, text):
    """Transform the given text into elegant, literary prose in the target language while preserving the essence and artistry of the original."""
    return _get_translation_prompt(lang, text)


def stream(prompt_function, *args):
    print("\n")
    print("─" * 80)
    response = ""
    for token in prompt_function(*args):
        print(token, end="", flush=True)
        response += token
    print("\n" + "─" * 80 + "\n")
    return response


def dump(data: str, filename: str) -> bool:
    """Write data to a file with error handling."""
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(data)
        return True
    except IOError as e:
        print(f"Error writing to {filename}: {e}")
        return False


async def async_stream(prompt_function, *args):
    """Execute stream function in a separate thread to allow concurrent execution."""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, stream, prompt_function, *args)


async def main():
    # Setup output directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("fun", exist_ok=True)

    # Generate world rules
    rules = stream(define_system_rules)
    dump(rules, "fun/rules.md")

    # Async generation of characters and abilities
    characters, abilities = await asyncio.gather(
        async_stream(create_character_profiles, rules),
        async_stream(define_character_abilities, rules),
    )
    dump(characters, "fun/characters.md")
    dump(abilities, "fun/abilities.md")

    # Create interactions and translations
    interactions = stream(create_character_interactions, characters, abilities)
    dump(interactions, "fun/interactions.md")

    # Async generation of translations
    sonnets, gpt4os, geminis = await asyncio.gather(
        async_stream(sonnet, "spanish", interactions),
        async_stream(gpt4o, "spanish", interactions),
        async_stream(gemini, "spanish", interactions),
    )
    dump(sonnets, "fun/sonnet.md")
    dump(gpt4os, "fun/gpt4o.md")
    dump(geminis, "fun/gemini.md")

    return rules, characters, abilities, interactions, sonnets, gpt4os, geminis


if __name__ == "__main__":
    asyncio.run(main())
