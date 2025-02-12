from chat import prompt
import asyncio
import concurrent.futures


cloud = {
    "model": "4o-mini",
    "stream": True,
}


local = {
    "model": "deepseek-r1-distill-qwen-7b",
    "provider": "openai",
    "base_url": "http://localhost:1234/v1",
    "stream": True,
}

current = cloud


@prompt(**current, temperature=0.9)
def define_system_rules():
    """Define the core rules and mechanics of a fictional system or world."""
    return "Create 3-5 fundamental rules that govern this fictional world/system."


@prompt(**current, temperature=0.9)
def create_character_profiles(rules):
    """Generate character profiles that exist within the defined system."""
    return f"Based on these system rules:\n{rules}\n\nCreate 2-3 character profiles with unique traits and motivations."


@prompt(**current, temperature=0.9)
def define_character_abilities(rules):
    """Define possible actions and abilities for characters within the system."""
    return f"Given these system rules:\n{rules}\n\nList 4-6 possible actions/abilities characters could have."


@prompt(**current, temperature=0.9)
def create_character_interactions(characters, abilities):
    """Create scenarios showing character interactions using their abilities."""
    return f"Using these characters:\n{characters}\n\nAnd these abilities:\n{abilities}\n\nCreate 2-3 interaction scenarios."


def stream(prompt_function, *args):
    print("\n")
    print("─" * 80)
    response = ""
    for token in prompt_function(*args):
        print(token, end="", flush=True)
        response += token
    print("\n" + "─" * 80 + "\n")
    return response


def to_file(data: str, filename: str) -> bool:
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(data)
        return True
    except IOError as e:
        return False


async def parallel_stream(prompt_function, *args):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, stream, prompt_function, *args)


async def main():
    rules = stream(define_system_rules)

    characters, abilities = await asyncio.gather(
        parallel_stream(create_character_profiles, rules),
        parallel_stream(define_character_abilities, rules),
    )

    interactions = stream(create_character_interactions, characters, abilities)

    return rules, characters, abilities, interactions


if __name__ == "__main__":
    asyncio.run(main())
