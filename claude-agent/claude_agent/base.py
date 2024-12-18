import os
import sys
import time

from anthropic import Anthropic
from claude_agent.tools import (
    ToolNames,
    colpali_delete_file,
    colpali_embed,
    colpali_get_all_files,
    colpali_search,
    tools,
    code_search,
    code_search_embed,
)
from dotenv import load_dotenv
from rich import print

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL_NAME = "claude-3-5-sonnet-20241022"


def process_tool_call(tool_name, tool_input):
    if tool_name == ToolNames.colpali_search:
        return colpali_search(
            tool_input["query"],
            tool_input["top_k"],
            tool_input["email"],
        )

    elif tool_name == ToolNames.colpali_embed:
        return colpali_embed(tool_input["filepaths"])

    elif tool_name == ToolNames.colpali_delete_file:
        return colpali_delete_file(tool_input["id"])

    elif tool_name == ToolNames.colpali_get_all_files:
        return colpali_get_all_files()
    elif tool_name == ToolNames.code_search:
        return code_search(tool_input["query"])
    elif tool_name == ToolNames.code_search_embed:
        return code_search_embed(tool_input["filepaths"])

    else:
        raise ValueError(f"Unsupported tool: {tool_name}")


def chat_with_claude(user_message, verbose: bool):
    start_time = time.time()

    if verbose:
        print(f"\n{'=' * 50}\nUser Message: {user_message}\n{'=' * 50}")

    message = client.beta.prompt_caching.messages.create(
        system="You will use the email colpalisearch@gmail.com when using the colpali_search tool."
        "You will patiently wait for the result to return.",
        model=MODEL_NAME,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
        extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
    )

    if verbose:
        print("\nInitial Response:")
        print(f"Stop Reason: {message.stop_reason}")
        print(f"Content: {message.content}")

    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        if verbose:
            print(f"\nTool Used: {tool_name}")
            print(f"Tool Input: {tool_input}")

        tool_result = process_tool_call(tool_name, tool_input)

        if verbose:
            print(f"Tool Result: {tool_result}")

        response = client.beta.prompt_caching.messages.create(
            system="Your response will be concise and straight to the point. It will end as a statement",
            model=MODEL_NAME,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result,
                        }
                    ],
                },
            ],
            tools=tools,
            extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
        )
    else:
        response = message

    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )

    end_time = time.time()

    if verbose:
        print(response.content)
        print(f"\nFinal Response: {final_response}")

        print(f"Cached API call time: {end_time - start_time:.2f} seconds")
        print(f"Cached API call input tokens: {response.usage.input_tokens}")
        print(f"Cached API call output tokens: {response.usage.output_tokens}")
    else:
        print(f"\nFinal Response: {final_response}")

    return final_response
