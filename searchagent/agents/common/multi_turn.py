# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
from typing import List, Optional

from llama_stack_client.types import Attachment, ToolResponseMessage, UserMessage
from pydantic import BaseModel
from searchagent.agents.common.execute_with_custom_tools import (
    execute_turn,
    tools_mapping,
)
from searchagent.agents.common.types import AgentWithCustomToolExecutor
from termcolor import cprint
import json


class UserTurnInput(BaseModel):
    message: UserMessage
    attachments: Optional[List[Attachment]] = None


def prompt_to_turn(
    content: str, attachments: Optional[List[Attachment]] = None
) -> UserTurnInput:
    return UserTurnInput(
        message=UserMessage(content=content, role="user"), attachments=attachments
    )


async def execute_turns(
    agent: AgentWithCustomToolExecutor,
    turn_inputs: List[UserTurnInput],
):
    current_agent = agent

    messages = []

    while len(turn_inputs) > 0:
        turn = turn_inputs.pop(0)
        messages.append(turn.message)

        iterator = execute_turn(
            current_agent,
            messages,
            turn.attachments,
        )
        cprint(f"User> {turn.message.content}", color="white", attrs=["bold"])
        async for log in iterator:
            if isinstance(log, ToolResponseMessage):
                agent_info = json.loads(log.content)
                name = agent_info["agent"]["name"]
                agent_dict = await tools_mapping[name]().run_impl()
                agent_dict["name"] = name
                agent = AgentWithCustomToolExecutor.from_dict(agent_dict)

                if current_agent != agent:
                    current_agent = agent
                    turn_inputs.append(prompt_to_turn(agent_info["message"]))

                print(agent.name)
