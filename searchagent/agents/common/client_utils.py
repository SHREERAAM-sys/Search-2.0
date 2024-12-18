# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import uuid
from enum import Enum
from typing import Any, List, Literal, Optional, Union

from llama_stack_client.types import SamplingParams
from llama_stack_client.types.agent_create_params import (
    AgentConfig,
    AgentConfigToolCodeInterpreterToolDefinition,
    AgentConfigToolMemoryToolDefinition,
    AgentConfigToolPhotogenToolDefinition,
)
from pydantic import BaseModel, Field
from searchagent.agents.common.custom_tools import CustomTool
from searchagent.agents.common.types import AgentWithCustomToolExecutor
from termcolor import cprint

ToolDefinition = Union[
    AgentConfigToolMemoryToolDefinition,
    AgentConfigToolPhotogenToolDefinition,
    AgentConfigToolCodeInterpreterToolDefinition,
]


class AttachmentBehavior(Enum):
    rag = "rag"
    code_interpreter = "code_interpreter"
    auto = "auto"


class QuickToolConfig(BaseModel):
    tool_definitions: List[Any] = Field(default_factory=list)
    custom_tools: List[CustomTool] = Field(default_factory=list)
    prompt_format: Literal["json", "function_tag", "python_list"] = "json"
    # use this to control whether you want the model to read file / write code to
    # process them, or you want to "RAG" them beforehand (aka chunk and add to index)
    attachment_behavior: Optional[str] = None
    # if you have a memory bank already pre-populated, specify it here
    memory_bank_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


def enable_memory_tool(cfg: QuickToolConfig) -> bool:
    if cfg.memory_bank_id:
        return True
    return (
        cfg.attachment_behavior
        and cfg.attachment_behavior != AttachmentBehavior.code_interpreter.value
    )


# This is a utility function; it does not provide all bells and whistles
# you can get from the underlying Agents API. Any limitations should
# ideally be resolved by making another well-scoped utility function instead
# of adding complex options here.
async def make_agent_config_with_custom_tools(
    model: str = "Llama3.1-8B-Instruct",
    disable_safety: bool = True,
    tool_config: QuickToolConfig = QuickToolConfig(),
) -> AgentConfig:
    input_shields = []
    output_shields = []
    if not disable_safety:
        for t in tool_config.tool_definitions:
            t["input_shields"] = ["llama_guard"]
            t["output_shields"] = ["llama_guard"]

        input_shields = ["llama_guard"]
        output_shields = ["llama_guard"]

    # ensure code interpreter is enabled if attachments need it
    tool_choice = "auto"
    if (
        tool_config.attachment_behavior
        and tool_config.attachment_behavior == AttachmentBehavior.code_interpreter.value
    ):
        if not any(
            t["type"] == "code_interpreter" for t in tool_config.tool_definitions
        ):
            tool_config.tool_definitions.append(
                AgentConfigToolCodeInterpreterToolDefinition(type="code_interpreter")
            )

        tool_choice = "required"

    # switch to memory
    if enable_memory_tool(tool_config):
        bank_configs = []
        if tool_config.memory_bank_id:
            bank_configs.append(
                {
                    "bank_id": tool_config.memory_bank_id,
                    "type": "vector",
                }
            )
        tool_config.tool_definitions.append(
            AgentConfigToolMemoryToolDefinition(
                type="memory",
                memory_bank_configs=bank_configs,
                query_generator_config={
                    "type": "default",
                    "sep": " ",
                },
                max_tokens_in_context=4096,
                max_chunks=10,
            )
        )

    tool_config.tool_definitions += [
        t.get_tool_definition() for t in tool_config.custom_tools
    ]

    agent_config = AgentConfig(
        model=model,
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(
            strategy="greedy",
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1,
            max_tokens=500,
            top_k=1,
        ).model_dump(),
        tools=tool_config.tool_definitions.copy(),
        tool_choice=tool_choice,
        tool_prompt_format=tool_config.prompt_format,
        input_shields=input_shields,
        output_shields=output_shields,
        enable_session_persistence=False,
        max_infer_iters=5,
    )
    return agent_config


async def get_agent_with_custom_tools(
    name: str,
    agent_config: AgentConfig,
    custom_tools: List[CustomTool],
):
    from searchagent.agents.app_context import client

    create_response = client.agents.create(
        agent_config=agent_config,
    )
    agent_id = create_response.agent_id
    cprint(f"> created agents with agent_id={agent_id}", "green")

    name = f"Session-{uuid.uuid4()}"
    session_response = client.agents.session.create(
        agent_id=agent_id,
        session_name=name,
    )
    session_id = session_response.session_id

    return AgentWithCustomToolExecutor(
        name=name,
        agent_id=agent_id,
        session_id=session_id,
        agent_config=agent_config,
        custom_tools=custom_tools,
    )
