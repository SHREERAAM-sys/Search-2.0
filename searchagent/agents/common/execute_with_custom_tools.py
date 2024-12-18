# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
from typing import AsyncGenerator, List, Optional, Union

from llama_stack_client.types import Attachment, ToolResponseMessage, UserMessage
from searchagent.agents.app_context import client
from searchagent.agents.common.custom_tools import CustomTool, Message
from searchagent.agents.common.types import (
    AgentWithCustomToolExecutor,
    HandoffAgentType,
    ToolType,
)
from searchagent.agents.tools.handoffs import (
    TransferBackToTriageAgent,
    TransferToEmbedAgent,
    TransferToFileRetrievalAgent,
    TransferToIndexAgent,
    TransferToSyncAgent,
)
from searchagent.agents.tools.routines import Embed, FileMonitor, Index, PDFSearchTool

tools_mapping = {
    ToolType.pdf_search.value: PDFSearchTool,
    ToolType.file_monitor.value: FileMonitor,
    ToolType.embed.value: Embed,
    ToolType.index.value: Index,
    HandoffAgentType.transfer_back_to_triage_agent.value: TransferBackToTriageAgent,
    HandoffAgentType.transfer_to_file_retrieval_agent.value: TransferToFileRetrievalAgent,
    HandoffAgentType.transfer_to_embed_agent.value: TransferToEmbedAgent,
    HandoffAgentType.transfer_to_index_agent.value: TransferToIndexAgent,
    HandoffAgentType.transfer_to_sync_agent.value: TransferToSyncAgent,
}


async def execute_turn(
    agent: AgentWithCustomToolExecutor,
    messages: List[Union[UserMessage, ToolResponseMessage]],
    attachments: Optional[List[Attachment]] = None,
    stream: bool = True,
) -> AsyncGenerator:
    current_agent = agent

    tools_dict = {t.get_name(): t for t in agent.custom_tools}

    current_messages = messages.copy()

    while True:
        response = client.agents.turn.create(
            agent_id=agent.agent_id,
            session_id=agent.session_id,
            messages=current_messages,
            attachments=attachments,
            stream=True,
        )
        turn = None
        for chunk in response:
            if chunk.event.payload.event_type != "turn_complete":
                yield chunk
            else:
                turn = chunk.event.payload.turn

        message = turn.output_message
        message.role = "user"
        messages.append(message)

        if len(message.tool_calls) == 0:
            yield chunk
            return

        if message.stop_reason == "out_of_tokens":
            yield chunk
            return

        for tool_call in message.tool_calls:
            if tool_call.tool_name not in tools_dict:
                m = ToolResponseMessage(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    content=f"Unknown tool `{tool_call.tool_name}` was called. Try again with something else",
                    role="ipython",
                )
                next_message = m
            else:
                tool = tools_dict[tool_call.tool_name]
                result = await execute_custom_tool(tool, message)

                if isinstance(result, AgentWithCustomToolExecutor):
                    current_agent = result
                    result = (
                        f"Transferred to {current_agent.name}. Adopt persona immediately. "
                        "Use necessary tools to complete the user request. "
                        "If a transfer is not complete, CONTINUE CALLING THE TOOL. "
                        "If the request is fulfilled, you can leave tool_calls empty."
                    )

                result_message = {
                    "agent": {
                        "name": current_agent.name,
                        "id": current_agent.agent_id,
                        "custom_tools": [
                            tool.get_name() for tool in current_agent.custom_tools
                        ],
                    },
                    "message": (
                        json.dumps(result.model_dump())
                        + " | **tool execution is completed!**"
                        if isinstance(result, ToolResponseMessage)
                        else result
                    ),
                }

                next_message = ToolResponseMessage(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    content=json.dumps(result_message),
                    role="ipython",
                )
                messages.append(next_message)

            yield next_message
            current_messages = [next_message]


async def execute_custom_tool(
    tool: CustomTool, message: Message
) -> Union[AgentWithCustomToolExecutor, ToolResponseMessage]:
    return await tool.run([message])
