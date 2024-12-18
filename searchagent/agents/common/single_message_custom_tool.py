import json
from abc import abstractmethod
from typing import List, Union

from llama_stack_client.types import CompletionMessage, ToolResponseMessage
from loguru import logger
from searchagent.agents.common.custom_tools import CustomTool
from searchagent.agents.common.types import AgentWithCustomToolExecutor


class SingleMessageCustomTool(CustomTool):
    """
    Helper class to handle custom tools that take a single message
    Extending this class and implementing the `run_impl` method will
    allow for the tool be called by the model and the necessary plumbing.
    """

    async def run(
        self, messages: List[CompletionMessage]
    ) -> Union[AgentWithCustomToolExecutor, ToolResponseMessage]:

        assert len(messages) == 1, "Expected single message"

        message = messages[0]

        tool_call = message.tool_calls[0]
        tool_name = tool_call.tool_name

        try:
            response = await self.run_impl(**tool_call.arguments)

            if "agent_id" in response:
                response["name"] = tool_name
                return AgentWithCustomToolExecutor.from_dict(response)
            else:
                response_str = json.dumps(response)
                return ToolResponseMessage(
                    call_id=tool_call.call_id,
                    tool_name=tool_name,
                    content=response_str,
                    role="ipython",
                )

        except Exception as e:
            logger.error(f"Error when running tool: {e}")

    @abstractmethod
    async def run_impl(self, *args, **kwargs):
        raise NotImplementedError()
