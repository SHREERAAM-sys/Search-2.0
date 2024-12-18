from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.single_message_custom_tool import SingleMessageCustomTool


class Index(SingleMessageCustomTool):
    def __init__(self):
        pass

    def get_name(self) -> str:
        return "index"

    def get_description(self) -> str:
        return "Index new or updated files"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return NotImplementedError

    async def run_impl(self):
        return NotImplementedError
