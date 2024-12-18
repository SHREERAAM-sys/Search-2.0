from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.single_message_custom_tool import SingleMessageCustomTool
from searchagent.sync_manager.base import Monitor


class FileMonitor(SingleMessageCustomTool):
    def __init__(self, input_dir=".") -> None:
        self.input_dir = input_dir
        self.monitor = Monitor(input_dir)

    def get_name(self) -> str:
        return "file_monitor"

    def get_description(self) -> str:
        return "Continuously monitor your files for any changes"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return NotImplementedError

    async def run_impl(self):
        return await self.monitor.run()
