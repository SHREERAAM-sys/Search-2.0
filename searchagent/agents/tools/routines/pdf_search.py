from pathlib import Path
from typing import Any, Dict, Union

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.custom_tools import CustomTool
from searchagent.agents.common.single_message_custom_tool import SingleMessageCustomTool
from searchagent.colpali.base import ColPaliRag


class PDFSearchTool(SingleMessageCustomTool):
    def __init__(self, input_dir: Union[Path, str]):
        super().__init__()
        self.input_dir = input_dir
        self.colpali = ColPaliRag(input_dir)

    def get_name(self) -> str:
        return "pdf_search"

    def get_description(self) -> str:
        return "Search PDF files for a given query"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "query": ToolParamDefinitionParam(
                param_type="str",
                description="The search query used to locate a file",
                required=True,
            )
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], input_dir: str) -> "PDFSearchTool":
        instance = cls(input_dir=input_dir)

        CustomTool.from_dict(data)

        return instance

    async def run_impl(self, query: str):
        return await self.colpali.search(query)
