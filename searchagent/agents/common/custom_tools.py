# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from abc import abstractmethod
from typing import Any, Dict, List, Union

from llama_stack_client.types import ToolResponseMessage, UserMessage
from llama_stack_client.types.agent_create_params import (
    AgentConfigToolFunctionCallToolDefinition,
)
from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from typing_extensions import TypeAlias

Message: TypeAlias = Union[UserMessage, ToolResponseMessage]


class CustomTool:
    def __init__(self):
        self._name = ""
        self._description = ""
        self._params_definition = {}

    """
    Developers can define their custom tools that models can use
    by extending this class.

    Developers need to provide
        - name
        - description
        - params_definition
        - implement tool's behavior in `run_impl` method

    NOTE: The return of the `run` method needs to be json serializable
    """

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        raise NotImplementedError

    def get_instruction_string(self) -> str:
        return f"Use the function '{self.get_name()}' to: {self.get_description()}"

    def parameters_for_system_prompt(self) -> str:
        return json.dumps(
            {
                "name": self.get_name(),
                "description": self.get_description(),
                "parameters": {
                    name: definition.__dict__
                    for name, definition in self.get_params_definition().items()
                },
            }
        )

    def get_tool_definition(self) -> AgentConfigToolFunctionCallToolDefinition:
        return AgentConfigToolFunctionCallToolDefinition(
            function_name=self.get_name(),
            description=self.get_description(),
            parameters=self.get_params_definition(),
            type="function_call",
        )

    def set_name(self, name: str):
        self._name = name

    def set_description(self, description: str):
        self._description = description

    def set_params_definition(
        self, params_definition: Dict[str, ToolParamDefinitionParam]
    ):
        self._params_definition = params_definition

    @abstractmethod
    async def run(self, messages: List[Message]) -> List[Message]:
        raise NotImplementedError

    def to_dict(self):
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "parameters": {
                name: param for name, param in self.get_params_definition().items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomTool":
        name = data["name"]
        description = data["description"]
        params_definition = {
            param_name: ToolParamDefinitionParam(**param_dict)
            for param_name, param_dict in data["parameters"].items()
        }

        instance = cls()
        instance.set_name(name)
        instance.set_description(description)
        instance.set_params_definition(params_definition)

        return instance
