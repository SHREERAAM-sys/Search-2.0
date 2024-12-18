from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.single_message_custom_tool import SingleMessageCustomTool
from searchagent.agents.common.types import AgentType, custom_tool_handoff_agent_params


class TransferBackToTriageAgent(SingleMessageCustomTool):
    def get_name(self) -> str:
        return "transfer_back_to_triage_agent"

    def get_description(self) -> str:
        return "Call this if the user brings up a topic outside of your purview"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return custom_tool_handoff_agent_params

    async def run_impl(self, *args, **kwargs):
        from searchagent.agents.app_context import factory

        return factory.get_agent(AgentType.triage_agent).model_dump()
