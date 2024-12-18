from typing import List, Optional

from searchagent.agents.common.client_utils import (
    QuickToolConfig,
    get_agent_with_custom_tools,
    make_agent_config_with_custom_tools,
)
from searchagent.agents.common.custom_tools import CustomTool
from searchagent.agents.common.types import AgentWithCustomToolExecutor


class AgentFactory:
    def __init__(self):
        self._agents = {}

    async def create_agent(
        self,
        name: str,
        custom_tools: List[CustomTool] = [],
    ) -> AgentWithCustomToolExecutor:
        agent_config = await make_agent_config_with_custom_tools(
            tool_config=QuickToolConfig(
                custom_tools=custom_tools,
                prompt_format="function_tag",
            )
        )
        return await get_agent_with_custom_tools(name, agent_config, custom_tools)

    async def create_and_register_agent(
        self,
        name: str,
        custom_tools: List[CustomTool] = [],
    ):
        agent = await self.create_agent(name, custom_tools)
        self._agents[name] = agent

    def get_agent(self, name: str) -> Optional[AgentWithCustomToolExecutor]:
        return self._agents.get(name)
