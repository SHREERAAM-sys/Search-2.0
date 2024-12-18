import asyncio

import fire
from searchagent.agents.api import AgentType
from searchagent.agents.app_context import agent_manager, factory
from searchagent.agents.common.multi_turn import execute_turns, prompt_to_turn


async def run_main():
    await agent_manager.create_agents()
    triage_agent = factory.get_agent(AgentType.triage_agent)

    await execute_turns(
        triage_agent,
        turn_inputs=[prompt_to_turn(content="Find me a paper on colpali")],
    )


def main():
    asyncio.run(run_main())


if __name__ == "__main__":
    fire.Fire(main)
