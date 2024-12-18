from llama_stack_client import LlamaStackClient
from searchagent.agents.api import AgentManager
from searchagent.agents.common.factory import AgentFactory

client = LlamaStackClient(base_url="http://localhost:11434")
factory = AgentFactory()
agent_manager = AgentManager(factory)
