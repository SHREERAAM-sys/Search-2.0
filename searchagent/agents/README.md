# LLM Search Agent

Use the following command to run an example
`python3 -m searchagent.agents.rag_as_attachments localhost 11434 --disable-safety`

## Relevant Readings

-   Light weight multi-agent orchestration design pattern: [OpenAI Swarm](https://github.com/openai/swarm/tree/main)
-   [Orchestrating Agents: Routines and Handoffs](https://cookbook.openai.com/examples/orchestrating_agents)

## How to deal with common issues?

-   If the error is about `Llama3.1-8B-Instruct` not being registered, most likely it's Ollama not yet running
    -   Execute `ollama run llama3.1:8b-instruct-fp16`
