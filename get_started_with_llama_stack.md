## Set up Llama Stack

After running `llama stack build`, we created a new file `csye7230-searchagent-stack-run.yaml` under the path `/Users/YOUR_NAME/.llama/builds/conda`

### Caveat

Since Llama Stack currently [doesn’t support the safety feature for Ollama](https://github.com/meta-llama/llama-stack), we need to disable the safety shield by using the --disable-safety flag, as shown below:

`python3 -m examples.agents.vacation localhost 11434 --disable-safety`

Also notice that we are using pgvector for memory. That means we need to configure the database with `dbname=remote:pgvector`, `user=llamastack_user`, and `password=attentionisallyouneed`

```
version: '2'
built_at: '2024-10-14T22:51:08.838065'
image_name: csye7230-searchagent-stack
docker_image: null
conda_env: csye7230-searchagent-stack
apis:
- inference
- safety
- agents
- memory
- telemetry
providers:
  inference:
  - provider_id: remote::ollama
    provider_type: remote::ollama
    config:
      host: localhost
      port: 11434
  safety:
  - provider_id: meta-reference
    provider_type: meta-reference
    config:
      llama_guard_shield:
        model: Llama-Guard-3-8B
        excluded_categories: []
      enable_prompt_guard: false
  agents:
  - provider_id: meta-reference
    provider_type: meta-reference
    config:
      persistence_store:
        namespace: null
        type: sqlite
        db_path: /Users/friedahuang/.llama/runtime/kvstore.db
  memory:
  - provider_id: remote::pgvector
    provider_type: remote::pgvector
    config:
      host: localhost
      port: 5432
      db: remote::pgvector
      user: llamastack_user
      password: attentionisallyouneed
  telemetry:
  - provider_id: meta-reference
    provider_type: meta-reference
    config: {}
```
