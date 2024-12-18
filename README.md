# Search2.0

Search2.0 allows users to search their local files using natural language, powered by LLM agents. It use tools like PDF and code search to find relevant files. For example, a user might ask, “Find the CSYE7230 project proposal,” and the app will return the most relevant file paths, a summary, and allow them to view the content. We will build our agentic system using Llama Stack, developed by Meta.

### PDFSearchTool

We plan to scale the PDF search tool using ColPali for production. ColPali eliminates the need for complex and fragile layout recognition or OCR pipelines by using a single model that understands both the text and visual content (e.g., layout, charts) of a document. It has delivered the best results so far. We will experiment with the following ideas:

-   Use smaller models such as Llama3.2 and key value cache to reduce latency
-   Cache warming when users are typing
-   KV cache compression
-   Binary Quantization
-   Model distillation
-   Sync local files and remote indexes using hash and hierarchical file traversal

### CodeSearchTool

The **CodeSearchTool** is a microservice-based application designed to enable efficient discovery and exploration of code files. By leveraging vector embeddings and similarity-based searches, the tool ensures contextually accurate results for code queries. It is powered by **MiniLM-L6** embeddings and utilizes **cosine similarity** for precise search functionality.

#### Features
- Upload code files to generate embeddings.
- Perform similarity-based searches across indexed code files.
- Powered by a large language model (LLM) for context-aware results.


#### Technical Details
- **Embedding Model**: MiniLM-L6
- **Similarity Algorithm**: Cosine similarity
- **Architecture**: Built with microservices and LLM integration for efficient and accurate code search.



This tool simplifies code exploration, making it ideal for developers managing large codebases or seeking relevant snippets efficiently.

---


## Tool Demo

[![CodeSearchTool Demo](https://img.youtube.com/vi/yQ2MKvpvHro/0.jpg)](https://www.youtube.com/watch?v=yQ2MKvpvHro)


## Evaluation

1. We use [LLama3.2-vision](https://ollama.com/x/llama3.2-vision:11b-instruct-q8_0) to generate a synthetic evaluation dataset.

2. We use the [CVPR 2019 Papers](https://www.kaggle.com/datasets/paultimothymooney/cvpr-2019-papers) dataset as our source of PDF documents.

3. We use [Langchain](https://github.com/langchain-ai/langchain) to generate synthetic data and create a baseline for our evaluation.

## Dataset

We use the [CVPR 2019 Papers](https://www.kaggle.com/datasets/paultimothymooney/cvpr-2019-papers) dataset from Kaggle, containing over 1,000 academic papers from the CVPR 2019 conference. From this dataset, 5 papers were randomly selected to generate 10 test sets (`cvpr2019_5papers_testset_12q.csv`) using the [Ragas](https://docs.ragas.io/en/latest/index.html) framework. The dataset can be found at [friedahuang/cvpr2019_5papers_testset_12q](https://huggingface.co/datasets/friedahuang/cvpr2019_5papers_testset_12q) See `ragas_evaluate.py` for the implementation.

1.  Aafaq_Spatio-Temporal_Dynamics_and_Semantic_Attribute_Enriched_Visual_Encoding_for_Video_CVPR_2019_paper.pdf
2.  Aakur_A_Perceptual_Prediction_Framework_for_Self_Supervised_Event_Segmentation_CVPR_2019_paper.pdf
3.  Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.pdf
4.  Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_Hand-Gesture_Recognition_With_Multimodal_CVPR_2019_paper.pdf
5.  Abbasnejad_A_Generative_Adversarial_Density_Estimator_CVPR_2019_paper.pdf

Additionally, we use the Huggingface dataset [(m-ric/huggingface_doc)](https://huggingface.co/datasets/m-ric/huggingface_doc) to generate 347 question-answer pairs (QA couples). The synthetically generated QA couples can be found at [friedahuang/m-ric_huggingface_doc_347](https://huggingface.co/datasets/friedahuang/m-ric_huggingface_doc_347). We will focus on this evaluation dataset because it has a larger volume and we've already established a baseline RAG system benchmarked against it. See `benchmark_rag.py` for the implementation.

## Tech Stack

### Frontend

-   Next.js
-   Typescript
-   [shadcn/ui](https://ui.shadcn.com/)

### Backend

-   Python
-   [SQLAlchemy](https://www.sqlalchemy.org/): Python SQL toolkit and Object Relational Mapper
-   [Llama Stack](https://github.com/meta-llama/llama-stack): Standardize the building blocks needed to bring generative AI applications to market
-   [pgvector](https://github.com/pgvector/pgvector-python): An extension of PostgreSQL with the ability to store and search vector embeddings alongside regular data
-   [LangChain](https://www.langchain.com/): Framework for LLM applications (It is only used for evaluation purpose)

### Database

-   [PostgreSQL 17](https://www.postgresql.org/)
-   [pgvector](https://github.com/pgvector/pgvector)
-   [Psycopg3](https://www.psycopg.org/psycopg3/docs/): PostgreSQL database adapter for Python
-   [Alembic](https://github.com/sqlalchemy/alembic): Database migrations tool

### Models

-   [ColPali](https://github.com/illuin-tech/colpali): A vision retriever based on the ColBERT architecture and the PaliGemma model
-   [Llama3.2](https://ollama.com/library/llama3.2:latest): llama3.2:latest

### Devops

-   Vercel
-   GCP

### MLops

-   [Unsloth](https://github.com/unslothai/unsloth): Finetune & train LLMs
-   [RunPod](https://www.runpod.io/): Cloud computing platform for ML apps
-   [Ollama](https://ollama.com/): Run LLM locally

### Code Quality & Tooling

-   [Loguru](https://github.com/Delgan/loguru): Simplified Python logging
-   [pre-commit](https://pre-commit.com/): Multi-language pre-commit hooks manager
-   [Ruff](https://docs.astral.sh/ruff/): Fast Python linter and formatter

## Setup

Instructions on how to set up the project locally. For example:

1. Clone the repository:

    ```
    git clone https://github.com/SHREERAAM-sys/Search-2.0.git
    ```

2. Install dependencies:

    ```
    poetry install
    ```

3. Set up pre-commit hooks:
    ```
    pre-commit install
    ```

### Database Setup

1. Create a Database user:
   `CREATE USER searchagent_user WITH PASSWORD 'your_secure_password';`
2. Create a new database:
   `CREATE DATABASE searchagent OWNER searchagent_user;`
3. Grant necessary privileges:
   `GRANT ALL PRIVILEGES ON DATABASE searchagent TO searchagent_user;`
4. Add PostgreSQL connection to .env
   `DATABASE_URL=postgresql+psycopg://searchagent_user:your_secure_password@localhost:5432/searchagent`
5. Use the searchagent_user in psql
   `psql -U searchagent_user -d searchagent`
6. Ensure searchagent_user has superuser privileges by logging as the superuser
   `ALTER USER searchagent_user WITH SUPERUSER;`
7. Enable the pgvector extension
   `CREATE EXTENSION vector;`

### Tune Postgres Server Performance

1. Find config file with `SHOW config_file`; in mac, it's in `/opt/homebrew/var/postgresql@17/postgresql.conf`
2. Use [PgTune](https://pgtune.leopard.in.ua/) to set initial values for Postgres server parameters

For example, on my machine (Apple M2 Pro), I have the following initial settings

```
# DB Version: 17
# OS Type: mac
# DB Type: web
# Total Memory (RAM): 32 GB
# CPUs num: 12
# Connections num: 100
# Data Storage: ssd

max_connections = 100
shared_buffers = 8GB
effective_cache_size = 24GB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
work_mem = 20971kB
huge_pages = try
min_wal_size = 1GB
max_wal_size = 4GB
max_worker_processes = 12
max_parallel_workers_per_gather = 4
max_parallel_workers = 12
max_parallel_maintenance_workers = 4
```

### Monitor Performance

We use [pg_stat_statements](https://www.postgresql.org/docs/current/pgstatstatements.html) to monitor overall query performance in PostgreSQL

#### Caveat

When querying high I/O activity, top time-consuming queries, or high memory usage, remember to use `pg_stat_statements_reset()` to flush data from pg_stat_statements

### Maintenance

`VACUUM ANALYZE;` to reduce bloat and refresh query planner statistics

### Run Database Migration

1. Create new migration by running `alembic revision --autogenerate -m "YOUR MSG"`
2. Apply new migration by running `alembic upgrade head`

### Caveat on PostgreSQL Column Type Updates

When updating column types in the ORM, ensure the database schema reflects these changes.

For instance, if changing the vector type in the query table from full precision to half-precision (halfvec), apply the corresponding database migration.

```
ALTER TABLE query
ALTER COLUMN vector_embedding
TYPE HALFVEC(128)[]
USING vector_embedding::HALFVEC(128)[];
```

For `flattened_embedding` table

```
ALTER TABLE flattened_embedding
ALTER COLUMN vector_embedding
TYPE HALFVEC(128)
USING vector_embedding::HALFVEC(128);
```

### To deal with invalid password after the computer didn't complete the shutdown process, we need to execute the following commands:

```
cd /opt/homebrew/var/postgresql

rm postmaster.pid

brew services restart postgresql@17
```

## File Access Scope

We will only access the user’s home directory, which contains most user-accessible files and data. The home directory includes:

-   Desktop
-   Documents
-   Downloads
-   Pictures

## Example

```
>>> response = searchagent.query("find csye7230 project proposal")
>>> response.documents

Output:

Document(metadata={'source': '../proposals/csye7230_project_proposal_part_a.pdf'}, page_content='...')
Document(metadata={'source': '../proposals/csye7230_project_proposal_part_b.pdf'}, page_content='...')
Document(metadata={'source': '../proposals/csye7230_project_benchmarking_report.pdf'}, page_content='...')

>>> response.answer

Output:

The following files match your query:

1. csye7230_project_proposal_part_a.pdf
`../proposals/csye7230_project_proposal_part_a.pdf`

This PDF contains the project proposal for CSYE7230, detailing objectives, methodologies, and expected outcomes.

2. csye7230_project_proposal_part_b.pdf
`../proposals/csye7230_project_proposal_part_b.pdf`

This PDF outlines the implementation plan for project proposal part B, focusing on architecture and design choices.

3. csye7230_project_benchmarking_report.pdf
`../proposals/csye7230_project_benchmarking_report.pdf`

This PDF presents the benchmarking report for CSYE7230, evaluating the performance metrics and analysis of the project components.
```

## Links to Docs

-   [Project Proposal Part A](https://docs.google.com/document/d/1ojm1jtU8u-KRpF2hjG2bRb_PP1dPwSrfAUV27Sl0KeQ/edit?usp=sharing)
-   [Project Part B](https://docs.google.com/document/d/1-DSOruZCWS8Qez2NkEPJyNLbkWkpuJq9jCWl9D10ed8/edit?usp=sharing)
-   [Project Part C](https://docs.google.com/document/d/1LHl1XXJS-V-2-x-HQ5pUwwbd4AccCRmH1UK_u1THqUE/edit?usp=sharing)
-   [UML diagrams](https://drive.google.com/file/d/1AIpMmYtItZ8XGqRUUux1majA1Ue5sLSE/view?usp=sharing)
-   [Figma Design](https://www.figma.com/design/H2o8kObQSkwgRQtMvehywK/CSYE7230?node-id=0-1&t=HTzYXd49McEctP41-1)

## Resources

-   [Python: Production-Level Coding Practices](https://medium.com/red-buffer/python-production-level-coding-practices-4c39246e0233)
-   [RAG Evaluation (LLM-as-a-judge)](https://huggingface.co/learn/cookbook/rag_evaluation)
-   [Analyze file system and folder structures with Python](https://janakiev.com/blog/python-filesystem-analysis/)
-   [Pytest Best Practices](https://realpython.com/pytest-python-testing/)
-   [Implement semantic cache to improve a RAG system](https://huggingface.co/learn/cookbook/semantic_cache_chroma_vector_database)
-   [Set up eval pipeline](https://www.youtube.com/watch?v=eLXF0VojuSs&t=140s)
-   [Reranking](https://medium.com/google-cloud/reranking-3b5f351cb398)
-   [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)
-   [A Reddit post on a new chunking algorithm](https://www.reddit.com/r/LangChain/comments/1flhtxi/a_new_chunking_algorithm_proposal_semantically/)
-   [5 levels of text splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
-   [A blog on ripgrep—a line-oriented search tool that recursively searches the current directory for a regex pattern](https://blog.burntsushi.net/ripgrep/)
-   [microsearch—a search engine in 80 lines of Python](https://www.alexmolas.com/2024/02/05/a-search-engine-in-80-lines.html)
-   [Agent architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
-   [Embedding Quantization](https://huggingface.co/blog/embedding-quantization)
-   [Llama3.2 is here](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
-   [The The Ultimate Guide to Vector Database Landscape — 2024 and Beyond](https://medium.com/madhukarkumar/the-ultimate-guide-to-vector-databases-2024-and-beyond-16dfb15bef12)
-   [pgvector: Multi-vector support](https://github.com/pgvector/pgvector/issues/640)
-   [pg_stat_statements](https://dbaclass.com/article/monitor-sql-queries-in-postgres-using-pg_stat_statements/)
