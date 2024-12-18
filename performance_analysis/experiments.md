### Basics

Since HNSW supports incremental indexing, we don’t DROP the index programmatically. As a result, when testing performance, we need to manually DROP the index beforehand to avoid any slowdown due to accumulated data.

### Metrics

**Testing on 100 PDF images from vidore/syntheticDocQA_artificial_intelligence_test**

#### Experiment #1

-   top_k = 10
-   Search and indexing strategy: ANNHNSWHamming

_Metrics_

```
{
    "average_recall": 0.03,
    "precision": 0.04,
    "mrr": 0.009999999999999998
}
```

_Latency and memory usage_

```
{
    "metric": "ram_usage",
    "function_name": "retrieve_pdfImage_from_vidore",
    "value": "RAM Usage - Current: 282.27 MB, Peak: 295.14 MB",
    "custom_msg": null
}
{
    "metric": "cpu_latency",
    "function_name": "retrieve_pdfImage_from_vidore",
    "value": "CPU Latency: 32881.650 ms",
    "custom_msg": null
}
```

#### Experiment #2

-   top_k = 10
-   Search strategy: ExactMaxSim

_Metrics_

```
{
    "average_recall": 0.99,
    "precision": 0.99,
    "mrr": 0.9670000000000001
}
```

#### Experiment #3

-   top_k = 10
-   Search and indexing strategy: ANNHNSWCosineSimilarity

_Metrics_

```
{
    "average_recall": 0.81,
    "precision": 0.81,
    "mrr": 0.6003174603174605
}
```

_Latency and memory usage_

```
{
    "metric": "ram_usage",
    "function_name": "convert_pdf2image_from_dir",
    "value": "RAM Usage - Current: 0.86 MB, Peak: 0.89 MB",
    "custom_msg": null
},
{
    "metric": "cpu_latency",
    "function_name": "convert_pdf2image_from_dir",
    "value": "CPU Latency: 643.443 ms",
    "custom_msg": null
}
```

### Implemented Optimization

#### Reduce Index Build Time

-   Applied `CREATE INDEX CONCURRENTLY ...`

    -   _Outcome_: it reduces total time slightly, but not by large margin

-   Experimented with different parameters for `ef_construction` = 128, 256, and 512 with `m = 16` and then `m = 32` with `ef_construction` being at least 4 \* `m` until reaching the target recall

### Cleaning

Before running any tests, run the following commands one by one

```
-- Find all indexes
SELECT
    tablename,
    indexname,
    indexdef
FROM
    pg_indexes
WHERE
    schemaname = 'public';

-- Drop index first
DROP INDEX IF EXISTS flattened_embedding_index;

-- Reset pg_stat_statements
SELECT
pg_stat_statements_reset ();

-- Truncate all tables and restart identity columns
TRUNCATE flattened_embedding RESTART IDENTITY CASCADE;

TRUNCATE embedding RESTART IDENTITY CASCADE;

TRUNCATE folder RESTART IDENTITY CASCADE;

TRUNCATE page RESTART IDENTITY CASCADE;

TRUNCATE file RESTART IDENTITY CASCADE;

TRUNCATE query RESTART IDENTITY CASCADE;

-- Analyze tables for updated statistics
ANALYZE;
```

### Relevant GitHub Issues

-   https://github.com/pgvector/pgvector/issues/543
-   https://github.com/pgvector/pgvector/issues/500
-   https://github.com/pgvector/pgvector/issues/299
