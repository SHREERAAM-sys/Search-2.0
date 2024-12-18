### Key Observations

-   `INSERT INTO flattened_embedding` and `CREATE INDEX ON flattened_embedding` are the most I/O-intensive queries.
-   `INSERT` takes 8103.57 ms mean_exec_time and 778.14 ms write time.
-   Given the high write operations and that disk I/O is a bottleneck, we can consider faster storage like NVMe SSDs to improve performance of theses disk-bound operations.
-   If we are using cloud infrastructure, we can provision dedicated disk resources for indexing.
-   Running index creation in parallel can significantly increase throughput by utilizing multiple CPU cores and I/O channels.
