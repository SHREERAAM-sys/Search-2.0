### Key Observations

-   avg_time_ms for `CREATE INDEX flattened_embedding_index...` went from 2693ms to 2962.78ms
-   `INSERT INTO flattened_embedding;` went from avg_time_ms being 7038ms to 26.58ms, resulting a 102% time reduction
-   `WITH hamming_results...` went from avg_time_ms being 383ms to 413.38ms

This experiment was ran after we did some house cleaning—drop extra indexes and used `VACUUM ANALYZE`
