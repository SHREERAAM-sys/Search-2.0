-- Retrieve the top 10 most time-consuming queries in terms of total execution time
SELECT
    substring(query, 1, 50) as query,
    round(total_exec_time::numeric, 2) AS total_time,
    calls,
    round(mean_exec_time::numeric, 2) AS mean,
    round(
        (
            100 * total_exec_time / sum(total_exec_time::numeric) OVER ()
        )::numeric,
        2
    ) AS percentage_cpu
FROM
    pg_stat_statements
ORDER BY
    total_time DESC
LIMIT
    10;