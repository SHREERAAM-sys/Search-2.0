SELECT
    userid::regrole, -- Converts the user ID to a readable role name.
    dbid,
    queryid,
    query,
    shared_blks_hit, -- Blocks found in shared memory (cache).
    shared_blks_dirtied, -- Blocks modified in shared memory.
    temp_blks_read, -- Temporary blocks read from disk (indicates memory spill).
    temp_blks_written, -- Temporary blocks written to disk (indicates memory spill).
    total_exec_time
FROM
    pg_stat_statements
ORDER BY
    (temp_blks_written + temp_blks_read) DESC -- Orders by temporary block usage.
LIMIT
    10;