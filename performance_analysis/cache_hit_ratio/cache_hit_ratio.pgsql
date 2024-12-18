select
    userid::regrole,
    dbid,
    queryid,
    substring(query, 1, 50) as query,
    shared_blks_hit,
    shared_blks_read,
    CASE
        WHEN (shared_blks_hit + shared_blks_read) > 0 THEN round(
            (
                shared_blks_hit::numeric / (shared_blks_hit + shared_blks_read)
            ) * 100,
            2
        )
        ELSE 0
    END AS cache_hit_ratio
from
    pg_stat_statements
order by
    cache_hit_ratio desc
limit
    10;