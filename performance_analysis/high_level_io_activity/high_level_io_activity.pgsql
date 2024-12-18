-- Identify queries that are potentially slowing down the database due to high disk I/O
-- Ensure track_io_timing is on inside postgresql.conf
select
    userid::regrole,
    dbid,
    substring(query, 1, 50) as query,
    queryid,
    mean_exec_time,
    shared_blk_read_time,
    shared_blk_write_time
from
    pg_stat_statements
order by
    (shared_blk_read_time + shared_blk_write_time) desc
limit
    10;