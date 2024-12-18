### Key Observations

NOTE: The following stats are produced by processing all 6 10-page PDFs in the pdfs directory

1. Top Query `INSERT into flattened_embedding`
    - Total time: 457,442.14 ms (~7.6 minutes) across 65 calls
    - Mean time: each `INSERT` takes approximately 7037.57 ms
    - CPU usage: this query alone accounts for 96.6% of the total execution time. It is highly resource-intensive
    - `INSERT` is very costly, best to optimize it
2. Second `INSERT into flattened_embedding`
    - For some reason, we are conducting two INSERT operations separately into the same table
    - Total time: 12,440.19 ms across 65 calls
    - Mean time: each `INSERT` takes about 191.39 ms on average
    - CPU usage: this query accounts for 2.63% of the total CPU time
    - Ensure there is no redundant insertions here
3. `CREATE INDEX on flattened_embedding`
    - Total time: 2,693.27 ms (single execution)
    - CPU usage: uses 0.57% of the total time
    - Consider creating index during low traffic time
4. Query with Hamming Distance (WITH hamming_results AS (...)):
    - Total time: 383.06 ms (single execution)
    - CPU usage: uses 0.08% of the total CPU time
5. `INSERT into embedding table`
    - Total time: 353.50 ms across 65 calls
    - Mean time: each call takes about 5.44 ms
    - CPU usage: uses 0.07% of the total time
