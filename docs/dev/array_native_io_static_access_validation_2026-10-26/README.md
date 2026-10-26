# Native Array revision validation

See [manifest](manifest.json) for results, exact source/binary hashes and limitations.
Passing CTest logs, Python matrices, profile verification and earlier failures are retained losslessly as gzip.
The Python driver records its exact commands; final GPU runs use test/demos/one_rank_per_gpu.py.
Only the two new static/eager CUDA suites are claimed Compute-Sanitizer clean.
