# Consumer diff against shared dev/harry base

Base `03891f8b`; before `1841a2e2`; after is the uncommitted working tree.
Git line counts include whitespace/comments. Exact counts with `-w` are also in
[consumer-diff.json](consumer-diff.json). Geom/DNDS implementation, docs and tests
are excluded from this consumer-source table.

| Consumer file | Before + / - | After + / - | Changed lines removed |
|---|---:|---:|---:|
| `app/CFV/vrBasic_Test.cpp` | 10 / 18 | 10 / 18 | 0 |
| `app/CFV/vrStatic_Test.cpp` | 16 / 29 | 16 / 29 | 0 |
| `app/Euler/jacobiLUTest.cpp` | 12 / 20 | 12 / 20 | 0 |
| `app/Geom/meshSerial_Test.cpp` | 29 / 31 | 29 / 31 | 0 |
| `app/Geom/partitionMeshSerial.cpp` | 9 / 34 | 9 / 34 | 0 |
| `python/DNDSR/EulerP/EulerP_Solver.py` | 42 / 11 | 42 / 11 | 0 |
| `python/DNDSR/Geom/utils.py` | 182 / 210 | 182 / 210 | 0 |
| `src/CFV/BenchmarkFiniteVolume.hpp` | 12 / 9 | 6 / 3 | 12 |
| `src/CFV/DOFFactory.hpp` | 7 / 6 | 7 / 6 | 0 |
| `src/CFV/FiniteVolume.cpp` | 142 / 73 | 86 / 1 | 128 |
| `src/CFV/FiniteVolume.hpp` | 239 / 32 | 239 / 32 | 0 |
| `src/CFV/FiniteVolume_DeviceView.hpp` | 55 / 29 | 55 / 29 | 0 |
| `src/CFV/FiniteVolume_bind.cpp` | 2 / 1 | 2 / 1 | 0 |
| `src/CFV/ModelEvaluator.cpp` | 11 / 5 | 7 / 0 | 9 |
| `src/CFV/VariationalReconstruction.cpp` | 90 / 68 | 27 / 3 | 128 |
| `src/CFV/VariationalReconstruction.hpp` | 88 / 27 | 88 / 27 | 0 |
| `src/CFV/VariationalReconstruction_LimiterProcedure.hxx` | 37 / 24 | 29 / 12 | 20 |
| `src/CFV/VariationalReconstruction_Reconstruction.hxx` | 49 / 33 | 33 / 12 | 37 |
| `src/Euler/EulerEvaluator.hpp` | 54 / 21 | 51 / 16 | 8 |
| `src/Euler/EulerEvaluator.hxx` | 215 / 156 | 94 / 18 | 259 |
| `src/Euler/EulerEvaluator_EvaluateDt.hxx` | 116 / 72 | 79 / 29 | 80 |
| `src/Euler/EulerEvaluator_EvaluateRHS.hxx` | 29 / 25 | 12 / 7 | 35 |
| `src/Euler/EulerSolver.hxx` | 1 / 1 | 1 / 1 | 0 |
| `src/Euler/EulerSolver_Init.hxx` | 66 / 182 | 66 / 182 | 0 |
| `src/Euler/EulerSolver_PrintData.hxx` | 84 / 41 | 84 / 41 | 0 |
| `src/EulerP/EulerP_ARS.hpp` | 10 / 10 | 10 / 10 | 0 |
| `src/EulerP/EulerP_Evaluator.hpp` | 145 / 80 | 72 / 7 | 146 |
| `src/EulerP/EulerP_Evaluator_impl_common.hxx` | 16 / 14 | 3 / 1 | 26 |

Total: 3,030 to 2,142 changed lines (888 fewer, 29.3%).
C++ CFV/Euler/EulerP: 2,377 to 1,489 (37.4% fewer).
Ignoring whitespace: 2,824 to 2,092 across all 28 files (25.9% fewer).

These figures exclude the new facade implementation and tests. The native
read surface is an adapter over retained buffers, with no legacy payload copy.
