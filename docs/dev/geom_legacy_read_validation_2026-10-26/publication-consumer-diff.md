# Published consumer diff

Compared to shared dev/harry base `03891f8b`; includes required publication formatting.

The prior [validated snapshot](consumer-diff.md) measured 2,142 changed lines.
Formatting adds 146 whitespace-only lines in EulerP declarations/comments. The
whitespace-insensitive count remains 2,092.

| File | Before + / - | Published + / - |
|---|---:|---:|
| `app/CFV/vrBasic_Test.cpp` | 10 / 18 | 10 / 18 |
| `app/CFV/vrStatic_Test.cpp` | 16 / 29 | 16 / 29 |
| `app/Euler/jacobiLUTest.cpp` | 12 / 20 | 12 / 20 |
| `app/Geom/meshSerial_Test.cpp` | 29 / 31 | 29 / 31 |
| `app/Geom/partitionMeshSerial.cpp` | 9 / 34 | 9 / 34 |
| `python/DNDSR/EulerP/EulerP_Solver.py` | 42 / 11 | 42 / 11 |
| `python/DNDSR/Geom/utils.py` | 182 / 210 | 182 / 210 |
| `src/CFV/BenchmarkFiniteVolume.hpp` | 12 / 9 | 6 / 3 |
| `src/CFV/DOFFactory.hpp` | 7 / 6 | 7 / 6 |
| `src/CFV/FiniteVolume.cpp` | 142 / 73 | 86 / 1 |
| `src/CFV/FiniteVolume.hpp` | 239 / 32 | 239 / 32 |
| `src/CFV/FiniteVolume_DeviceView.hpp` | 55 / 29 | 55 / 29 |
| `src/CFV/FiniteVolume_bind.cpp` | 2 / 1 | 2 / 1 |
| `src/CFV/ModelEvaluator.cpp` | 11 / 5 | 7 / 0 |
| `src/CFV/VariationalReconstruction.cpp` | 90 / 68 | 27 / 3 |
| `src/CFV/VariationalReconstruction.hpp` | 88 / 27 | 88 / 27 |
| `src/CFV/VariationalReconstruction_LimiterProcedure.hxx` | 37 / 24 | 29 / 12 |
| `src/CFV/VariationalReconstruction_Reconstruction.hxx` | 49 / 33 | 33 / 12 |
| `src/Euler/EulerEvaluator.hpp` | 54 / 21 | 51 / 16 |
| `src/Euler/EulerEvaluator.hxx` | 215 / 156 | 94 / 18 |
| `src/Euler/EulerEvaluator_EvaluateDt.hxx` | 116 / 72 | 79 / 29 |
| `src/Euler/EulerEvaluator_EvaluateRHS.hxx` | 29 / 25 | 12 / 7 |
| `src/Euler/EulerSolver.hxx` | 1 / 1 | 1 / 1 |
| `src/Euler/EulerSolver_Init.hxx` | 66 / 182 | 66 / 182 |
| `src/Euler/EulerSolver_PrintData.hxx` | 84 / 41 | 84 / 41 |
| `src/EulerP/EulerP_ARS.hpp` | 10 / 10 | 10 / 10 |
| `src/EulerP/EulerP_Evaluator.hpp` | 145 / 80 | 145 / 80 |
| `src/EulerP/EulerP_Evaluator_impl_common.hxx` | 16 / 14 | 3 / 1 |

Total: 3,030 → 2,288 changed lines (742 fewer, 24.5%).
C++ CFV/Euler/EulerP: 2,377 → 1,635 (31.2% fewer).
Ignoring whitespace across all28 source consumers: 2,824 → 2,092 (25.9% fewer).

| Main purpose of file bucket | Remaining changed lines |
|---|---:|
| Native frontend/setup migration | 901 |
| CFV metrics/reconstruction validation and read scopes | 676 |
| Euler operator read prologues, wall epochs and guards | 306 |
| Euler output/restart and coordinate hashing | 127 |
| EulerP owning views, kernel adapters and numerical fix | 249 |
| CFV field indexing, benchmark and model entry adapter | 29 |

Buckets group whole files and contain mixed changes, comments and formatting;
they do not assign each individual line to one semantic purpose. The facade
implementation, tests and documentation are excluded from consumer totals.
