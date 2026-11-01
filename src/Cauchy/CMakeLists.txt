set(CPPS Core/DomainPlacement.cpp Core/Domain.cpp Core/Simulation.cpp Config/SimulationConfig.cpp Physics/EquationSystem.cpp
    Boundary/BoundarySystem.cpp Discretization/FiniteVolume.cpp IO/Checkpoint.cpp Riemann/RiemannFlux.cpp)
dnds_add_lib(cauchy "${CPPS}" "cfv;geom;dnds;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
dnds_add_lib(cauchy_shared "${CPPS}" "cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_shared LIBRARY DESTINATION "DNDSR/lib" COMPONENT py OPTIONAL)
install(TARGETS cauchy_shared LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib" COMPONENT py OPTIONAL)
dnds_add_py_module(cauchy_pybind11 "Bindings/cauchy_pybind11.cpp" "cauchy_shared;cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_pybind11 DESTINATION "DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
install(TARGETS cauchy_pybind11 DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
