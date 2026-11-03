set(CPPS Core/DomainPlacement.cpp Core/Domain.cpp Core/ImplicitStep.cpp Core/Simulation.cpp Config/SimulationConfig.cpp Physics/EquationSystem.cpp State/EquationDofs.cpp State/DerivedFields.cpp Motion/RigidMotion.cpp
    Boundary/BoundarySystem.cpp Discretization/FiniteVolume.cpp Discretization/LimitedVR.cpp Discretization/PrepareAuxValues.cpp Discretization/RHS.cpp Discretization/TimeStep.cpp Core/TimeStep.cpp Implicit/LinearProblem.cpp Implicit/SGS.cpp Implicit/ILU.cpp Implicit/LinearSolver.cpp Implicit/NonlinearSolver.cpp Implicit/PrepareJacobian.cpp Multigrid/PolynomialMultigrid.cpp IO/Checkpoint.cpp Riemann/RiemannFlux.cpp)
dnds_add_lib(cauchy "${CPPS}" "cfv;geom;dnds;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
dnds_add_lib(cauchy_shared "${CPPS}" "cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_shared LIBRARY DESTINATION "DNDSR/lib" COMPONENT py OPTIONAL)
install(TARGETS cauchy_shared LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib" COMPONENT py OPTIONAL)
dnds_add_py_module(cauchy_pybind11 "Bindings/cauchy_pybind11.cpp" "cauchy_shared;cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_pybind11 DESTINATION "DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
install(TARGETS cauchy_pybind11 DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
