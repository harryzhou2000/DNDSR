set(CPPS
    Core/DomainPlacement.cpp
    Core/DomainExecutionContext.cpp
    Core/DomainComponents.cpp
    Core/DomainController.cpp
    Core/DomainStateSnapshot.cpp
    Core/Domain.cpp
    Core/Domain_ImplicitStep.cpp
    Core/Domain_TimeStep.cpp
    Core/Simulation.cpp
    Core/SimulationController.cpp

    Config/SimulationConfig.cpp

    Physics/EquationSystem.cpp
    Physics/EquationSetup.cpp
    Physics/EquationProperties.cpp
    Physics/EquationPositivity.cpp
    Physics/EquationSource.cpp
    Physics/EquationRANS.cpp
    Physics/StateInput.cpp
    Physics/InitialField.cpp

    State/EquationDofs.cpp
    State/EquationDofCommunication.cpp
    State/DerivedFields.cpp
    Motion/RigidMotion.cpp

    Boundary/BoundaryInputs.cpp
    Boundary/BoundarySystem.cpp
    Boundary/BoundarySystem_Operators.cpp
    Boundary/BoundaryPointOperator.cpp
    Interface/InterfaceCoupling.cpp

    Discretization/FiniteVolume.cpp
    Discretization/FiniteVolume_Workspace.cpp
    Discretization/FiniteVolume_WorkspaceFactory.cpp
    Discretization/FiniteVolume_Evaluation.cpp
    Discretization/FiniteVolume_ReconstructionState.cpp
    Discretization/FiniteVolume_BoundaryInputs.cpp
    Discretization/FiniteVolume_Flux.cpp
    Discretization/FiniteVolume_Source.cpp
    Discretization/FiniteVolume_Diagnostics.cpp
    Discretization/CPU/FiniteVolumeGradientKernels.cpp
    Discretization/CPU/FiniteVolumeBoundaryKernels.cpp
    Discretization/CPU/FiniteVolumeLimitedVRKernels.cpp
    Discretization/CPU/FiniteVolumeAuxiliaryKernels.cpp
    Discretization/CPU/FiniteVolumePositivityKernels.cpp
    Discretization/CPU/FiniteVolumeRHSKernels_Flow5.cpp
    Discretization/CPU/FiniteVolumeRHSKernels_Equation.cpp
    Discretization/CPU/FiniteVolumeSourceKernels.cpp
    Discretization/FiniteVolume_AuxiliaryValues.cpp
    Discretization/FiniteVolume_LimitedVR.cpp
    Discretization/FiniteVolume_PrepareAuxValues.cpp
    Discretization/FiniteVolume_TimeStep.cpp
    Discretization/FiniteVolume_InitializeField.cpp
    Discretization/WallDistanceSamples.cpp

    Implicit/LinearProblem.cpp
    Implicit/LinearProblem_Dof.cpp
    Implicit/LinearProblem_SGS.cpp
    Implicit/LinearProblem_DofSGS.cpp
    Implicit/LinearProblem_ILU.cpp
    Implicit/LinearSolver.cpp
    Implicit/NonlinearController.cpp
    Implicit/NonlinearSolver.cpp
    Implicit/UpdatePolicy.cpp
    Implicit/FiniteVolume_PrepareJacobian.cpp
    Implicit/FiniteVolume_Jacobian.cpp
    Implicit/JacobianKernels.cpp
    Multigrid/PolynomialMultigrid.cpp

    IO/Domain_Checkpoint.cpp
    IO/StateArchive.cpp
    IO/FiniteVolume_FieldOutput.cpp
    IO/Simulation_Checkpoint.cpp
    IO/ProgressOutput.cpp
    IO/OutputPlan.cpp
    IO/OutputCoordinator.cpp

    Riemann/RiemannFlux.cpp)
dnds_add_lib(cauchy "${CPPS}" "physics;cfv;geom;dnds;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" OFF ON ON)
dnds_add_lib(cauchy_shared "${CPPS}" "physics_shared;cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_shared LIBRARY DESTINATION "DNDSR/lib" COMPONENT py OPTIONAL)
install(TARGETS cauchy_shared LIBRARY DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/_lib" COMPONENT py OPTIONAL)
dnds_add_py_module(cauchy_pybind11 "Bindings/cauchy_pybind11.cpp;Bindings/Operators_bind.cpp" "cauchy_shared;cfv_shared;geom_shared;dnds_shared;${DNDS_EXTERNAL_LIBS}" "${DNDS_PCH_TARGETS_FAST}" ON ON ON)
install(TARGETS cauchy_pybind11 DESTINATION "DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
install(TARGETS cauchy_pybind11 DESTINATION "${PROJECT_SOURCE_DIR}/python/DNDSR/Cauchy/_ext" COMPONENT py OPTIONAL)
