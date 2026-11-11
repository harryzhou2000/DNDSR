#pragma once

#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Implicit/NonlinearSolver.hpp"
#include "Cauchy/Implicit/NonlinearController.hpp"
#include "Cauchy/Multigrid/PolynomialMultigrid.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/State/EquationTimeStepper.hpp"

namespace DNDS::Cauchy
{
    /// Long-lived physical/discretization definition of one domain. Mutable
    /// solution history and backend scratch deliberately do not live here.
    struct DomainProblem
    {
        std::string name;
        EquationSystem equation;
        FiniteVolume finiteVolume;
        std::optional<EquationAdmissibilityPolicy> admissibility;

        DomainProblem(std::string name, ssp<Geom::UnstructuredMesh> mesh,
                      const DomainSettings &settings, const TimeSettings &time,
                      EquationSystem equation);
    };

    /// Restart-relevant mutable numerical state. Every member either appears
    /// in a checkpoint or defines the active trial/cursor from which an exact
    /// internal checkpoint is formed. No engine, stream or temporary workspace
    /// is retained here.
    struct DomainState
    {
        EquationDofs accepted, trial, residual, initial;
        std::shared_ptr<FiniteVolume::ReconstructionState> reconstruction;
        std::optional<EquationDofs> reconstructionBackup;
        std::optional<Runtime::Array<real>> limitedVRBackup;
        std::unique_ptr<EquationTimeStepper> timeStepper;
        NonlinearSolveState coupledNonlinear;
        bool restoreReconstruction = false;
        StateEpoch epoch, trialEpoch;
        uint64_t nextTrial = 0;
        real time = 0, trialTime = 0, trialDt = 0, lastAcceptedDt = 0,
             progressTime = 0;
        bool trialReady = false, internalResume = false;
        std::optional<EquationTimeStepper::StageEquation> stageEquation;
        int coordinatedStage = 0;
        bool coordinatedPrepared = false;

        DomainState(const DomainProblem &problem, const DomainSettings &settings,
                    const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers,
                    const std::vector<real> &initialConservative);
    };

    /// Current two-stage nonlinear operator bundle. Durable midpoint/end
    /// reconstruction state remains explicit while the callable portion will
    /// migrate behind UpdatePolicy in the next controller phase.
    struct DomainCoupledStageOperator
    {
        EquationTimeStepper::Evaluate evaluate;
        std::function<NonlinearIterationReport(
            const ODE::CoupledStageEquation<EquationDofs> &,
            const EquationDofs &, real)>
            correct;
        std::function<std::pair<int, real>()> reconstruction;
        std::shared_ptr<FiniteVolume::ReconstructionState> state;
    };

    /// Rebuildable backend-resolved execution resources. This owner contains
    /// no accepted numerical state. Its context handle is declared last and is
    /// released first; solver-held copies keep that same runtime alive through
    /// solver teardown. Their final local context release drains the engine
    /// before workspace fields and, subsequently, DomainState/Problem vanish.
    /// Managed operation records retain buffers required during that drain.
    struct DomainExecution
    {
        std::shared_ptr<FiniteVolume::Workspace> workspace,
            observationWorkspace;
        std::shared_ptr<FiniteVolume::ReconstructionState>
            observationReconstruction;
        std::optional<EquationDofs> observationInput, observationState;
        std::unique_ptr<NonlinearSolver> implicit;
        std::unique_ptr<PolynomialMultigrid> multigrid;
        mutable std::optional<AuxiliaryValues> timeAuxiliary;
        mutable std::optional<MeanGradients> timeMeanGradients;
        mutable std::vector<Runtime::TransferBinding> timeMeanHalo;
        std::shared_ptr<FiniteVolume::MeanWorkspace> timePPWorkspace;
        std::optional<EquationDofs> timePPResidual;
        std::optional<BoundaryInputs> interfaceInputs;
        std::function<DomainCoupledStageOperator(EquationDofs &)>
            coupledFactory;
        Runtime::ArrayStorage::Shared coupledMiddleOwner;
        std::optional<EquationDofs> coupledResidual;
        std::unique_ptr<NonlinearController> nonlinearController;
        DomainExecutionContext context;

        DomainExecution(const DomainProblem &problem, DomainState &state,
                        const ExecutionSettings &settings,
                        std::optional<Physics::Chemistry::ChemicalWorkerPool> workers);

        Runtime::EagerEngine &Engine() const { return context.Engine(); }
        DomainExecutionContext &Context() { return context; }
        const DomainExecutionContext &Context() const { return context; }
        const Runtime::ExecutionStream &PrimaryStream() const
        {
            return context.PrimaryStream();
        }
        const std::optional<Physics::Chemistry::ChemicalWorkerPool> &
        ChemicalWorkers() const
        {
            return context.ChemicalWorkers();
        }
        void JoinPrimary() const { context.JoinPrimary(); }
    };
}
