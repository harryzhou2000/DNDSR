#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/CPU/TimeStageKernels.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "DNDS/Runtime/Execution/EagerArray.hpp"
namespace DNDS::Cauchy
{
    namespace R = Runtime;
    StepResult Domain::UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime)
    {
        _problem.finiteVolume.Execute(_state.trial, *_state.reconstruction, *_execution.workspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(_state.residual, stageTime));
        _lastFallbacks += _problem.finiteVolume.LastFaceFallbacks(*_execution.workspace);
        _lastCharacteristicFallbacks += _problem.finiteVolume.LastCharacteristicFallbacks(*_execution.workspace);
        if (_problem.equation.HasMixture() || _problem.equation.HasRANS())
        {
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            {
                const auto geometry = _problem.finiteVolume.Metrics().AcquireRead(); const auto metric = geometry.HostView();
                const auto initial = _state.initial.ReadOwned(), residual = _state.residual.ReadOwned();
                auto output = _state.trial.WriteOwned();
                const auto flowOutput = output.Block("flow");
                const EquationBlockArrayView<const real> flow{
                    flowOutput.parent.ConstView(), flowOutput.first, flowOutput.components};
                for (index iBlock = 0; iBlock < _state.trial.BlockNumber(); ++iBlock)
                {
                    const auto base = initial.Block(iBlock), rhs = residual.Block(iBlock); const auto state = output.Block(iBlock);
                    const auto &ransState = _problem.admissibility->RANS();
                    const bool rans = ransState.components &&
                                      _state.trial.Schema().blocks[iBlock].name == "rans";
                    RequireNumericalSuccess(CPU::UpdateExplicitBlock(base.owned, rhs.owned, state, flow, metric, ransState, rans,
                        dt, historyWeight, stageWeight), "UpdateExplicitBlock");
                } });
            if (!Admissible(_state.trial))
                return StepResult::Rejected("Cauchy explicit extended-equation stage is inadmissible; accepted state unchanged");
            return {};
        }
        auto &residual = _state.residual.CoupledField();
        auto geometry = _problem.finiteVolume.Metrics().AcquireRead();
        const auto metrics = geometry.HostView();
        real alpha = 1;
        int safe[2]{1, 1}; // high candidate, low candidate
        if (_positivity)
        {
            // PP POLICY -- pure-flow5 compatibility baseline: one domain-global
            // conservative high/low face-flux blend. This is the only explicit
            // thermal compression currently allowed; extended equations retry
            // the step instead of compressing their pointwise increments.
            // Deferred explicit PP work should limit shared face-flux
            // corrections using both adjacent cells, including MPI faces.
            // A cell-local increment factor is not a conservative substitute.
            const auto reconstruction = _problem.finiteVolume.PositivityReconstructionStats(*_execution.workspace);
            _lastBetaCells += reconstruction.first;
            _lastBeta = std::min(_lastBeta, reconstruction.second);
            index limited = 0;
            MPI::LocalStage(MPIInfo(_problem.finiteVolume.Metrics().Communicator()), [&]
                            {
                const auto initialRead = _state.initial.CoupledField().Owned().Read(), stateRead = _state.trial.CoupledField().Owned().Read();
                const auto highRead = residual.Owned().Read(), lowRead = _problem.finiteVolume.LowOrderResidual(*_execution.workspace).CoupledField().Owned().Read();
                const auto initial = initialRead.View<R::UniformTableRows>(), state = stateRead.View<R::UniformTableRows>();
                const auto high = highRead.View<R::UniformTableRows>(), low = lowRead.View<R::UniformTableRows>();
                const auto result = RequireNumericalSuccess(CPU::MeasureExplicitBlend(initial, state, high, low, metrics,
                    _problem.finiteVolume.ThermalConstraints(), dt, historyWeight, stageWeight), "MeasureExplicitBlend");
                alpha = result.alpha; limited = result.limited;
                safe[0] = result.safe[0]; safe[1] = result.safe[1]; });
            alpha = MPI::AllreduceValue<MPI::Reduction::Min>(alpha, _problem.finiteVolume.Metrics().Communicator());
            MPI::AllreduceInPlace<MPI::Reduction::Min>(
                MPI::ContiguousView<int>{safe, 2}, _problem.finiteVolume.Metrics().Communicator());
            if (safe[0])
            {
                alpha = 1;
                limited = 0; // No correction was applied to this high stage.
            }
            else
            {
                // A shared flux blend uses the low-order endpoint everywhere;
                // local success of the high endpoint cannot excuse an invalid
                // low endpoint when another cell actually requires blending.
                if (!safe[1])
                    return StepResult::Rejected("global conservative PP blend requires admissible low-order stage; reduce dt");
            }
            limited = MPI::AllreduceValue<MPI::Reduction::Sum>(limited, _problem.finiteVolume.Metrics().Communicator());
            _lastAlpha = std::min(_lastAlpha, alpha);
            _lastAlphaCells += limited;
        }
        std::optional<R::Array<real>::ReadLease> lowRead;
        R::ArrayView<const real, R::AccessBoundsChecks, R::UniformTableRows> low;
        if (_positivity)
        {
            lowRead.emplace(_problem.finiteVolume.LowOrderResidual(*_execution.workspace).CoupledField().Owned().Read());
            low = lowRead->View<R::UniformTableRows>();
        }
        R::SubmitArrays(_execution.Engine(), _execution.PrimaryStream(), [dt, historyWeight, stageWeight, metrics, alpha, low](auto, auto initial, auto rhs, auto stage)
                        { CPU::UpdateExplicitFlow(initial, rhs, low, stage, metrics, dt, historyWeight, stageWeight, alpha); }, R::ReadAs<R::UniformTableRows>(_state.initial.CoupledField().Owned()), R::ReadAs<R::UniformTableRows>(residual.Owned()), R::ReadWriteAs<R::UniformTableRows>(_state.trial.CoupledField().Owned()));
        _execution.JoinPrimary();
        if (!Admissible(_state.trial))
            return StepResult::Rejected("Cauchy SSPRK3 stage is inadmissible; accepted state unchanged");
        return {};
    }

}
