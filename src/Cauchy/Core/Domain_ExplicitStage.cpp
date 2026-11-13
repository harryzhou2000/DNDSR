#include "Cauchy/Core/Domain.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Core/TimeStageOperators.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
namespace DNDS::Cauchy
{
    namespace R = Runtime;
    StepResult Domain::UpdateStage(real dt, real historyWeight, real stageWeight, real stageTime)
    {
        _problem.finiteVolume.Execute(_state.trial, *_state.reconstruction, *_execution.workspace, _execution.Context(),
                                      FiniteVolumeEvaluation::Spatial(_state.residual, stageTime));
        _lastFallbacks += _problem.finiteVolume.LastFaceFallbacks(*_execution.workspace);
        _lastCharacteristicFallbacks += _problem.finiteVolume.LastCharacteristicFallbacks(*_execution.workspace);
        if (!_execution.timeStages)
            _execution.timeStages.emplace(_problem.finiteVolume.Metrics(), _execution.ComputeStream().Context());
        const ExplicitStageCoefficients coefficients{dt, historyWeight, stageWeight};
        if (_problem.equation.HasMixture() || _problem.equation.HasRANS())
        {
            _execution.timeStages->UpdateExtended(_execution.Engine(), _execution.ComputeStream(),
                                                  _state.initial, _state.residual, _state.trial, _problem.admissibility->RANS(), coefficients);
            if (!Admissible(_state.trial))
                return StepResult::Rejected("Cauchy explicit extended-equation stage is inadmissible; accepted state unchanged");
            return {};
        }
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
            const auto result = _execution.timeStages->MeasureBlend(_execution.Engine(), _execution.ComputeStream(),
                                                                    _state.initial, _state.trial, _state.residual, _problem.finiteVolume.LowOrderResidual(*_execution.workspace),
                                                                    _problem.finiteVolume.ThermalConstraints(), coefficients);
            alpha = result.alpha;
            limited = result.limited;
            safe[0] = result.safe[0];
            safe[1] = result.safe[1];
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
        const auto low = _positivity ? OptionalRef<const EquationDofs>(_problem.finiteVolume.LowOrderResidual(*_execution.workspace))
                                     : OptionalRef<const EquationDofs>{};
        _execution.timeStages->UpdateFlow(_execution.Engine(), _execution.ComputeStream(),
                                          _state.initial, _state.residual, _state.trial, low, alpha, coefficients);
        if (!Admissible(_state.trial))
            return StepResult::Rejected("Cauchy SSPRK3 stage is inadmissible; accepted state unchanged");
        return {};
    }

}
