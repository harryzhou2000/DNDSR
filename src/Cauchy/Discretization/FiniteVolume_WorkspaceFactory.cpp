#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Runtime/Execution/SessionAccess.hpp"
#include "DNDS/MPI/Failure.hpp"
#include "DNDS/Algebra/DofHalo.hpp"
#include "Cauchy/Core/DomainExecutionContext.hpp"
#include "Cauchy/Core/OperatorSchedule.hpp"
#include "Cauchy/Physics/EquationProperties.hpp"
#include "Cauchy/Physics/EquationTransport.hpp"
#include "Cauchy/Physics/EquationFlux.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/Physics/EquationRANS.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "Cauchy/State/EquationDofCommunication.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;
    void FiniteVolume::InitializeConservative(EquationDofs &state, const std::vector<real> &values) const
    {
        Metrics().Communication()->LocalStage([&]
                                              {
            Metrics().Validate();
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            index width = 0;
            for (const auto &block : _data->equation.Schema().blocks) width += index(block.components.size());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && index(values.size()) == width &&
                                      std::all_of(values.begin(), values.end(), [](real v) { return std::isfinite(v); }),
                                  "conservative initializer row space or values differ from equation");
            FlowState flow;
            std::copy_n(values.begin(), 5, flow.begin());
            flow = _data->motion.At(0).LocalState(flow);
            auto output = state.WriteOwned();
            index offset = 0;
            for (const auto &block : _data->equation.Schema().blocks)
            {
                const auto rows = output.Block(block.name);
                for (index iCell = 0; iCell < rows.RowCount(); ++iCell)
                    for (index j = 0; j < rows.ComponentNumber(); ++j)
                        rows(iCell, j) = block.name == "flow" ? flow[j] : values[offset + j];
                offset += index(block.components.size());
            } });
    }
    void FiniteVolume::InitializeConstant(EquationDofs &state, const FlowState &primitive,
                                          const std::map<std::string, real> &massFractions, real temperature,
                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                          const std::map<std::string, real> &ransConservative) const
    {
        Metrics().Communication()->LocalStage([&]
                                              {
            Metrics().Validate();
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            const auto rans = ResolveRANSValues(_data->equation, ransConservative);
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()) && _data->equation.HasMixture() == bool(workers),
                                  "equation initialization row space/worker profile differs");
            auto fractions = _data->equation.IndependentMassFractions(massFractions);
            for (auto &value : fractions) value *= primitive[0];
            auto evaluator = _data->equation.MakePointEvaluator(workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{}, {});
            const auto value = _data->motion.At(0).LocalState(evaluator.FromPrimitive(primitive, {fractions.data(), int(fractions.size())}, temperature));
            // Resolve/validate EOS input before obtaining any state writer.
            auto output = state.WriteOwned();
            const auto flow = output.Block("flow");
            for (index iCell = 0; iCell < flow.RowCount(); ++iCell)
                for (int j = 0; j < 5; ++j) flow(iCell, j) = value[j];
            if (!fractions.empty())
            {
                const auto species = output.Block("species");
                for (index iCell = 0; iCell < species.RowCount(); ++iCell)
                    for (index k = 0; k < index(fractions.size()); ++k) species(iCell, k) = fractions[k];
            }
            if (_data->equation.HasRANS())
            {
                const auto scalars = output.Block("rans");
                for (index iCell = 0; iCell < scalars.RowCount(); ++iCell)
                    for (index j = 0; j < scalars.ComponentNumber(); ++j) scalars(iCell, j) = rans[j];
            } });
    }

    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspace(EquationDofs &state,
                                                                          std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                                          WorkspaceOptions options) const
    {
        return BuildWorkspaceImpl(state, std::move(workers), false, options);
    }
    std::shared_ptr<FiniteVolume::ReconstructionState> FiniteVolume::BuildReconstructionState(EquationDofs &state) const
    {
        std::shared_ptr<ReconstructionState> result;
        Metrics().Communication()->LocalStage([&]
                                              {
            _data->equation.ValidateStateSchema(state);
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "reconstruction state belongs to another row space");
            result = std::make_shared<ReconstructionState>(_data, state); });
        return result;
    }
    std::shared_ptr<FiniteVolume::MeanWorkspace> FiniteVolume::BuildMeanWorkspace(EquationDofs &state,
                                                                                  std::optional<Physics::Chemistry::ChemicalWorkerPool> workers) const
    {
        auto inner = BuildWorkspaceImpl(state, std::move(workers), true);
        auto reconstruction = BuildReconstructionState(state);
        std::shared_ptr<MeanWorkspace> result;
        Metrics().Communication()->LocalStage([&]
                                              { result = std::make_shared<MeanWorkspace>(MeanWorkspace{std::move(reconstruction), std::move(inner)}); });
        return result;
    }
    std::shared_ptr<FiniteVolume::Workspace> FiniteVolume::BuildWorkspaceImpl(EquationDofs &state,
                                                                              std::optional<Physics::Chemistry::ChemicalWorkerPool> workers,
                                                                              bool meanOnly, WorkspaceOptions options) const
    {
        const MPIInfo mpi(Metrics().Communicator());
        const auto &n = _data->settings.numerics;
        // Resolve an operator-local fixed-width leaf without changing the one
        // EquationDofs workspace or outer schedule. Unsupported layouts and
        // physics retain the generic equation kernel.
        const bool flow5Profile = !_data->equation.HasMixture() && !_data->equation.HasRANS() &&
                                  state.ComponentNumber() == 5 && state.PhysicalFieldNumber() == 1;
        std::string operationControls;
        Metrics().Communication()->LocalStage([&]
                                              {
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "equation FV state belongs to another row space");
            DNDS_check_throw_info(options.rhsKernel != RHSKernelPolicy::FixedFlow5 || flow5Profile,
                                  "fixed flow5 RHS kernel requires one perfect-gas flow5 physical field");
            DNDS_check_throw_info((n.reconstruction == "constant" || n.reconstruction == "gradient" || n.reconstruction == "variational") &&
                                  n.reconstructionIterations >= 0, "invalid equation reconstruction controls");
            DNDS_check_throw_info(!_data->equation.HasMixture() || _data->fluxPolicy.rusanov,
                "extended EOS or stencil flux needs an explicit preparation provider");
            DNDS_check_throw_info(n.limiter == "none" || n.limiter == "Barth" || n.limiter == "LVR" ||
                                      (n.limiter == "WBAP" && (!n.characteristicLimiter || !_data->equation.HasMixture())),
                "requested extended FV flux/PP/limiter profile is not connected yet");
            DNDS_check_throw_info(!n.positivity || n.positivityReconstruction != "Segment" || flow5Profile,
                                  "Segment reconstruction PP is available only for one perfect-gas flow5 field");
            DNDS_check_throw_info(!_data->equation.Viscous() || n.viscousGradient == "polynomial" || flow5Profile,
                                  "extended viscosity requires the conservative polynomial gradient");
            DNDS_check_throw_info(_data->settings.equation.source == "none" ||
                (_data->settings.equation.source == "chemical" && _data->equation.HasMixture()), "unavailable equation volume source");
            for (const auto &sample : _data->samples)
                if (sample.other == UnInitIndex)
                    sample.boundary.ValidateEquationProducer();
            if (_data->equation.HasMixture())
            {
                DNDS_check_throw_info(workers, "mixture FV requires explicit chemistry workers");
                auto checked = _data->equation.MakePointEvaluator(workers->Acquire(), {});
            }
            else DNDS_check_throw_info(!workers, "perfect-gas FV does not require chemistry workers");
            operationControls = Json{{"meanOnly", meanOnly}, {"rhsKernel", int(options.rhsKernel)},
                {"numerics", _data->settings.numerics}, {"equation", _data->settings.equation},
                {"motion", _data->settings.motion}, {"boundaries", _data->settings.boundaries}, {"overlap", _data->settings.execution.overlap},
                {"collectiveChecks", _data->settings.implicit.collectiveChecks}, {"properties", _data->equation.PropertyKey().parameters},
                {"rans", _data->equation.HasRANS() ? Json(_data->equation.RANS().Settings()) : Json()}}.dump(); });
        AgreeEquationBinding(state, mpi.comm, operationControls);
        std::shared_ptr<Workspace> result;
        Metrics().Communication()->LocalStage([&]
                                              { result = std::make_shared<Workspace>(_data, state, std::move(workers), meanOnly); });
        result->BindHalos(state);
        if (flow5Profile && options.rhsKernel != RHSKernelPolicy::Generic)
            result->fixedFlowRHS.emplace(AcquireRHSOperator());
        auto request = CompleteRHSAuxiliaryRequest({});
        if (request.cellWaveJump || request.cellPressureMinimum)
        {
            request.meanThermodynamics = true;
            result->rhsAuxiliary.emplace(BuildAuxiliaryValues(request));
        }
        if (!meanOnly && _data->equation.HasRANS() && _data->equation.RANS().ComponentNumber() == 2)
            result->meanGradients.emplace(BuildMeanGradients(state));
        return result;
    }
}
