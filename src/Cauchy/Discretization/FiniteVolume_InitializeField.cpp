#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/CPU/FiniteVolumeStateKernels.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Input/InitialField.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::InitializeField(EquationDofs &state, const std::optional<Physics::Chemistry::ChemicalWorkerPool> &workers) const
    {
        MPI::LocalStage(MPIInfo(Metrics().Communicator()), [&]
                        {
            if (_data->settings.initialField.Empty()) return;
            DNDS_check_throw_info(_data->settings.initialProfile.kind == "constant", "spatial overrides require a constant background profile");
            Metrics().Validate();
            state.RequireSchema(_data->equation.Schema());
            _data->equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "initial field row space differs from mesh");
            InitialFieldEvaluator evaluator(_data->settings.initialField, _data->equation, ResolveReferenceScales(_data->settings.equation),
                                            workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
            const auto geometry = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = geometry.HostView();
            const auto frame = _data->motion.At(0);
            index width = 0;
            for (const auto &block : state.Schema().blocks) width += index(block.components.size());
            auto output = state.WriteOwned();
            const auto layout = output.ComponentLayout();
            RequireNumericalSuccess(CPU::InitializeProgrammedField(mesh, layout.View(), frame, evaluator), "InitializeProgrammedField"); });
    }
}
