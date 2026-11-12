#include "Cauchy/Discretization/FiniteVolume.hpp"
#include "Cauchy/Core/NumericalFailure.hpp"
#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Input/InitialField.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include "Geom/Quadratures/Quadrature.hpp"

namespace DNDS::Cauchy
{
    namespace
    {
        // Setup-only programmable input evaluation belongs at the checked
        // operator boundary. Exprtk and physical-input conversion may reject
        // user data; these exceptions must never escape a noexcept CPU kernel.
        NumericalStatus InitializeProgrammedField(Geom::UnstructuredMeshView mesh, Algebra::DofComponentView<real> rows,
                                                  const RigidFrame &frame, InitialFieldEvaluator &evaluator)
        {
            const index width = rows.ComponentNumber();
            std::vector<real> background(width), average(width);
            for (index iCell = 0; iCell < rows.RowCount(); ++iCell)
            {
                for (index j = 0; j < width; ++j)
                    background[j] = rows(iCell, j);
                FlowState flow;
                std::copy_n(background.begin(), 5, flow.begin());
                flow = frame.WorldState(flow);
                std::copy(flow.begin(), flow.end(), background.begin());
                std::fill(average.begin(), average.end(), 0);
                Geom::tSmallCoords coords;
                mesh.GetCoordsOnCell(iCell, coords);
                const auto element = mesh.GetCellElement(iCell);
                constexpr int order = Geom::Elem::INT_ORDER_MAX;
                const Geom::Elem::Quadrature quadrature(element, order);
                real volume = 0;
                for (int iQ = 0; iQ < quadrature.GetNumPoints(); ++iQ)
                {
                    const auto &shape = Geom::Elem::GetCachedShapeAtQuadrature(element.type, order, iQ);
                    const Geom::tPoint point = Geom::Elem::PPhysicsCoordD01Nj(coords, shape);
                    const real jacobian = Geom::Elem::CellJacobianDet(mesh.Dimension(), coords, shape);
                    if (!(std::isfinite(jacobian) && jacobian > 0))
                        return NumericalStatus{NumericalError::InvalidState, 1, iCell};
                    const real weight = jacobian * quadrature.GetWeight(iQ);
                    const auto value = evaluator.Evaluate(frame.WorldPoint({point[0], point[1], point[2]}), 0, iCell, background);
                    for (index j = 0; j < width; ++j)
                        average[j] += weight * value[j];
                    volume += weight;
                }
                if (!(std::isfinite(volume) && volume > 0))
                    return NumericalStatus{NumericalError::InvalidState, 2, iCell};
                for (auto &value : average)
                    value /= volume;
                std::copy_n(average.begin(), 5, flow.begin());
                flow = frame.LocalState(flow);
                std::copy(flow.begin(), flow.end(), average.begin());
                for (index j = 0; j < width; ++j)
                    rows(iCell, j) = average[j];
            }
            return {};
        }
    }
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
            auto output = state.WriteOwned();
            const auto layout = output.ComponentLayout();
            RequireNumericalSuccess(InitializeProgrammedField(mesh, layout.View(), frame, evaluator), "InitializeProgrammedField"); });
    }
}
