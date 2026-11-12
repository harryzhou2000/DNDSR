#include "Cauchy/Discretization/FiniteVolumeInternal.hpp"
#include "Cauchy/Physics/InitialField.hpp"
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
            _data->equation.ValidateStateSchema(state);
            _data->equation.ResolveKernelProfile(state.Device());
            DNDS_check_throw_info(state.Space().Same(Metrics().Space()), "initial field row space differs from mesh");
            InitialFieldEvaluator evaluator(_data->settings.initialField, _data->equation, ResolveReferenceScales(_data->settings.equation),
                                            workers ? workers->Acquire() : Physics::Chemistry::ChemicalWorkerPool::Lease{});
            const auto geometry = _data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
            const auto mesh = geometry.HostView();
            const auto frame = _data->motion.At(0);
            index width = 0;
            for (const auto &block : state.Schema().blocks) width += index(block.components.size());
            std::vector<real> background(width), average(width);
            auto output = state.WriteOwned();
            for (index iCell = 0; iCell < Metrics().NumCell(); ++iCell)
            {
                index offset = 0;
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto rows = output.Block(iBlock);
                    for (index j = 0; j < rows.ComponentNumber(); ++j) background[offset + j] = rows(iCell, j);
                    offset += rows.ComponentNumber();
                }
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
                    DNDS_check_throw_info(std::isfinite(jacobian) && jacobian > 0, "initial field requires positive finite cell Jacobians");
                    const real weight = jacobian * quadrature.GetWeight(iQ);
                    const auto value = evaluator.Evaluate(frame.WorldPoint({point[0], point[1], point[2]}), 0, iCell, background);
                    for (index j = 0; j < width; ++j) average[j] += weight * value[j];
                    volume += weight;
                }
                DNDS_check_throw_info(std::isfinite(volume) && volume > 0, "initial field has invalid cell volume");
                for (auto &value : average) value /= volume;
                std::copy_n(average.begin(), 5, flow.begin());
                flow = frame.LocalState(flow);
                std::copy(flow.begin(), flow.end(), average.begin());
                offset = 0;
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto rows = output.Block(iBlock);
                    for (index j = 0; j < rows.ComponentNumber(); ++j) rows(iCell, j) = average[offset + j];
                    offset += rows.ComponentNumber();
                }
            } });
    }
}
