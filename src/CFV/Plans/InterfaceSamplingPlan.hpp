#pragma once

#include "FiniteVolumePlan.hpp"
#include "Geom/Interface/BoundaryInterface.hpp"

namespace DNDS::CFV
{
    /// CFV owns numerical sampling; Geom supplies only donor correspondence.
    /// Retains the immutable map and plan. Output follows original query order.
    /// Values are in the donor's physical frame; frame/velocity conversion belongs
    /// to the equation adapter. A pose or mesh update requires another plan.
    class InterfaceSamplingPlan
    {
        VariationalReconstructionPlan _reconstruction;
        Geom::InterfacePointMap _map;
        Runtime::Array<index> _cells;
        Runtime::Array<real> _points;

    public:
        InterfaceSamplingPlan(VariationalReconstructionPlan reconstruction, Geom::InterfacePointMap map)
            : _reconstruction(std::move(reconstruction)), _map(std::move(map)),
              _cells(Runtime::RowLayout::Dense(_map.Size(), 1)), _points(Runtime::RowLayout::Dense(_map.Size(), 3))
        {
            _map.Validate();
            _reconstruction.Validate();
            DNDS_check_throw_info(_map.Donor().Cells().SameOrder(_reconstruction.Metrics().Cells()),
                                  "interface donors and reconstruction belong to different mesh/order");
            auto cells = _cells.Write();
            auto points = _points.Write();
            const auto c = cells.View<Runtime::FixedRows<1>>();
            const auto p = points.View<Runtime::FixedRows<3>>();
            for (index i = 0; i < _map.Size(); ++i)
            {
                const auto match = _map.Get(i);
                c(i, 0) = match.donorCell;
                for (int d = 0; d < 3; ++d)
                    p(i, d) = match.donorPoint[d];
            }
        }
        index Size() const { return _map.Size(); }
        void Evaluate(const FVField &mean, const FVField &coefficients, Runtime::Array<real> &values) const
        {
            _map.Validate();
            _reconstruction.Evaluate(mean, coefficients, _cells, _points, values);
        }
    };
}
