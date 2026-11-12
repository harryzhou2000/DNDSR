#include "CFV/CPU/DistributedInterfaceSamplingKernels.hpp"
#include "DNDS/Core/Errors.hpp"

namespace DNDS::CFV::CPU
{
    void EvaluateDirectDonors(const DirectDonorSamplingRequest &request)
    {
        DNDS_assert_info(request.dimension >= 0 && request.dimension <= 3 &&
                             request.cells.RowCount() == request.output.RowCount(),
                         "invalid direct donor sampling request");
        for (index iQuery = 0; iQuery < request.output.RowCount(); ++iQuery)
        {
            const index iCell = request.cells(iQuery, 0);
            Geom::tPoint center{};
            if (request.gradient)
                center = request.geometry.GetCellCent(iCell);
            for (index j = 0; j < request.output.Width(); ++j)
            {
                real value = request.means(iCell, j);
                if (request.gradient)
                    for (int d = 0; d < request.dimension; ++d)
                        value += request.gradients(
                                     iCell, j * request.dimension + d) *
                                 (request.points(iQuery, d) - center[d]);
                request.output(iQuery, j) = value;
            }
        }
    }
}
