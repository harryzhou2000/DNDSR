#include "CFV/CPU/DistributedInterfaceSamplingKernels.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV::CPU
{
    void EvaluateDirectDonors(const Detail::DirectDonorSamplingView &request)
    {
        request.ValidateDevice(Runtime::Device::Host());
        DNDS_assert_info(request.dimension >= 0 && request.dimension <= 3 &&
                             request.cells.RowCount() == request.output.RowCount(),
                         "invalid direct donor sampling request");
        OMP::ForRows(0, request.output.RowCount(), [&](index iQuery)
                     {
            const index iCell = request.cells(iQuery, 0);
            for (index j = 0; j < request.output.Width(); ++j)
            {
                real value = request.means(iCell, j);
                if (request.gradient)
                    for (int d = 0; d < request.dimension; ++d)
                        value += request.gradients(
                                     iCell, j * request.dimension + d) *
                                 (request.points(iQuery, d) - request.geometry(iCell, d + 1));
                request.output(iQuery, j) = value;
            } });
    }
}
