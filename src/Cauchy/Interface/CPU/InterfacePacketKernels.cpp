#include "Cauchy/Interface/CPU/InterfacePacketKernels.hpp"
#include "DNDS/Core/Errors.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::Cauchy::CPU
{
    void FillInterfacePacket(const InterfacePacketRequest &request)
    {
        DNDS_assert_info(!request.fields.empty(),
                         "invalid interface packet kernel request");
        for (const auto &field : request.fields)
        {
            DNDS_assert_info(!field.receiverComponents.empty() &&
                                 field.values.RowCount() == request.rows.RowCount(),
                             "invalid interface packet field view");
        }
        OMP::ForRows(0, request.rows.RowCount(), [&](index iPoint)
                     {
            const auto row = request.rows(iPoint, 0);
            const auto destination = row.faceCenter ? request.faceCenters
                                                    : request.quadrature;
            for (const auto &field : request.fields)
            {
                for (index j = 0; j < field.receiverComponents.size(); ++j)
                    destination(row.row, field.receiverComponents[j]) =
                        field.values(iPoint, j);
            }
            if (!request.rotation.donorActive && !request.rotation.receiverActive)
                return;
            for (const index first : request.vectorStarts)
            {
                const std::array<real, 3> input{
                    destination(row.row, first),
                    destination(row.row, first + 1),
                    destination(row.row, first + 2)};
                const auto local = request.rotation.Apply(input);
                for (int d = 0; d < 3; ++d)
                {
                    destination(row.row, first + d) = local[size_t(d)];
                }
            } });
    }
}
