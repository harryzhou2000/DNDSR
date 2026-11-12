#include "Cauchy/Interface/CPU/InterfacePacketKernels.hpp"
#include "DNDS/Core/Errors.hpp"

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
        for (index iPoint = 0; iPoint < request.rows.RowCount(); ++iPoint)
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
                continue;
            for (const index first : request.vectorStarts)
            {
                const std::array<real, 3> input{
                    destination(row.row, first),
                    destination(row.row, first + 1),
                    destination(row.row, first + 2)};
                std::array<real, 3> world = input, local = input;
                if (request.rotation.donorActive)
                    for (int d = 0; d < 3; ++d)
                    {
                        world[size_t(d)] = 0;
                        for (int e = 0; e < 3; ++e)
                            world[size_t(d)] += request.rotation.donor[size_t(3 * d + e)] * input[size_t(e)];
                    }
                local = world;
                if (request.rotation.receiverActive)
                    for (int d = 0; d < 3; ++d)
                    {
                        local[size_t(d)] = 0;
                        for (int e = 0; e < 3; ++e)
                            local[size_t(d)] += request.rotation.receiver[size_t(3 * d + e)] * world[size_t(e)];
                    }
                for (int d = 0; d < 3; ++d)
                {
                    destination(row.row, first + d) = local[size_t(d)];
                }
            }
        }
    }
}
