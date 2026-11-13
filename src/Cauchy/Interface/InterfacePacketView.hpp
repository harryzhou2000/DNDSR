#pragma once
#include "DNDS/Runtime/Array/ArrayView.hpp"
#include "DNDS/Core/Span.hpp"
#include <array>

namespace DNDS::Cauchy
{
    /// One unique receiver destination in either packet table. Construction
    /// validates row bounds and uniqueness before parallel numerical execution.
    struct InterfacePacketRow
    {
        index row = 0;
        bool faceCenter = false;
    };

    /// Row-major transforms: donor reference axes -> world -> receiver axes.
    /// These rotate components, not the geometric query or the reference mesh.
    struct InterfacePacketRotation
    {
        std::array<real, 9> donor{1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::array<real, 9> receiver{1, 0, 0, 0, 1, 0, 0, 0, 1};
        bool donorActive = false, receiverActive = false;

        DNDS_DEVICE_CALLABLE std::array<real, 3> Apply(std::array<real, 3> input) const
        {
            std::array<real, 3> world = input;
            if (donorActive)
                for (int d = 0; d < 3; ++d)
                {
                    world[d] = 0;
                    for (int e = 0; e < 3; ++e)
                        world[d] += donor[3 * d + e] * input[e];
                }
            std::array<real, 3> local = world;
            if (receiverActive)
                for (int d = 0; d < 3; ++d)
                {
                    local[d] = 0;
                    for (int e = 0; e < 3; ++e)
                        local[d] += receiver[3 * d + e] * world[e];
                }
            return local;
        }
    };

    /** Combined-AoS packet leaf I/O. The operator owns the routing arrays,
     * schema-sized metadata and source/output leases. Metadata spans point to
     * the selected device; their owning MetadataBuffer validates placement.
     * Every logical component is supplied exactly once by receiverComponents,
     * vectorStarts names disjoint length-three blocks in receiver ordering.
     * Unselected output rows are preserved. No MPI, lease acquisition, host
     * callbacks or physics calculations belong in this mapping kernel.
     */
    struct InterfacePacketTableView
    {
        Runtime::ArrayView<const InterfacePacketRow, Runtime::AccessBoundsChecks, Runtime::FixedRows<1>> rows;
        Runtime::ArrayView<const real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> values;
        Runtime::ArrayView<real, Runtime::AccessBoundsChecks, Runtime::UniformTableRows> quadrature, faceCenters;
        SpanView<const index> receiverComponents, vectorStarts;
        InterfacePacketRotation rotation;
        void ValidateDevice(Runtime::Device device) const
        {
            Runtime::ValidateViewDevices(device, rows, values, quadrature, faceCenters);
        }
    };
}
