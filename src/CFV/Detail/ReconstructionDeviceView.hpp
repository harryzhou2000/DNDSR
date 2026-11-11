#pragma once

#include "DNDS/Runtime/Array/ArrayView.hpp"

namespace DNDS::CFV::Detail
{
    /// Immutable numeric publication; all pointers have the placement of the
    /// resident plan. No host owner, Eigen object, map or functor reaches kernels.
    struct ReconstructionDeviceView
    {
        Runtime::ArrayView<const index> offsets, neighbors, faces, boundaryOffsets;
        Runtime::ArrayView<const uint8_t> rotationActive;
        Runtime::ArrayView<const real> matrixAInvB, vectorAInvB, secondary, boundaryInjection, rotation;
        Runtime::ArrayView<const index> core, dependent;
        index owned = 0, n = 0;
        real relax = 1, wbapCenterWeight = 10;
        bool rotated = false;
    };
    struct DeviceWorkSelection
    {
        Runtime::ArrayView<const index> rows;
        index count = 0;
        bool indirect = false;
        DNDS_DEVICE_CALLABLE index Cell(index i) const { return indirect ? rows(i, 0) : i; }
    };
    struct DeviceCharacteristicView
    {
        Runtime::ArrayView<const real> forward, inverse;
        index first = 0, count = 0;
    };
    struct DevicePeriodicBlocks
    {
        Runtime::ArrayView<const index> starts;
        DNDS_DEVICE_CALLABLE index First(index component) const
        {
            for (index i = 0; i < starts.RowCount(); ++i)
            {
                const auto first = starts(i, 0);
                if (component >= first && component < first + 3)
                    return first;
            }
            return -1;
        }
    };
}
