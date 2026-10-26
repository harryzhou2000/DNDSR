#pragma once

#include "NativeMeshSmoothing.hpp"
#include "NativeMeshSmoothingMath.hpp"
#include "FieldTransfer.hpp"

namespace DNDS::Geom::SmoothingLegacyBridge
{
    inline void CheckAuthority(const MPIInfo &mpi, bool native)
    {
        int flags[]{int(native), -int(native)};
        MPI_Allreduce(MPI_IN_PLACE, flags, 2, MPI_INT, MPI_MAX, mpi.comm);
        DNDS_check_throw_info(flags[0] == -flags[1], "smoothing authority differs across ranks");
    }
    template <class T>
    NativeSmoothingOptions Options(const T &legacy)
    {
        NativeSmoothingOptions result;
        result.RBFRadius = legacy.RBFRadius;
        result.MaxIncludedAngle = legacy.MaxIncludedAngle;
        result.RBFPower = legacy.RBFPower;
        result.refDWall = legacy.refDWall;
        result.nIter = legacy.nIter;
        result.nSearch = legacy.nSearch;
        result.kernel = legacy.kernel;
        return result;
    }
}
