#pragma once

#include "Geom/Geometry/NativeMeshSmoothing.hpp"
#include "Geom/Geometry/NativeMeshSmoothingMath.hpp"
#include "Geom/Transition/FieldTransfer.hpp"

namespace DNDS::Geom::NativeSmoothingFacade
{
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
