#pragma once

#include "Geom/Access/MeshView.hpp"

namespace DNDS::CFV::Detail
{
    /// Map a shared face point into its selected parent's coordinate frame.
    inline Geom::tPoint FacePointInCell(const Geom::UnstructuredMeshView &mesh, index iFace, int side, const Geom::tPoint &point)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if (side == 1 && Geom::FaceIDIsPeriodicMain(zone))
            return mesh.Periodic().TransCoord(point, zone);
        if (side == 1 && Geom::FaceIDIsPeriodicDonor(zone))
            return mesh.Periodic().TransCoordBack(point, zone);
        return point;
    }
    inline Geom::tPoint OtherPointInCell(const Geom::UnstructuredMeshView &mesh, index iFace, int side, const Geom::tPoint &point)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if ((side == 1 && Geom::FaceIDIsPeriodicMain(zone)) || (side == 0 && Geom::FaceIDIsPeriodicDonor(zone)))
            return mesh.Periodic().TransCoord(point, zone);
        if ((side == 0 && Geom::FaceIDIsPeriodicMain(zone)) || (side == 1 && Geom::FaceIDIsPeriodicDonor(zone)))
            return mesh.Periodic().TransCoordBack(point, zone);
        return point;
    }
    /// Orthogonal map from the other parent to this parent's frame.
    inline Geom::tGPoint OtherFrameInCell(const Geom::UnstructuredMeshView &mesh, index iFace, int side)
    {
        const auto zone = mesh.GetFaceZone(iFace);
        if (!Geom::FaceIDIsPeriodic(zone))
            return Geom::tGPoint::Identity();
        const int ordinal = Geom::FaceIDIsPeriodicMain(zone) ? -zone : -zone - 3;
        const Geom::tGPoint rotation = mesh.Periodic().rotation.at(ordinal).map();
        if ((side == 1 && Geom::FaceIDIsPeriodicMain(zone)) || (side == 0 && Geom::FaceIDIsPeriodicDonor(zone)))
            return rotation;
        return rotation.transpose();
    }
    inline void ValidateVectorBlocks(index components, const std::vector<index> &starts)
    {
        index end = 0;
        for (auto first : starts)
        {
            DNDS_check_throw_info(first >= end && first <= components && components - first >= 3,
                                  "periodic vector triples must be sorted, disjoint and within the component range");
            end = first + 3;
        }
    }
}
