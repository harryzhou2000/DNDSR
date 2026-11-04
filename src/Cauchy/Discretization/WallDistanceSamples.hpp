#pragma once

#include "Geom/Access/MeshAccess.hpp"

namespace DNDS::Cauchy
{
    namespace Detail
    {
        struct WallDistanceData;
    }

    /// Borrowed CPU spans, indexed by local cell/face and quadrature ordinal.
    /// Retain WallDistanceRead through every use. CellMean is the arithmetic
    /// quadrature mean used by Euler's mean-state turbulence source and wall BC.
    struct WallDistanceView
    {
        Runtime::ArrayView<const real, false, Runtime::CSRRows> cells, faces;
        Runtime::ArrayView<const real, false, Runtime::FixedRows<1>> cellMean;
    };

    /// Pins the mesh generation, nodal source and immutable sampled buffers.
    /// All dependency checks occur at acquisition, outside numerical loops.
    class WallDistanceRead
    {
        std::shared_ptr<const Detail::WallDistanceData> _data;
        Geom::MeshReadAccess _mesh;
        Runtime::Array<real>::ReadLease _cells, _faces, _mean;
        explicit WallDistanceRead(std::shared_ptr<const Detail::WallDistanceData> data);
        friend class WallDistanceSamples;

    public:
        WallDistanceView HostView() const;
    };

    /// Optional CPU cache of a Geom-owned nodal wall-distance field. This does
    /// not search walls or create another wall-distance provider. Scalar nodal
    /// distances (norms of Geom's wall vectors) are interpolated by the mesh's
    /// P1/P2 shape functions, including mid-edge/face/interior nodes. Periodic
    /// vector rotations do not change their norms. Negative higher-order
    /// overshoot is clipped to the explicit nonnegative minimumDistance.
    ///
    /// Construct and AcquireRead are rank-local and make no MPI calls; all
    /// required nodal ghosts must already be published by Geom. Call inside a
    /// coordinated setup stage if subsequent work is collective. Copies share
    /// the immutable publication. Rebuild after wall, geometry, topology or
    /// distribution edits; acquisition rejects a stale publication.
    class WallDistanceSamples
    {
        std::shared_ptr<const Detail::WallDistanceData> _data;
        explicit WallDistanceSamples(std::shared_ptr<const Detail::WallDistanceData> data)
            : _data(std::move(data)) {}

    public:
        static WallDistanceSamples Construct(ssp<Geom::UnstructuredMesh> mesh, int intOrder,
                                             real minimumDistance = 0);
        WallDistanceRead AcquireRead() const;
        int IntegrationOrder() const;
    };
}
