#pragma once

#include "Geom/Mesh/MeshView.hpp"

namespace DNDS::Geom::Test
{
    DNDS_DEVICE_CALLABLE inline index MeshViewProbeIndex(index value)
    {
        if (value == UnInitIndex)
            return -97;
        // Convert before negating so even the most negative encoded index is
        // handled with defined unsigned arithmetic. Keep test weights bounded.
        if (value < 0)
            return -100 - index((uint64_t(0) - uint64_t(value)) % 1000003);
        return value % 1000003;
    }

    struct MeshViewProbeResult
    {
        index counts[15]{}; // Node, cell, face, boundary, edge: owned/ghost/proc.
        index relationChecksum = 0;
        index elementChecksum = 0;
        index faceSideChecksum = 0;
        real coordinateChecksum = 0;
        real wallDistanceChecksum = 0;
        real periodicChecksum = 0;
        real cellBatchChecksum = 0, cellExternalChecksum = 0, cellNodeChecksum = 0;
        real faceBatchChecksum = 0, faceExternalChecksum = 0, faceNodeChecksum = 0;
        real transformedCellSample[3]{}, transformedFaceSample[3]{};
    };

    /// Shared host/CUDA parity calculation; fixture tests supply independent
    /// expected connectivity and geometry in addition to comparing executions.
    DNDS_DEVICE_CALLABLE inline MeshViewProbeResult MeshViewProbe(UnstructuredMeshView view)
    {
        MeshViewProbeResult result;
        const index counts[]{view.NumNode(), view.NumNodeGhost(), view.NumNodeProc(),
                             view.NumCell(), view.NumCellGhost(), view.NumCellProc(),
                             view.NumFace(), view.NumFaceGhost(), view.NumFaceProc(),
                             view.NumBnd(), view.NumBndGhost(), view.NumBndProc(),
                             view.NumEdge(), view.NumEdgeGhost(), view.NumEdgeProc()};
        for (int i = 0; i < 15; ++i)
            result.counts[i] = counts[i];
        const index relationRows[]{view.NumCellProc(), view.NumBndProc(), view.NumBndProc(),
                                   view.NumCellProc(), view.NumNodeProc(), view.NumNodeProc(),
                                   view.NumCellProc(), view.NumFaceProc(), view.NumFaceProc(),
                                   view.NumFaceProc(), view.NumBndProc(), view.NumCellProc(),
                                   view.NumEdgeProc(), view.NumEdgeProc(), view.NumCellProc()};
        for (int r = 0; r < static_cast<int>(MeshRelation::Count); ++r)
        {
            auto relation = static_cast<MeshRelation>(r);
            if (!view.HasRelation(relation))
                continue;
            for (index row = 0; row < relationRows[r]; ++row)
            {
                auto entries = view.Relation(relation, row);
                result.relationChecksum += (r + 1) * (row + 1) * index(entries.size() + 1);
                size_t j = 0;
                for (const auto &entry : entries)
                {
                    result.relationChecksum += (r + 1) * (row + 1) * index(j + 1) * MeshViewProbeIndex(entry);
                    ++j;
                }
            }
        }
        for (index node = 0; node < view.NumNodeProc(); ++node)
            for (int d = 0; d < 3; ++d)
            {
                result.coordinateChecksum += (node + 1) * (d + 1) * view.Coord(node)[d];
                if (view.HasNodeWallDist())
                    result.wallDistanceChecksum += (node + 1) * (d + 1) * view.NodeWallDist(node)[d];
            }
        for (index cell = 0; cell < view.NumCellProc(); ++cell)
        {
            result.elementChecksum += (cell + 1) * (view.GetCellInfo(cell).type + 31 * view.GetCellZone(cell) +
                                                    47 * view.GetCellElement(cell).GetNumNodes());
            MeshQueryCoords coords;
            view.GetCoordsOnCell(cell, coords);
            MeshQueryCoords externalCoords;
            view.GetCoordsOnCell(cell, externalCoords, view);
            if (cell == 0 && coords.cols() > 2)
                for (int d = 0; d < 3; ++d)
                    result.transformedCellSample[d] = coords(d, 2);
            for (rowsize j = 0; j < coords.cols(); ++j)
                for (int d = 0; d < 3; ++d)
                {
                    result.periodicChecksum += (cell + 1) * (j + 1) * (d + 1) *
                                               (coords(d, j) + externalCoords(d, j) + view.GetCoordNodeOnCell(cell, j)[d]);
                    result.cellBatchChecksum += (cell + 1) * (j + 1) * (d + 1) * coords(d, j);
                    result.cellExternalChecksum += (cell + 1) * (j + 1) * (d + 1) * externalCoords(d, j);
                    result.cellNodeChecksum += (cell + 1) * (j + 1) * (d + 1) * view.GetCoordNodeOnCell(cell, j)[d];
                    if (view.HasNodeWallDist())
                        result.wallDistanceChecksum += (cell + 1) * (j + 1) * (d + 1) * view.GetCoordWallDistOnCell(cell, j)[d];
                }
            if (view.HasRelation(MeshRelation::CellFaces))
            {
                auto faces = view.CellFaces(cell);
                for (size_t j = 0; j < faces.size(); ++j)
                {
                    if (faces[j] < 0 || faces[j] == UnInitIndex)
                        continue;
                    result.faceSideChecksum += (cell + 1) * index(j + 1) *
                                               (7 * view.CellIsFaceBack(cell, faces[j], j) + MeshViewProbeIndex(view.CellFaceOther(cell, faces[j], j)));
                }
            }
        }
        for (index face = 0; face < view.NumFaceProc(); ++face)
        {
            result.elementChecksum += (face + 1) * (view.GetFaceInfo(face).type + 37 * view.GetFaceZone(face) +
                                                    53 * view.GetFaceElement(face).GetNumNodes());
            MeshQueryCoords coords;
            view.GetCoordsOnFace(face, coords);
            MeshQueryCoords externalCoords;
            view.GetCoordsOnFace(face, externalCoords, view);
            if (face == 0 && coords.cols() > 1)
                for (int d = 0; d < 3; ++d)
                    result.transformedFaceSample[d] = coords(d, 1);
            for (rowsize j = 0; j < coords.cols(); ++j)
                for (int d = 0; d < 3; ++d)
                {
                    result.periodicChecksum += (face + 1) * (j + 1) * (d + 1) *
                                               (coords(d, j) + externalCoords(d, j) + view.GetCoordNodeOnFace(face, j)[d]);
                    result.faceBatchChecksum += (face + 1) * (j + 1) * (d + 1) * coords(d, j);
                    result.faceExternalChecksum += (face + 1) * (j + 1) * (d + 1) * externalCoords(d, j);
                    result.faceNodeChecksum += (face + 1) * (j + 1) * (d + 1) * view.GetCoordNodeOnFace(face, j)[d];
                    if (view.HasNodeWallDist())
                        result.wallDistanceChecksum += (face + 1) * (j + 1) * (d + 1) * view.GetCoordWallDistOnFace(face, j)[d];
                }
        }
        for (index bnd = 0; bnd < view.NumBndProc(); ++bnd)
            result.elementChecksum += (bnd + 1) * (view.GetBndInfo(bnd).type + 41 * view.GetBndZone(bnd) +
                                                   59 * view.GetBndElement(bnd).GetNumNodes());
        for (index edge = 0; edge < view.NumEdgeProc(); ++edge)
            result.elementChecksum += (edge + 1) * (view.GetEdgeInfo(edge).type + 43 * view.GetEdgeZone(edge) +
                                                    61 * view.GetEdgeElement(edge).GetNumNodes());
        return result;
    }

#ifdef DNDS_USE_CUDA
    /// Read-only reference bundle assembled exclusively from the established
    /// const ArrayPair CUDA views. It owns no new implementation of those views.
    struct LegacyConstMeshReference
    {
        tCoordPair::t_deviceViewConst<DeviceBackend::CUDA> coords;
        ArrayPair<ParArray<index, NonUniformSize>>::t_deviceViewConst<DeviceBackend::CUDA> cellNodes, faceNodes, cellFaces;
        ArrayPair<ParArray<index, 2>>::t_deviceViewConst<DeviceBackend::CUDA> faceCells;
        tElemInfoArrayPair::t_deviceViewConst<DeviceBackend::CUDA> cellInfo, faceInfo;
        tPbiPair::t_deviceViewConst<DeviceBackend::CUDA> cellBits, faceBits, slotBits;
        Periodicity periodic;
        bool isPeriodic;
    };
    cudaError_t LaunchLegacyMeshParity(UnstructuredMeshView view, LegacyConstMeshReference legacy,
                                       int *mismatches, cudaStream_t stream);
    /// Caller retains view leases through stream completion and supplies a device
    /// output pointer. No synchronization or implicit host copy is performed.
    cudaError_t LaunchMeshViewProbe(UnstructuredMeshView view, MeshViewProbeResult *deviceOutput, cudaStream_t stream);
#endif
}
