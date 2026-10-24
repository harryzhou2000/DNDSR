#include "MeshViewKernel.hpp"

namespace DNDS::Geom::Test
{
    __global__ void LegacyMeshParityKernel(UnstructuredMeshView view, LegacyConstMeshReference legacy, int *mismatches)
    {
        if (blockIdx.x || threadIdx.x)
            return;
        int errors = 0;
        errors += view.NumNode() != legacy.coords.father.Size() || view.NumNodeGhost() != legacy.coords.son.Size();
        errors += view.NumCell() != legacy.cellNodes.father.Size() || view.NumCellGhost() != legacy.cellNodes.son.Size();
        for (index node = 0; node < legacy.coords.Size(); ++node)
            for (int d = 0; d < 3; ++d)
                errors += view.Coord(node)[d] != legacy.coords[node][d];
        for (index cell = 0; cell < legacy.cellNodes.Size(); ++cell)
        {
            const auto width = legacy.cellNodes.RowSize(cell);
            errors += view.CellNodes(cell).size() != size_t(width);
            errors += view.GetCellInfo(cell).type != legacy.cellInfo(cell, 0).type;
            errors += view.GetCellZone(cell) != legacy.cellInfo(cell, 0).zone;
            for (rowsize j = 0; j < width; ++j)
            {
                const auto node = legacy.cellNodes(cell, j);
                errors += view.CellNodes(cell)[j] != node;
                tPoint expected = legacy.coords[node];
                if (legacy.isPeriodic)
                {
                    auto bits = legacy.cellBits(cell, j);
                    errors += uint8_t(view.CellNodePeriodicBits(cell)[j]) != uint8_t(bits);
                    for (int b = 3; b >= 1; --b)
                        if (uint8_t(bits) & (uint8_t(1) << (b - 1)))
                            expected = legacy.periodic.TransCoord(expected, -b);
                }
                for (int d = 0; d < 3; ++d)
                    errors += view.GetCoordNodeOnCell(cell, j)[d] != expected[d];
            }
            const auto faces = legacy.cellFaces.RowSize(cell);
            errors += view.CellFaces(cell).size() != size_t(faces);
            for (rowsize slot = 0; slot < faces; ++slot)
            {
                const auto face = legacy.cellFaces(cell, slot);
                errors += view.CellFaces(cell)[slot] != face;
                bool back = legacy.faceCells(face, 0) == cell;
                if (back && legacy.faceCells(face, 1) == cell)
                    back = !bool(legacy.slotBits(cell, slot));
                errors += view.CellIsFaceBack(cell, face, slot) != back;
                errors += view.CellFaceOther(cell, face, slot) != legacy.faceCells(face, back ? 1 : 0);
            }
        }
        for (index face = 0; face < legacy.faceNodes.Size(); ++face)
        {
            const auto width = legacy.faceNodes.RowSize(face);
            errors += view.FaceNodes(face).size() != size_t(width);
            errors += view.GetFaceZone(face) != legacy.faceInfo(face, 0).zone;
            for (rowsize j = 0; j < width; ++j)
            {
                const auto node = legacy.faceNodes(face, j);
                errors += view.FaceNodes(face)[j] != node;
                tPoint expected = legacy.coords[node];
                if (legacy.isPeriodic)
                {
                    const auto bits = legacy.faceBits(face, j);
                    for (int b = 3; b >= 1; --b)
                        if (uint8_t(bits) & (uint8_t(1) << (b - 1)))
                            expected = legacy.periodic.TransCoord(expected, -b);
                }
                for (int d = 0; d < 3; ++d)
                    errors += view.GetCoordNodeOnFace(face, j)[d] != expected[d];
            }
        }
        *mismatches = errors;
    }

    cudaError_t LaunchLegacyMeshParity(UnstructuredMeshView view, LegacyConstMeshReference legacy, int *mismatches, cudaStream_t stream)
    {
        LegacyMeshParityKernel<<<1, 1, 0, stream>>>(view, legacy, mismatches);
        return cudaGetLastError();
    }

    __global__ void MeshViewProbeKernel(UnstructuredMeshView view, MeshViewProbeResult *output)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
            *output = MeshViewProbe(view);
    }

    cudaError_t LaunchMeshViewProbe(UnstructuredMeshView view, MeshViewProbeResult *deviceOutput, cudaStream_t stream)
    {
        MeshViewProbeKernel<<<1, 1, 0, stream>>>(view, deviceOutput);
        return cudaGetLastError();
    }
}
