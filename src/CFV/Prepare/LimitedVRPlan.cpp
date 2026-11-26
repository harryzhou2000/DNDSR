#include "CFV/Plans/LimitedVRPlan.hpp"
#include "CFV/Detail/ReconstructionData.hpp"
#include "DNDS/Core/OMP.hpp"

namespace DNDS::CFV
{
    LimitedVRPlan LimitedVRPlan::Prepare(FiniteVolumePlan metrics)
    {
        LimitedVRPlan result(std::move(metrics));
        DNDS_check_throw_info(result.Device() == Runtime::Device::Host(), "LVR topology preparation requires host metrics");
        auto read = result._metrics._data->mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux());
        const auto mesh = read.HostView();
        result._offsets = Runtime::Array<index>(Runtime::RowLayout::Dense(result._metrics.NumCell() + 1, 1));
        result._offsets.SetName("CFV/limitedVR/offsets", DNDS_ALLOCATION_SITE());
        auto offsets = result._offsets.Write();
        const auto o = offsets.View<Runtime::FixedRows<1>>();
        const index count = OMP::ExclusivePrefixRows(
            result._metrics.NumCell(),
            [&](index iCell) -> index
            { return index(mesh.CellFaces(iCell).size()); },
            [&](index iCell, index start)
            { o(iCell, 0) = start; });
        o(result._metrics.NumCell(), 0) = count;
        result._neighbors = Runtime::Array<index>(Runtime::RowLayout::Dense(count, 1));
        result._boundary = Runtime::Array<index>(Runtime::RowLayout::Dense(result._metrics.NumCell(), 1));
        result._neighbors.SetName("CFV/limitedVR/neighbors", DNDS_ALLOCATION_SITE());
        result._boundary.SetName("CFV/limitedVR/boundaryMask", DNDS_ALLOCATION_SITE());
        auto neighbors = result._neighbors.Write(), boundary = result._boundary.Write();
        const auto n = neighbors.View<Runtime::FixedRows<1>>(), b = boundary.View<Runtime::FixedRows<1>>();
        std::vector<uint8_t> valid(size_t(result._metrics.NumCell()), 1);
        OMP::ForRows(0, result._metrics.NumCell(), [&](index iCell)
                     {
            index position = o(iCell, 0);
            b(iCell, 0) = 0;
            for (index slot = 0; slot < index(mesh.CellFaces(iCell).size()); ++slot)
            {
                const auto other = mesh.CellFaceOther(iCell, mesh.CellFaces(iCell)[slot], rowsize(slot));
                valid[iCell] &= other == UnInitIndex || (other >= 0 && other < result._metrics.NumCellProc());
                n(position++, 0) = other;
                b(iCell, 0) |= other == UnInitIndex;
            } });
        OMP::ForSetupRows(0, result._metrics.NumCell(), [&](index iCell)
                          { DNDS_check_throw_info(valid[iCell], "LVR requires complete face-neighbor ghosts"); });
        const auto &core = result._partition.Core();
        OMP::ForSetupRows(0, index(core.size()), [&](index iRow)
                          {
            const index iCell = core[iRow];
            for (index entry = o(iCell, 0); entry < o(iCell + 1, 0); ++entry)
                DNDS_check_throw_info(n(entry, 0) == UnInitIndex || n(entry, 0) < result._metrics.NumCell(), "LVR core depends on ghost input"); });
        return result;
    }
}
