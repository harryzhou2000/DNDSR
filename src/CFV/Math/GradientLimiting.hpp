#pragma once

#include "CFV/Detail/GradientKernels.hpp"
#include "CFV/Math/WBAPScalar.hpp"

namespace DNDS::CFV::Math
{
    /** One physical-gradient 3WBAP level, without polynomial transfer matrices.
     * Neighbor gradients rotate on both spatial and selected equation axes.
     * Only [first,first+count) uses face characteristics; count==0 never reads
     * matrix data. All other components are limited as scalars. The frozen
     * input is never overwritten, so selected cells are independent.
     * Optional static component/dimension bounds must match the input row;
     * validated CUDA wrappers supply them, while CPU callers use runtime bounds.
     * @pre scratch has at least six times the gradient row width elements.
     */
    template <int StaticComponents = 0, int StaticDimension = 0, class Vectors>
    DNDS_DEVICE_CALLABLE void LimitGradientCell(const Detail::GradientLimiterView &plan, Vectors vectors,
                                                const Detail::GradientLimiterValues &values, Detail::GradientOutput output,
                                                index iCell, SpanView<real> scratch) noexcept
    {
        const auto center = values.input[iCell];
        const auto result = output[iCell];
        const index dim = StaticDimension ? StaticDimension : plan.dimension;
        const index width = StaticComponents ? StaticComponents * dim : center.size();
        const index components = StaticComponents ? StaticComponents : width / dim;
        for (index j = 0; j < width; ++j)
            result[j] = center[j];
        if (values.activated)
        {
            DNDS_HD_assert(std::isfinite(values.indicator(iCell, 0)));
            if (values.indicator(iCell, 0) < values.threshold)
                return;
        }
        const auto other = scratch.Subspan(0, width), limited = scratch.Subspan(width, width),
                   up = scratch.Subspan(2 * width, width), down = scratch.Subspan(3 * width, width),
                   valid = scratch.Subspan(4 * width, width), temporary = scratch.Subspan(5 * width, width);
        for (index j = 0; j < width; ++j)
        {
            up[j] = down[j] = values.centerWeight;
            valid[j] = 1;
        }
        const real epsilon = std::pow(verySmallReal, .25);
        for (index entry = plan.offsets(iCell, 0); entry < plan.offsets(iCell + 1, 0); ++entry)
        {
            const rowsize slot = rowsize(entry - plan.offsets(iCell, 0));
            const index iFace = plan.topology.CellFaces(iCell)[slot];
            const index neighbor = plan.topology.CellFaceOther(iCell, iFace, slot);
            if (neighbor == UnInitIndex)
                continue;
            const bool rotated = plan.periodic.Rotates(iFace);
            const int side = plan.topology.CellIsFaceBack(iCell, iFace, slot) ? 0 : 1;
            const auto rotation = rotated ? plan.periodic.OtherRotation(iFace, side) : Geom::tGPoint::Identity().eval();
            if (!rotated)
                for (index j = 0; j < width; ++j)
                    other[j] = values.input(neighbor, j);
            else
                for (index component = 0; component < components; ++component)
                {
                    const index first = vectors.First(component);
                    for (index d = 0; d < dim; ++d)
                    {
                        real value = 0;
                        if (first < 0)
                            for (index k = 0; k < dim; ++k)
                                value += rotation(d, k) * values.input(neighbor, component * dim + k);
                        else
                            for (index v = 0; v < 3; ++v)
                                for (index k = 0; k < dim; ++k)
                                    value += rotation(component - first, v) *
                                             values.input(neighbor, (first + v) * dim + k) * rotation(d, k);
                        other[component * dim + d] = value;
                    }
                }
            const index first = values.first, count = values.count;
            if (count)
            {
                const auto forward = values.forward[iFace];
                for (index d = 0; d < dim; ++d)
                    for (index i = 0; i < count; ++i)
                    {
                        real a = 0, b = 0;
                        for (index j = 0; j < count; ++j)
                        {
                            const real transform = forward[i + count * j];
                            a += transform * center[(first + j) * dim + d];
                            b += transform * other[(first + j) * dim + d];
                        }
                        limited[(first + i) * dim + d] = WBAPBiway(a, b);
                    }
            }
            for (index component = 0; component < components; ++component)
                if (component < first || component >= first + count)
                    for (index d = 0; d < dim; ++d)
                        limited[component * dim + d] = WBAPBiway(center[component * dim + d], other[component * dim + d]);
            if (count)
            {
                const auto inverse = values.inverse[iFace];
                for (index d = 0; d < dim; ++d)
                {
                    for (index i = 0; i < count; ++i)
                    {
                        temporary[i] = 0;
                        for (index j = 0; j < count; ++j)
                            temporary[i] += inverse[i + count * j] * limited[(first + j) * dim + d];
                    }
                    for (index i = 0; i < count; ++i)
                        limited[(first + i) * dim + d] = temporary[i];
                }
            }
            for (index j = 0; j < width; ++j)
            {
                const real value = limited[j];
                if (value == 0)
                {
                    valid[j] = 0;
                    continue;
                }
                const real ratio = WBAPClampedRatio(center[j], value, epsilon);
                const real r2 = ratio * ratio;
                up[j] += r2 * ratio;
                down[j] += r2 * r2;
            }
        }
        for (index j = 0; j < width; ++j)
            result[j] = valid[j] == 0 ? 0 : center[j] * up[j] / down[j];
    }
}
