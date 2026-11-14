#pragma once

#include "CFV/Plans/PolynomialSamplingPlan.hpp"

namespace DNDS::Cauchy::Detail
{
    /// PP queries preserve their canonical cell-major order while borrowing
    /// basis rows from face/source publications. Each segment writes a disjoint
    /// subset of the output. Setup establishes complete coverage; all segments
    /// are submitted on the same stream, with ordinary Runtime access retention.
    /// No mutable solution or sampled values belong to this immutable plan.
    struct QuadratureSampling
    {
        index rows = 0;
        std::vector<CFV::PolynomialSamplingPlan> segments;
        index Size() const { return rows; }
        void EvaluateValues(const CFV::FVField &mean, const CFV::FVField &coefficients,
                            Runtime::Array<real> &output) const
        {
            for (const auto &segment : segments)
                segment.EvaluateValues(mean, coefficients, output);
        }
        void Evaluate(Runtime::EagerEngine &engine, const Runtime::ExecutionStream &stream,
                      const CFV::FVField &mean, const CFV::FVField &coefficients,
                      Runtime::Array<real> &output) const
        {
            for (const auto &segment : segments)
                segment.Evaluate(engine, stream, mean, coefficients, output);
        }
        QuadratureSampling Placed(CFV::SamplingPlacementMemo &memo) const
        {
            auto result = *this;
            for (auto &segment : result.segments)
                segment = segment.Placed(memo);
            return result;
        }
    };

    /// Source-volume rule and weights, always shared with PP when it is active.
    /// PP has no independent rule. Center Jacobian queries remain distinct.
    struct SourceQuadrature
    {
        Runtime::Array<index> offsets{Runtime::RowLayout::Dense(0, 1)};
        Runtime::Array<real> weights{Runtime::RowLayout::Dense(0, 1)};
        std::optional<CFV::PolynomialSamplingPlan> sampling, jacobianSampling;
        bool valuesAreMeans = false;
    };

    /// One immutable hot-path publication per spatial plan and placement.
    /// Face quadrature coordinates/weights are in the shared CFV metric plan;
    /// faceBasis aliases its reconstruction basis publication. Source and PP
    /// samplers share that allocation or the same volume basis. With no source,
    /// volume PP owns a value-only cache of that same rule. VR matrix assembly
    /// uses independent full-derivative scratch, never this hot-path cache.
    struct FiniteVolumeQuadrature
    {
        Runtime::Array<real> faceBasis{Runtime::RowLayout::Dense(0, 0)};
        std::optional<QuadratureSampling> positivity;
        std::shared_ptr<const SourceQuadrature> source;
    };
}
