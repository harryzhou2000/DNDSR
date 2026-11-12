#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "Cauchy/Physics/EquationPositivity.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    using namespace EquationFVDetail;

    void FiniteVolume::Workspace::RefreshDirectGradients(const EquationDofs &state, bool ghost,
                                                         real time)
    {
        if (owner->settings.numerics.reconstruction == "variational")
            return;
        if (owner->settings.numerics.reconstruction == "constant" && directGradients.empty())
            return;
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            if (nativeDirectGradient)
            {
                const auto &mean = ghost
                                       ? state.PhysicalField(iField).Ghost()
                                       : state.PhysicalField(iField).Owned();
                const auto &output = ghost
                                         ? directGradients[iField].Ghost()
                                         : directGradients[iField].Owned();
                auto &published = directGradientPublications[iField][ghost ? 1 : 0];
                published.Publish(*owner, mean, mean, output, incoming, time);
                continue;
            }
            const bool fromGradientField = owner->settings.numerics.reconstruction == "constant" && fields[iField].gradient;
            // Constant face fluxes never read remote source gradients.
            if (fromGradientField && ghost)
                continue;
            // Pure P0 has no polynomial or physical-gradient publication.
            // Its directGradients slot is retained only for the common
            // workspace shape; no modal zero field is manufactured.
            if (owner->settings.numerics.reconstruction == "constant" && !fromGradientField)
                continue;
            const auto refresh = [&]
            {
                if (fromGradientField)
                {
                    // The cache check may retain a compatible input read.
                    // Use the explicit engine entry: the immediate algebra
                    // wrapper deliberately requires completely idle fields.
                    R::EagerEngine engine;
                    const auto stream = engine.CreateCPUStream();
                    Algebra::Copy(engine, stream, directGradients[iField], *fields[iField].gradient);
                    engine.Synchronize();
                }
                else
                {
                    DNDS_check_throw_info(owner->reconstruction,
                                          "modal gradient extraction requires variational reconstruction");
                    owner->gradient.ExtractLinearGradients(FluxCoefficients().PhysicalField(iField), *owner->reconstruction,
                                                           directGradients[iField], ghost);
                }
            };
            // Only RANS source sampling repeats this pass. Flow-only and
            // chemical-only paths retain their original preparation path.
            if (ransFirst < 0)
            {
                refresh();
                continue;
            }
            const auto &source = fromGradientField ? fields[iField].gradient->Owned()
                                 : ghost           ? FluxCoefficients().PhysicalField(iField).Ghost()
                                                   : FluxCoefficients().PhysicalField(iField).Owned();
            const auto &mean = ghost ? state.PhysicalField(iField).Ghost()
                                     : state.PhysicalField(iField).Owned();
            const auto &destination = ghost ? directGradients[iField].Ghost() : directGradients[iField].Owned();
            // Pin the input through both the check and a possible refresh:
            // an unfinished writer has not published its next generation.
            auto &published = directGradientPublications[iField][ghost ? 1 : 0];
            if (published.Current(*owner, mean, source, destination,
                                  incoming, time))
            {
                owner->metrics.Validate();
                continue;
            }
            // The independently callable source path still executes this
            // on a miss. P0 sources keep their physical gradients, and P1
            // extracts the final limiter/PP coefficients, never raw slopes.
            refresh();
            published.Publish(*owner, mean, source, destination, incoming,
                              time);
        }
        if (!nativeDirectGradient && !directViscousGradients.empty())
        {
            DNDS_check_throw_info(directViscousGradients.size() == fields.size(),
                                  "unlimited direct viscous-gradient fields are incomplete");
            for (index iField = 0; iField < index(fields.size()); ++iField)
            {
                DNDS_check_throw_info(owner->reconstruction,
                                      "modal viscous-gradient extraction requires variational reconstruction");
                owner->gradient.ExtractLinearGradients(Coefficients().PhysicalField(iField), *owner->reconstruction,
                                                       directViscousGradients[iField], ghost);
            }
        }
    }

    /// Cauchy-owned explicit-P1 componentwise Barth stage. CFV supplies
    /// the unlimited physical gradients; equation block layout, periodic
    /// vector components and conservative state semantics remain here.
    void FiniteVolume::Workspace::PrepareDirectGradientLimiter(const EquationDofs &state, bool barth)
    {
        DNDS_check_throw_info(nativeDirectGradient && directGradients.size() == fields.size(),
                              "direct-P1 limiter workspace is incomplete");
        const int dim = owner->metrics.Dimension();
        const auto meanScope = state.ReadWithGhost();
        const auto meanLayout = meanScope.ComponentLayout();
        const auto means = meanLayout.View();
        std::vector<R::Array<real>::ReadLease> sourceReads;
        std::vector<R::Array<real>::WriteLease> destinationWrites;
        sourceReads.reserve(fields.size());
        destinationWrites.reserve(fields.size());
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            DNDS_check_throw_info(fields[iField].gradient,
                                  "direct-P1 limiter has no unlimited gradient");
            DNDS_check_throw_info(
                fields[iField].gradient->Owned().SharedStorage() !=
                    directGradients[iField].Owned().SharedStorage(),
                "direct-P1 limiter needs distinct input and output fields");
            sourceReads.emplace_back(fields[iField].gradient->Owned().Read());
            destinationWrites.emplace_back(directGradients[iField].Owned().Write());
        }
        using ReadView = R::ArrayView<const real, R::AccessBoundsChecks,
                                      R::UniformTableRows>;
        using WriteView = R::ArrayView<real, R::AccessBoundsChecks,
                                       R::UniformTableRows>;
        std::vector<ReadView> source;
        std::vector<WriteView> destination;
        source.reserve(fields.size());
        destination.reserve(fields.size());
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            source.push_back(sourceReads[iField].View<R::UniformTableRows>());
            destination.push_back(destinationWrites[iField].View<R::UniformTableRows>());
        }
        const index components = state.ComponentNumber();
        DNDS_check_throw_info(
            directLimiterScratch.size() == size_t(6 * components),
            "direct-P1 limiter scratch extent differs");
        real *mean = directLimiterScratch.data();
        real *neighbor = mean + components;
        real *lower = neighbor + components;
        real *upper = lower + components;
        real *positive = upper + components;
        real *negative = positive + components;
        const auto gradient = [&](index logical, index iCell, int d)
        {
            const index field = logicalField[size_t(logical)];
            const index component = logicalComponent[size_t(logical)];
            return source[field](iCell, component * dim + d);
        };
        for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
        {
            for (index j = 0; j < components; ++j)
            {
                mean[j] = means(iCell, j);
                lower[j] = upper[j] = mean[j];
                positive[j] = negative[j] = 0;
            }
            if (barth)
                for (index entry = owner->offsets[iCell];
                     entry < owner->offsets[iCell + 1]; ++entry)
                {
                    const auto &sample = owner->samples[size_t(entry)];
                    if (sample.other != UnInitIndex)
                    {
                        for (index j = 0; j < components; ++j)
                            neighbor[j] = means(sample.other, j);
                        if (sample.changesFrame)
                            for (const index first : vectorStarts)
                            {
                                const Geom::tPoint value{
                                    neighbor[first], neighbor[first + 1],
                                    neighbor[first + 2]};
                                const auto rotated = sample.rotation * value;
                                for (int d = 0; d < 3; ++d)
                                    neighbor[first + d] = rotated[d];
                            }
                        for (index j = 0; j < components; ++j)
                        {
                            lower[j] = std::min(lower[j], neighbor[j]);
                            upper[j] = std::max(upper[j], neighbor[j]);
                        }
                    }
                    const auto displacement = sample.ownPoint -
                                              owner->metrics.GetCellCent(iCell);
                    for (index j = 0; j < components; ++j)
                    {
                        real increment = 0;
                        for (int d = 0; d < dim; ++d)
                            increment += gradient(j, iCell, d) * displacement[d];
                        positive[j] = std::max(positive[j], increment);
                        negative[j] = std::min(negative[j], increment);
                    }
                }
            for (index j = 0; j < components; ++j)
            {
                real alpha = 1;
                if (barth)
                {
                    alpha = std::min(alpha,
                                     (upper[j] - mean[j]) /
                                         (positive[j] + verySmallReal));
                    alpha = std::min(alpha,
                                     (mean[j] - lower[j]) /
                                         (-negative[j] + verySmallReal));
                    alpha = std::max(real(0), std::min(real(1), alpha));
                }
                const index field = logicalField[size_t(j)];
                const index component = logicalComponent[size_t(j)];
                for (int d = 0; d < dim; ++d)
                    destination[field](iCell, component * dim + d) =
                        alpha * gradient(j, iCell, d);
            }
        }
    }

    /// Equation-aware P1 positivity stage. One beta scales the flow5 and
    /// transported-species gradients; RANS uses its independent scalar
    /// policy. This method deliberately lives in Cauchy rather than CFV.
    void FiniteVolume::Workspace::ApplyDirectGradientPositivity(const EquationDofs &state)
    {
        DNDS_check_throw_info(nativeDirectGradient &&
                                  owner->settings.numerics.positivity &&
                                  !owner->positivityOffsets.empty() &&
                                  owner->positivityDisplacements.size() ==
                                      size_t(owner->positivityOffsets.back()),
                              "direct-P1 positivity workspace is incomplete");
        const int dim = owner->metrics.Dimension();
        const index components = state.ComponentNumber();
        const auto meanScope = state.ReadOwned();
        // DofComponentView borrows its descriptor table. Retain the
        // owning layout for the complete pointwise PP traversal.
        const auto meanLayout = meanScope.ComponentLayout();
        const auto means = meanLayout.View();
        std::vector<R::Array<real>::WriteLease> gradientWrites;
        gradientWrites.reserve(directGradients.size());
        for (auto &field : directGradients)
            gradientWrites.emplace_back(field.Owned().Write());
        using WriteView = R::ArrayView<real, R::AccessBoundsChecks,
                                       R::UniformTableRows>;
        std::vector<WriteView> gradients;
        for (auto &write : gradientWrites)
            gradients.push_back(write.View<R::UniformTableRows>());
        const auto bounds = thermal.View();
        const auto rans = owner->equation.HasRANS()
                              ? ResolveRANSReconstructionPolicy(owner->equation)
                              : RANSReconstructionView{};
        // PP POLICY -- direct-P1 publication. This is the same locked
        // flow5/species beta and clipped EOS/T endpoint as modal PP, applied
        // directly to physical gradients. It imposes no simplex bound on
        // raw rhoY gradients; RANS uses its separate model scalar.
        struct PointValues
        {
            const std::vector<real> *values;
            index width;
            real operator()(index point, index component) const
            {
                return (*values)[size_t(point * width + component)];
            }
        } higher{&directPPSamples, components};
        struct MeanValues
        {
            decltype(means) values;
            index cell;
            real operator()(index, index component) const
            {
                return values(cell, component);
            }
        } base{means, 0};
        directPPLimited = 0;
        directPPMinimum = 1;
        for (index iCell = 0; iCell < owner->metrics.NumCell(); ++iCell)
        {
            const EquationPositivityDetail::PointRow<decltype(means)> mean{
                means, iCell};
            DNDS_check_throw_info(bounds.Admissible(mean),
                                  "direct-P1 PP requires an admissible cell mean");
            const index begin = owner->positivityOffsets[iCell];
            const index end = owner->positivityOffsets[iCell + 1];
            const index points = end - begin;
            for (index q = 0; q < points; ++q)
                for (index j = 0; j < components; ++j)
                {
                    const index field = logicalField[size_t(j)];
                    const index component = logicalComponent[size_t(j)];
                    real value = mean[j];
                    const auto &displacement =
                        owner->positivityDisplacements[size_t(begin + q)];
                    for (int d = 0; d < dim; ++d)
                        value += gradients[field](iCell, component * dim + d) *
                                 displacement[d];
                    directPPSamples[size_t(q * components + j)] = value;
                }
            base.cell = iCell;
            const auto clippedPolicy = owner->settings.numerics.positivityClippedReferencePolicy == "backtracking"
                                           ? ReconstructionClippedReferencePolicy::Backtracking
                                           : ReconstructionClippedReferencePolicy::FallbackToBase;
            real beta = bounds.ReconstructionFraction(
                mean, base, higher, 0, points,
                owner->settings.numerics.positivityBetaPower,
                owner->settings.numerics.positivityBetaCutoff, 1,
                clippedPolicy);
            DNDS_check_throw_info(std::isfinite(beta) && beta >= 0 && beta <= 1,
                                  "invalid direct-P1 thermal beta");
            const real ransScale = rans.Compute(base, higher, ransFirst, 0, points);
            DNDS_check_throw_info(std::isfinite(ransScale) && ransScale >= 0 &&
                                      ransScale <= 1,
                                  "invalid direct-P1 RANS factor");
            for (index j = 0; j < components; ++j)
            {
                real scale = bounds.CompressesReconstructionComponent(j) ? beta : 1;
                if (rans.components && j >= ransFirst &&
                    j < ransFirst + rans.components)
                    scale = ransScale;
                if (scale == 1)
                    continue;
                const index field = logicalField[size_t(j)];
                const index component = logicalComponent[size_t(j)];
                for (int d = 0; d < dim; ++d)
                    gradients[field](iCell, component * dim + d) *= scale;
            }

            // PP POLICY -- verify the actual scaled gradient storage. Raw
            // rhoY remains the advection/diffusion polynomial; only this
            // predicate constructs the temporary clipped EOS/T composition.
            struct ScratchRow
            {
                const std::vector<real> *values;
                index first;
                real operator[](index j) const { return (*values)[size_t(first + j)]; }
            };
            for (int iteration = 0; iteration < 1000; ++iteration)
            {
                bool clippedGood = true;
                for (index q = 0; q < points; ++q)
                {
                    for (index j = 0; j < components; ++j)
                    {
                        const index field = logicalField[size_t(j)];
                        const index component = logicalComponent[size_t(j)];
                        real value = mean[j];
                        const auto &displacement =
                            owner->positivityDisplacements[size_t(begin + q)];
                        for (int d = 0; d < dim; ++d)
                            value += gradients[field](iCell, component * dim + d) * displacement[d];
                        directPPSamples[size_t(q * components + j)] = value;
                    }
                    const ScratchRow point{&directPPSamples, q * components};
                    clippedGood = clippedGood && bounds.ClippedReferenceAdmissible(point);
                }
                if (clippedGood)
                    break;
                const real previous = beta;
                if (clippedPolicy == ReconstructionClippedReferencePolicy::FallbackToBase)
                    beta = 0;
                else
                    beta *= .9;
                if (beta < owner->settings.numerics.positivityBetaCutoff)
                    beta = 0;
                DNDS_check_throw_info(previous > 0,
                                      "direct-P1 mean base violates its final physical-state predicate");
                const real ratio = beta / previous;
                for (index j = 0; j < components; ++j)
                    if (bounds.CompressesReconstructionComponent(j))
                    {
                        const index field = logicalField[size_t(j)];
                        const index component = logicalComponent[size_t(j)];
                        for (int d = 0; d < dim; ++d)
                            gradients[field](iCell, component * dim + d) *= ratio;
                    }
            }
            if (beta < 1)
            {
                ++directPPLimited;
                directPPMinimum = std::min(directPPMinimum, beta);
            }
            for (index q = 0; q < points; ++q)
            {
                for (index j = 0; j < components; ++j)
                {
                    const index field = logicalField[size_t(j)];
                    const index component = logicalComponent[size_t(j)];
                    real value = mean[j];
                    const auto &displacement =
                        owner->positivityDisplacements[size_t(begin + q)];
                    for (int d = 0; d < dim; ++d)
                        value += gradients[field](iCell, component * dim + d) * displacement[d];
                    directPPSamples[size_t(q * components + j)] = value;
                }
                const ScratchRow point{&directPPSamples, q * components};
                DNDS_check_throw_info(bounds.ClippedReferenceAdmissible(point),
                                      "published direct-P1 gradient violates its clipped EOS/T bound");
            }
        }
    }

    void FiniteVolume::Workspace::PreparePrimitiveGradient(const EquationDofs &state, CFV::ReconstructionRegion region)
    {
        if (!primitiveGradient)
            return;
        DNDS_assert_info(primitive && primitiveBoundary && state.PhysicalFieldNumber() == 1,
                         "primitive-gradient workspace is incomplete");
        const auto fill = [&](const R::Array<real> &source, R::Array<real> &destination)
        {
            const auto read = source.Read();
            auto write = destination.Write();
            const auto from = read.View<R::UniformTableRows>();
            const auto to = write.View<R::UniformTableRows>();
            for (index iCell = 0; iCell < to.RowCount(); ++iCell)
            {
                FlowState conservative;
                for (int j = 0; j < 5; ++j)
                    conservative[j] = from(iCell, j);
                const auto value = owner->equation.Gas().Primitive(conservative);
                const real converted[5]{value.density, value.velocity[0], value.velocity[1], value.velocity[2], value.temperature};
                for (int j = 0; j < 5; ++j)
                    to(iCell, j) = converted[j];
            }
        };
        if (region != CFV::ReconstructionRegion::HaloDependent)
            fill(state.PhysicalField(0).Owned(), primitive->Owned());
        if (region != CFV::ReconstructionRegion::Core)
            fill(state.PhysicalField(0).Ghost(), primitive->Ghost());
        owner->gradient.Evaluate(*primitive, *primitiveBoundary, *primitiveGradient, {1}, region);
    }

}
