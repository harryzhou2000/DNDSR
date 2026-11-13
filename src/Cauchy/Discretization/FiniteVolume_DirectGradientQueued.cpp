#include "Cauchy/Discretization/FiniteVolumeWorkspaceInternal.hpp"
#include "DNDS/Algebra/DofAlgebra.hpp"

namespace DNDS::Cauchy
{
    void FiniteVolume::Workspace::SubmitDirectGradients(
        R::EagerEngine &engine, const R::ExecutionStream &stream,
        const EquationDofs &state, bool ghost)
    {
        DNDS_check_throw_info(engine.Owns(stream) && stream.Context().device == owner->metrics.Device(),
                              "direct-gradient submission stream or placement differs");
        FiniteVolume(owner).ValidateWorkspace(state, *this);
        const auto &reconstruction = owner->settings.numerics.reconstruction;
        if (reconstruction == "variational" || (meanOnly && !owner->equation.HasRANS()) ||
            (reconstruction == "constant" && directGradients.empty()))
            return;
        DNDS_check_throw_info(directGradients.size() == fields.size(), "direct-gradient fields are incomplete");
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            // Native direct-FV kernels already produce final physical
            // gradients. Publication belongs to the controller after drain.
            if (nativeDirectGradient)
                continue;
            const bool fromGradientField = (meanOnly || reconstruction == "constant") && fields[iField].gradient;
            if ((fromGradientField && ghost) || (reconstruction == "constant" && !fromGradientField))
                continue;
            auto &output = directGradients[iField];
            if (fromGradientField)
            {
                const auto &input = *fields[iField].gradient;
                if (input.Owned().SharedStorage() != output.Owned().SharedStorage())
                    Algebra::Copy(engine, stream, output, input);
                // P0 RANS source gradients share the already prepared field.
            }
            else
            {
                DNDS_check_throw_info(owner->reconstruction, "modal gradient extraction requires reconstruction geometry");
                owner->gradient.ExtractLinearGradients(engine, stream,
                                                       FluxCoefficients().PhysicalField(iField), *owner->reconstruction, output, ghost);
            }
        }
        if (!nativeDirectGradient && !directViscousGradients.empty())
        {
            DNDS_check_throw_info(directViscousGradients.size() == fields.size() && owner->reconstruction,
                                  "unlimited direct viscous-gradient fields or geometry are incomplete");
            for (index iField = 0; iField < index(fields.size()); ++iField)
                owner->gradient.ExtractLinearGradients(engine, stream,
                                                       Coefficients().PhysicalField(iField), *owner->reconstruction, directViscousGradients[iField], ghost);
        }
    }

    void FiniteVolume::Workspace::PublishDirectGradients(const EquationDofs &state, bool ghost, real time)
    {
        FiniteVolume(owner).ValidateWorkspace(state, *this);
        DNDS_check_throw_info(std::isfinite(time), "direct-gradient publication time is invalid");
        const auto &reconstruction = owner->settings.numerics.reconstruction;
        if (reconstruction == "variational" || (meanOnly && !owner->equation.HasRANS()) ||
            (reconstruction == "constant" && directGradients.empty()))
            return;
        DNDS_check_throw_info(directGradients.size() == fields.size() && directGradientPublications.size() == fields.size(),
                              "direct-gradient publication fields are incomplete");
        for (index iField = 0; iField < index(fields.size()); ++iField)
        {
            const auto &mean = ghost ? state.PhysicalField(iField).Ghost() : state.PhysicalField(iField).Owned();
            const auto &output = ghost ? directGradients[iField].Ghost() : directGradients[iField].Owned();
            auto &publication = directGradientPublications[iField][ghost ? 1 : 0];
            if (nativeDirectGradient)
            {
                publication.Publish(*owner, mean, mean, output, incoming, time);
                continue;
            }
            const bool fromGradientField = (meanOnly || reconstruction == "constant") && fields[iField].gradient;
            if (ransFirst < 0 || (fromGradientField && ghost) || (reconstruction == "constant" && !fromGradientField))
                continue;
            const auto &source = fromGradientField ? fields[iField].gradient->Owned()
                                 : ghost           ? FluxCoefficients().PhysicalField(iField).Ghost()
                                                   : FluxCoefficients().PhysicalField(iField).Owned();
            publication.Publish(*owner, mean, source, output, incoming, time);
        }
    }
}
