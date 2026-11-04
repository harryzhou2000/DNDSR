#pragma once

#include "DNDS/Algebra/DofAlgebra.hpp"
#include "Solver/StateGMRES.hpp"

namespace DNDS::Linear
{
    /// Historical native-field adapter; the Krylov mathematics is shared with
    /// logical equation states. No layout conversions or extra collectives.
    struct FieldKrylovOperations
    {
        using Field = Algebra::DofField<real>;
        static Field NewVector(const Field &prototype)
        {
            prototype.Validate();
            DNDS_check_throw_info(prototype.Device() == Runtime::Device::Host(), "FieldGMRES CPU profile cannot execute device vectors");
            return Field(prototype.Space(), Runtime::Array<real>(prototype.Owned().Layout()), Runtime::Array<real>(prototype.Ghost().Layout()));
        }
        static void CheckCompatible(const Field &prototype, const Field &field)
        {
            field.Validate();
            DNDS_check_throw_info(field.Space().Same(prototype.Space()) && field.Width() == prototype.Width() && field.Device() == prototype.Device(),
                                  "GMRES field space, width or backend differs");
        }
        static void RequireDistinct(const Field &source, const Field &workspace)
        {
            for (const auto *input : {&source.Owned(), &source.Ghost()})
                for (const auto *output : {&workspace.Owned(), &workspace.Ghost()})
                    DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "GMRES input aliases mutable workspace");
        }
        static void CopyGuess(Field &out, const Field &input)
        {
            if (out.Owned().Storage().LegacyPointer() == input.Owned().Storage().LegacyPointer() && out.Owned().Layout() == input.Owned().Layout())
                return;
            Algebra::Copy(out, input);
        }
        static void Copy(Field &out, const Field &input) { Algebra::Copy(out, input); }
        static void Fill(Field &out, real value) { Algebra::Fill(out, value); }
        static void Scale(Field &out, real value) { Algebra::Scale(out, value); }
        static void Axpy(Field &out, real value, const Field &x) { Algebra::Axpy(out, value, x); }
        static void LinearCombination(Field &out, real a, const Field &x, real b, const Field &y) { Algebra::LinearCombination(out, a, x, b, y); }
    };
    using FieldGMRES = StateGMRES<Algebra::DofField<real>, FieldKrylovOperations>;
}
