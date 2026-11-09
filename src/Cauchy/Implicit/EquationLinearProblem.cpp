#include "Cauchy/Implicit/LinearProblem.hpp"
#include "Cauchy/Implicit/LinearBlockKernels.hpp"
#include "Cauchy/State/EquationDofAlgebra.hpp"
#include "DNDS/MPI/Failure.hpp"

namespace DNDS::Cauchy
{
    namespace R = Runtime;
    void LinearProblemContext::JacobiImpl(const EquationDofs &rhs, const EquationDofs *input, EquationDofs &output, LinearRegion region) const
    {
        ValidateMatrix();
        ValidateOutputs(rhs, output);
        if (input)
            ValidateOutputs(*input, output);
        DNDS_check_throw_info(region == LinearRegion::All || region == LinearRegion::Core || region == LinearRegion::HaloDependent, "invalid Jacobi region");
        const auto diagonal = _diagonal.Read();
        const auto inverse = _inverse.Read();
        DNDS_check_throw_info(_inverseCurrent && _diagonal.Version() == _diagonalVersion && _inverse.Version() == _inverseVersion,
                              "Jacobi diagonal inverse is missing or stale");
        const auto right = rhs.ReadOwned();
        std::optional<EquationDofs::ReadScope> read;
        if (input)
            read.emplace(region != LinearRegion::Core && _pattern->hasGhostColumns ? input->ReadWithGhost() : input->ReadOwned());
        const auto b = right.ComponentLayout();
        const auto x = read ? read->ComponentLayout() : right.ComponentLayout();
        auto write = output.WriteOwned();
        const auto y = write.ComponentLayout();
        const auto inv = Algebra::MakeEigenRowMatrices(inverse.View<R::TableRows>(), _components, _diagonalStorage == DiagonalStorage::Block ? _components : 1);
        const auto blocks = ReadOffDiagonal();
        const auto &rows = region == LinearRegion::All ? _pattern->order : region == LinearRegion::Core ? _pattern->core
                                                                                                        : _pattern->dependent;
        try
        {
            blocks.WithActions([&](const auto &action)
                               {
            if (input)
                Detail::JacobiRows<false>(b.View().OwnedPart(), x.View().OwnedPart(), x.View().GhostPart(), y.View(), inv, action, *_pattern, rows, _components, _diagonalStorage == DiagonalStorage::Block);
            else
                Detail::JacobiRows<true>(b.View().OwnedPart(), x.View().OwnedPart(), x.View().GhostPart(), y.View(), inv, action, *_pattern, rows, _components, _diagonalStorage == DiagonalStorage::Block); });
        }
        catch (...)
        {
            write.Fail();
            MPI::AbortIfSolverFailure("Cauchy Jacobi kernel failed");
            throw;
        }
    }
    void LinearProblemContext::Jacobi(const EquationDofs &rhs, const EquationDofs &input, EquationDofs &output, LinearRegion region) const { JacobiImpl(rhs, &input, output, region); }
    void LinearProblemContext::JacobiZero(const EquationDofs &rhs, EquationDofs &output, LinearRegion region) const { JacobiImpl(rhs, nullptr, output, region); }
    void LinearProblemContext::ValidateField(const EquationDofs &field) const
    {
        Layout();
        field.Validate();
        DNDS_check_throw_info(field.Space().Same(_space) && field.ComponentNumber() == _components && field.Device() == R::Device::Host(),
                              "logical linear field space, component count or backend differs");
    }
    void LinearProblemContext::ValidateOutputs(const EquationDofs &input, const EquationDofs &output) const
    {
        ValidateField(input);
        ValidateField(output);
        input.RequireCompatible(output);
        EquationDofOperations::RequireDistinct(input, output);
        for (index iField = 0; iField < output.PhysicalFieldNumber(); ++iField)
        {
            const auto &destination = output.PhysicalField(iField).Owned();
            for (const auto *source : {&_diagonal, &_inverse})
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(destination.Storage(), source->Storage()), "logical linear output aliases matrix storage");
            if (_offDiagonal)
                DNDS_check_throw_info(!Algebra::Detail::Overlaps(destination.Storage(), _offDiagonal->Values().Storage()), "logical linear output aliases matrix storage");
        }
    }
    void LinearProblemContext::Apply(const EquationDofs &input, EquationDofs &output, LinearRegion region) const
    {
        ValidateMatrix();
        ValidateOutputs(input, output);
        DNDS_check_throw_info(region == LinearRegion::All || region == LinearRegion::Core || region == LinearRegion::HaloDependent, "invalid logical linear region");
        const auto read = region != LinearRegion::Core && _pattern->hasGhostColumns ? input.ReadWithGhost() : input.ReadOwned();
        const auto source = read.ComponentLayout();
        auto write = output.WriteOwned();
        const auto destination = write.ComponentLayout();
        const auto diagonal = _diagonal.Read();
        const auto offDiagonal = ReadOffDiagonal();
        const auto matrices = Algebra::MakeEigenRowMatrices(diagonal.View<R::TableRows>(), _components,
                                                            _diagonalStorage == DiagonalStorage::Block ? _components : 1);
        offDiagonal.WithActions([&](const auto &blocks)
                                { Detail::ApplyBlockRows(source.View().OwnedPart(), source.View().GhostPart(), destination.View(), matrices, blocks, *_pattern,
                                                         region == LinearRegion::All ? _pattern->order : region == LinearRegion::Core ? _pattern->core
                                                                                                                                      : _pattern->dependent,
                                                         _components, _diagonalStorage == DiagonalStorage::Block); });
    }
}
