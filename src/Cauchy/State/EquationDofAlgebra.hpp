#pragma once

#include "Cauchy/State/EquationViews.hpp"
#include "DNDS/Algebra/LocalProducts.hpp"
#include <optional>

namespace DNDS::Cauchy
{
    /// Semantic adapter over DNDS pointwise algebra. These synchronous CPU
    /// operations touch owned values only, with no halo/collective or global
    /// repacking. All output reservations precede numerical writes. Binary
    /// operands must have distinct allocations; Scale is explicitly in-place.
    /// Caller serializes state rebinding; run/move the state owners with CPU work.
    struct EquationDofOperations
    {
        using State = EquationDofs;
        static State NewVector(const State &prototype)
        {
            prototype.Validate();
            // Allocation preserves the prototype backend independently of
            // arithmetic dispatch. This creates uninitialized scratch, with
            // no copy, host fallback or change to the equation grouping.
            const auto device = prototype.Device();
            Algebra::DofBundle fields(prototype.Space());
            for (index iField = 0; iField < prototype.PhysicalFieldNumber(); ++iField)
            {
                const auto &field = prototype.PhysicalField(iField);
                fields.Add(prototype.Fields().Name(iField), Algebra::DofField<real>(prototype.Space(),
                                                                                    Runtime::Array<real>(field.Owned().Layout(), device), Runtime::Array<real>(field.Ghost().Layout(), device)));
            }
            return prototype.RebindFields(std::move(fields));
        }
        static void CheckCompatible(const State &prototype, const State &state)
        {
            prototype.RequireCompatible(state);
            DNDS_check_throw_info(state.Device() == Runtime::Device::Host(), "CPU equation algebra has no device fallback");
        }
        static void RequireDistinct(const State &x, const State &y)
        {
            for (index iField = 0; iField < x.PhysicalFieldNumber(); ++iField)
                for (index jField = 0; jField < y.PhysicalFieldNumber(); ++jField)
                {
                    const auto &a = x.PhysicalField(iField), &b = y.PhysicalField(jField);
                    for (const auto *input : {&a.Owned(), &a.Ghost()})
                        for (const auto *output : {&b.Owned(), &b.Ghost()})
                            DNDS_check_throw_info(!Algebra::Detail::Overlaps(input->Storage(), output->Storage()), "equation algebra operands alias");
                }
        }
        struct Binding
        {
            State snapshot;
            std::vector<uint64_t> versions;
            explicit Binding(const State &state) : snapshot(state)
            {
                for (index iField = 0; iField < state.PhysicalFieldNumber(); ++iField)
                    versions.push_back(state.PhysicalField(iField).Owned().Version());
            }
            void Check(const State &state, bool unchangedOwned = false) const
            {
                snapshot.RequireCompatible(state);
                DNDS_check_throw_info(snapshot.PhysicalFieldNumber() == state.PhysicalFieldNumber(), "time callback changed physical equation grouping");
                for (index iBlock = 0; iBlock < state.BlockNumber(); ++iBlock)
                {
                    const auto before = snapshot.LocateBlock(iBlock), after = state.LocateBlock(iBlock);
                    const auto &a = snapshot.PhysicalField(before.physicalField), &b = state.PhysicalField(after.physicalField);
                    DNDS_check_throw_info(before.firstComponent == after.firstComponent &&
                                              a.Owned().SharedStorage() == b.Owned().SharedStorage() && a.Ghost().SharedStorage() == b.Ghost().SharedStorage(),
                                          "time callback rebound an equation block");
                    DNDS_check_throw_info(!unchangedOwned || versions[before.physicalField] == b.Owned().Version(),
                                          "time callback modified an owned equation input/source");
                }
            }
        };

    private:
        static bool SameOwnedBinding(const State &a, const State &b)
        {
            for (index iBlock = 0; iBlock < a.BlockNumber(); ++iBlock)
            {
                const auto x = a.LocateBlock(iBlock), y = b.LocateBlock(iBlock);
                if (x.firstComponent != y.firstComponent ||
                    a.PhysicalField(x.physicalField).Owned().SharedStorage() != b.PhysicalField(y.physicalField).Owned().SharedStorage())
                    return false;
            }
            return true;
        }
        static void Apply(Algebra::PointwiseOperation operation, State &output, const State *x, const State *y, real a, real b = 0)
        {
            CheckCompatible(output, output);
            if (x)
            {
                CheckCompatible(output, *x);
                if (operation == Algebra::PointwiseOperation::Copy && SameOwnedBinding(output, *x))
                    return;
                RequireDistinct(output, *x);
            }
            if (y)
            {
                CheckCompatible(output, *y);
                RequireDistinct(output, *y);
            }
            std::optional<State::ReadScope> readX, readY;
            if (x)
                readX.emplace(x->ReadOwned());
            if (y)
                readY.emplace(y->ReadOwned());
            auto write = output.WriteOwned();
            const EquationViewLayout<real> destination(output, write);
            if (readY)
            {
                const EquationViewLayout<const real> left(*x, *readX), right(*y, *readY);
                const auto kernel = [&](auto out, auto u, auto v)
                { Algebra::ApplyHostRows(operation, out, u, v, a, b); };
                DispatchCPUEquationLayouts(kernel, destination, left, right);
            }
            else if (readX)
            {
                const EquationViewLayout<const real> source(*x, *readX);
                const auto kernel = [&](auto out, auto u)
                { Algebra::ApplyHostRows(operation, out, u, EquationBlockedView<const real>{}, a, b); };
                DispatchCPUEquationLayouts(kernel, destination, source);
            }
            else
            {
                const auto kernel = [&](auto out)
                { Algebra::ApplyHostRows(operation, out, EquationBlockedView<const real>{}, EquationBlockedView<const real>{}, a, b); };
                DispatchCPUEquationLayouts(kernel, destination);
            }
        }

    public:
        static void Copy(State &out, const State &x) { Apply(Algebra::PointwiseOperation::Copy, out, &x, nullptr, 0); }
        static void CopyGuess(State &out, const State &x) { Copy(out, x); }
        static void Fill(State &out, real value) { Apply(Algebra::PointwiseOperation::Fill, out, nullptr, nullptr, value); }
        static void Scale(State &out, real value) { Apply(Algebra::PointwiseOperation::Scale, out, nullptr, nullptr, value); }
        static void Axpy(State &out, real value, const State &x) { Apply(Algebra::PointwiseOperation::Axpy, out, &x, nullptr, value); }
        static void LinearCombination(State &out, real a, const State &x, real b, const State &y)
        {
            Apply(Algebra::PointwiseOperation::Combination, out, &x, &y, a, b);
        }
        /// Owned local dot-product batch in logical row/component order. The
        /// caller owns its communicator/reduction and distributed error policy.
        /// Descriptors/scopes are resolved once; no per-element schema checks,
        /// halo access or global payload flattening occurs.
        static std::vector<real> LocalProducts(const State &x, const std::vector<const State *> &probes,
                                               const std::vector<real> &inverseScales = {})
        {
            CheckCompatible(x, x);
            DNDS_check_throw_info(inverseScales.empty() || index(inverseScales.size()) == x.ComponentNumber(), "equation product scale width differs");
            for (const auto *probe : probes)
            {
                DNDS_check_throw_info(probe, "null equation product probe");
                CheckCompatible(x, *probe);
            }
            const auto read = x.ReadOwned();
            const EquationViewLayout<const real> values(x, read);
            std::vector<State::ReadScope> readers;
            readers.reserve(probes.size());
            for (const auto *probe : probes)
                readers.push_back(probe->ReadOwned());
            std::vector<real> result(probes.size(), 0);
            for (index k = 0; k < index(probes.size()); ++k)
            {
                const EquationViewLayout<const real> input(*probes[k], readers[k]);
                const auto kernel = [&](auto u, auto v)
                { result[k] = Algebra::Detail::AccumulateLocalProduct(u, v, u.RowCount(), u.ComponentNumber(), inverseScales); };
                DispatchCPUEquationLayouts(kernel, values, input);
            }
            return result;
        }
    };
}
