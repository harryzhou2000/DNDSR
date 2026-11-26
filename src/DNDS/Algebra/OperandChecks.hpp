#pragma once

#include "DNDS/Core/Errors.hpp"
#include "DNDS/Runtime/Array/Array.hpp"

#include <initializer_list>
#include <iterator>

namespace DNDS::Algebra::Detail
{
    /// Validate that every non-null output Array is pairwise disjoint from
    /// every other non-null output and from every non-null input Array.
    template <class OutputArray, class InputArray = OutputArray>
    void RequirePairwiseDisjoint(
        std::initializer_list<const OutputArray *> outputs,
        std::initializer_list<const InputArray *> inputs,
        const char *label = "aliased array operand")
    {
        for (auto outIt = outputs.begin(); outIt != outputs.end(); ++outIt)
        {
            const auto *out = *outIt;
            if (!out)
                continue;
            const auto outStorage = out->Storage();
            for (auto otherIt = std::next(outIt); otherIt != outputs.end(); ++otherIt)
            {
                const auto *other = *otherIt;
                if (other)
                    DNDS_check_throw_info(out->Identity() != other->Identity() && !outStorage.Overlaps(other->Storage()), label);
            }
            for (const auto *in : inputs)
                if (in)
                    DNDS_check_throw_info(out->Identity() != in->Identity() && !outStorage.Overlaps(in->Storage()), label);
        }
    }

    /// Range overload for dynamic lists of Runtime::Array<T>.
    template <class OutputArrayRange, class InputArrayRange>
    void RequirePairwiseDisjointArrays(
        const OutputArrayRange &outputs,
        const InputArrayRange &inputs,
        const char *label = "aliased array operand")
    {
        const auto outBegin = std::begin(outputs);
        const auto outEnd = std::end(outputs);
        for (auto outIt = outBegin; outIt != outEnd; ++outIt)
        {
            const auto outStorage = outIt->Storage();
            for (auto otherIt = std::next(outIt); otherIt != outEnd; ++otherIt)
                DNDS_check_throw_info(outIt->Identity() != otherIt->Identity() && !outStorage.Overlaps(otherIt->Storage()), label);
            for (const auto &in : inputs)
                DNDS_check_throw_info(outIt->Identity() != in.Identity() && !outStorage.Overlaps(in.Storage()), label);
        }
    }
}
