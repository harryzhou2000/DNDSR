#pragma once

#include "Cauchy/Physics/StateInput.hpp"
#include "DNDS/Config/ExprtkWrapper.hpp"

namespace DNDS::Cauchy
{
    /// Serial, reusable setup evaluator. Geometry supplies a world point in
    /// code-length units; velocity components are absolute in world axes.
    /// Boxes -> planes -> expressions, each in config order. Later matches
    /// override earlier values at the point, before conservative averaging.
    /// Owns the compiled expressions and one EOS worker. Use a separate
    /// instance per concurrent CPU worker; no hidden OMP/global state.
    class InitialFieldEvaluator
    {
        struct Box
        {
            Vector3 lower, upper;
            std::vector<real> value;
        };
        struct Plane
        {
            Vector3 normal;
            real offset;
            std::vector<real> value;
        };
        struct Expression
        {
            Physics::StateInput::Form form;
            bool physical;
            std::unique_ptr<DNDS::ExprtkWrapperEvaluator> program;
        };
        PhysicalStateConverter _converter;
        index _width;
        std::vector<Box> _boxes;
        std::vector<Plane> _planes;
        std::vector<Expression> _expressions;

    public:
        InitialFieldEvaluator(const InitialFieldSettings &settings, const EquationSystem &equation,
                              const Physics::ReferenceScales &scales,
                              Physics::Chemistry::ChemicalWorkerPool::Lease worker = {});
        /// background is a complete conservative world-frame state. t is in
        /// code-time units; iCell is rank-local and for diagnostics only (do not
        /// use it to specify a partition-independent field). Every expression
        /// receives a fresh seed and inRegion=0; stateType never rescales x/t.
        std::vector<real> Evaluate(const Vector3 &point, real time, index iCell, const std::vector<real> &background);
    };
}
