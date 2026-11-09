"""Extract the current Euler RHS body for an opt-in standalone diagnostic.

No production file is edited. The generated subclass exposes face-only,
source-only and composed calls on frozen, already exchanged inputs. The face
and source loops are copied, not rewritten. Anchors fail closed when the source
changes. Gradient construction, integration recording and all internal clocks
are excluded explicitly; the driver supplies gradients and times whole calls.
"""
import argparse
import hashlib
import pathlib


def unique(text, anchor):
    if text.count(anchor) != 1:
        raise ValueError(f"expected one extraction anchor: {anchor!r}")
    return text.index(anchor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    source = args.source.read_text()
    signature = unique(source, "    void EulerEvaluator<model>::EvaluateRHS(")
    begin = source.index("\n    {", signature) + len("\n    {")
    end = source.rindex("\n    }\n}")
    body = source[begin:end]
    # The frozen-gradient contract excludes preparation and its hidden halo.
    begin_gradient = unique(body, "        // direct2ndRec = true;")
    end_gradient = unique(body, "        auto faceOp =")
    # Preserve the OMP face buffer declaration, which follows preparation.
    buffer = body.index("#if defined(DNDS_DIST_MT_USE_OMP)", begin_gradient)
    body = body[:begin_gradient] + body[buffer:end_gradient] + body[end_gradient:]
    # The preceding slice still contains the gradient block up to the first
    # OMP directive only when the expected production block shape changes.
    if "LimiterUGrad(" in body or "DoReconstruction2ndGrad(" in body:
        raise ValueError("gradient preparation leaked into extracted operator")
    for clock in ("        double t0 = MPI_Wtime();", "        double t1 = MPI_Wtime();", "        double t2 = MPI_Wtime();"):
        unique(body, clock)
        body = body.replace(clock, "")
    body = "\n".join(line for line in body.splitlines() if "DNDS_MPI_InsertCheck(" not in line)
    # Legacy RHS includes historical instrumentation within the face loop.
    # Remove those calls in the extracted copy, not just its entry clocks.
    body = "\n".join(line for line in body.splitlines()
                     if not ("PerformanceTimer" in line and ("StartTimer(" in line or "StopTimer(" in line)))
    faces = unique(body, "        auto faceOp =")
    sources = unique(body, "        auto cellOp =")
    body = body[:faces] + "        if (phase != Phase::Source)\n        {\n" + body[faces:sources] + "        }\n" + body[sources:]
    body = body.replace("if (!settings.ignoreSourceTerm)", "if (phase != Phase::Faces && !settings.ignoreSourceTerm)")
    # The inherited public accessor has the exact same const reference type.
    body = body.replace("phys_.", "this->phys().")
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    header = r'''// Generated from EulerEvaluator_EvaluateRHS.hxx; do not edit.
#pragma once
#include "Euler/EulerEvaluator.hpp"
#include "DNDS/Core/ScopedValueAlternator.hpp"
namespace DNDS::Euler
{
class DiagnosticEulerSA : public EulerEvaluator<NS_SA>
{
    using Base = EulerEvaluator<NS_SA>;
    static constexpr auto model = NS_SA;
    static constexpr int nVars = 6;
public:
    using Base::Base;
    enum class Phase { Faces, Source, Composed };
    static constexpr const char *SourceSHA256 = "DIGEST";
    void Apply(Phase phase, TDof &rhs, JacobianDiagBlock<6> &JSource,
               TDof &u, TRec &uRecUnlim, TRec &uRec, ArrayDOFV<1> &uRecBeta,
               ArrayDOFV<1> &cellRHSAlpha, uint64_t flags)
    {
        const bool onlyOnHalfAlpha = false;
        const real t = 0;
        OptionalRef<ArrayDOFV<1>> cellTWarm;
        OptionalRef<const ArrayDOFV<1>> reactiveSplitChi;
        CFV::ReconstructionMaskView reconstructionMask;
        DNDS_check_throw_info((flags & RHS_Dont_Update_Integration) && (flags & RHS_Dont_Record_Bud_Flux),
                              "extracted diagnostic excludes integration recording");
#define IF_NOT_NOREC (1)
BODY
#undef IF_NOT_NOREC
    }
};
}
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(header.replace("DIGEST", digest).replace("BODY", body) + "\n")


if __name__ == "__main__":
    main()
