"""Expose Euler's real driver closures in a diagnostic-only translation unit.

The initialization and all callback definitions remain verbatim. Only the
time/output loop is replaced. No production source or public API is changed.
"""
import argparse
import hashlib
import re
from pathlib import Path


def unique(text, anchor):
    if text.count(anchor) != 1:
        raise ValueError(f"expected one driver anchor: {anchor!r}")
    return text.index(anchor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    # Time original matrix calls without changing their branch/argument order.
    source, matrix_calls = re.subn(r"eval\.LUSGSMatrixInit\([^;]+;",
        lambda match: "::DiagnosticTiming::MeasureMatrix([&] { " + match.group(0) + " });", source)
    if matrix_calls != 5:
        raise ValueError("Euler matrix-preparation callsites changed")
    anchor = "    {\n        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS\n        auto &eval = *pEval;\n        auto initUDOF ="
    unique(source, anchor)
    source = source.replace(anchor,
        "    {\n        ::DiagnosticTiming::Level diagnosticLevel(gridLevel);\n"
        "        ::DiagnosticTiming::Scope diagnosticScope(::DiagnosticTiming::Block::Linear);\n"
        "        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS\n        auto &eval = *pEval;\n        auto initUDOF =")
    # Scope the original callbacks themselves so recursive coarse calls are
    # observed too. The time/correction algorithms remain verbatim.
    for old, new in [
        ("                int iter, real ct, int uPos, int reconstructionFlag)\n        {",
         "                int iter, real ct, int uPos, int reconstructionFlag)\n        {\n            ::DiagnosticTiming::Scope diagnosticScope(::DiagnosticTiming::Block::Spatial);"),
        ("auto fdtau = [&](ArrayDOFV<nVarsFixed> &cx, ArrayDOFV<1> &dTau, real alphaDiag, int uPos)\n        {",
         "auto fdtau = [&](ArrayDOFV<nVarsFixed> &cx, ArrayDOFV<1> &dTau, real alphaDiag, int uPos)\n        {\n            ::DiagnosticTiming::Scope diagnosticScope(::DiagnosticTiming::Block::Auxiliary);"),
        ("                    auto call_evaluate_rhs = [&]()\n                    {",
         "                    auto call_evaluate_rhs = [&]()\n                    {\n                        ::DiagnosticTiming::Scope diagnosticScope(::DiagnosticTiming::Block::Spatial);"),
        ("                    DNDS_assert(mgLevel > 0 && mgLevel <= mgLevelMax);",
         "                    ::DiagnosticTiming::Level diagnosticLevel(mgLevel);\n                    DNDS_assert(mgLevel > 0 && mgLevel <= mgLevelMax);")]:
        unique(source, old)
        source = source.replace(old, new)
    start = unique(source, "        auto fstop =")
    end = source.index("\n    DNDS_SWITCH_INTELLISENSE(", start)
    hook = r'''
        // Test-only replacement of the time/output loop. These actions call
        // the original closures above, including their communication and PP.
        static_assert(model == NS_SA, "driver diagnostic supports SA only");
        TDof diagnosticIncrement, diagnosticInitial;
        for (auto *field : {&diagnosticIncrement, &diagnosticInitial})
            initUDOF(*field);
        auto initDiagnosticScalar = [&](ArrayDOFV<1> &field) { vfv->BuildUDof(field, 1); };
        // Use the real BDF1 driver for its residual/history arithmetic. A
        // no-op stopping callback excludes only the outer convergence check,
        // as in the matching production CorrectOnce smoother contract.
        ODE::ImplicitBDFDualTimeStep<TDof, ArrayDOFV<1>> diagnosticODE(
            mesh->NumCell(), initUDOF, initDiagnosticScalar, 1);
        auto &diagnosticRHS = diagnosticODE.rhsbuf[0];
        auto &diagnosticDTau = diagnosticODE.dTau;
        diagnosticInitial = u;
        step = 1;
        curDtImplicit = 1e200;
        CFLNow = config.implicitCFLControl.CFL;
        int diagnosticUpdates = 0;
        DriverDiagnostic::Actions actions;
        actions.reset = [&]
        {
            u = diagnosticInitial;
            uRec.setConstant(0.);
            uRecLimited.setConstant(0.);
            betaPP.setConstant(1.);
            alphaPP.setConstant(1.);
            alphaPP_tmp.setConstant(1.);
            diagnosticUpdates = 0;
        };
        actions.spatial = [&] { frhs(diagnosticRHS, u, diagnosticDTau, 1, 0., 0); };
        actions.auxiliary = [&] { fdtau(u, diagnosticDTau, 1., 0); };
        actions.implicit = [&]
        {
            diagnosticODE.Step(u, diagnosticIncrement,
                [](TDof &, TDof &, ArrayDOFV<1> &, int, real, int) {},
                [](TDof &, ArrayDOFV<1> &, real, int) {},
                fsolve, 1, [](int, TDof &, int) { return false; }, fincrement, curDtImplicit);
        };
        actions.composed = [&]
        {
            auto timedRHS = [&](TDof &rhs, TDof &state, ArrayDOFV<1> &tau, int iteration, real ct, int position)
            {
                frhs(rhs, state, tau, iteration, ct, position);
            };
            auto timedAux = [&](TDof &state, ArrayDOFV<1> &tau, real alpha, int position)
            {
                fdtau(state, tau, alpha, position);
            };
            auto countIncrement = [&](TDof &state, TDof &increment, real alpha, int position)
            {
                fincrement(state, increment, alpha, position);
                ++diagnosticUpdates;
            };
            diagnosticODE.Step(u, diagnosticIncrement, timedRHS, timedAux,
                fsolve, DriverDiagnostic::corrections,
                [&](int iteration, TDof &residual, int stage)
                {
                    return DriverDiagnostic::levels ? functor_fstop(iteration, residual, stage, runningEnvironment) : false;
                }, countIncrement, curDtImplicit);
            DNDS_check_throw_info(diagnosticUpdates == DriverDiagnostic::corrections,
                                  "Euler diagnostic did not perform its requested update count");
        };
        actions.residual = &diagnosticRHS;
        actions.pseudoStep = &diagnosticDTau;
        DriverDiagnostic::Run(actions);
    }
'''
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    output = ('// Generated diagnostic driver; original SHA256: ' + digest + '\n' +
              source[:start] + hook + source[end:] +
              '\nDNDS_EULERSOLVER_INS_EXTERN(NS_SA, );\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)


if __name__ == "__main__":
    main()
