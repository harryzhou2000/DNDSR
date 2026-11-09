"""Add external matrix/linear scopes to a private production nonlinear TU."""
import argparse
import hashlib
import re
from pathlib import Path


def replace_once(text, old, new):
    pattern = r"\s+".join(re.escape(part) for part in old.strip().split())
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"nonlinear scope anchor changed: {old!r}")
    match = matches[0]
    return text[:match.start()] + new + text[match.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()
    anchor = "        const index nv = data.binding.snapshot.ComponentNumber();"
    text = replace_once(text, anchor,
        "        const auto &diagnosticNumerics = data.discrete.Settings().numerics;\n"
        "        ::DiagnosticTiming::Level diagnosticLevel(diagnosticNumerics.firstOrderConvection ? 2 : diagnosticNumerics.reconstruction == \"variational\" ? 0 : 1);\n" + anchor)
    for phase in ("Jacobian", "LinearSolve"):
        anchor = f"                    const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::{phase});"
        block = "Matrix" if phase == "Jacobian" else "Linear"
        text = replace_once(text, anchor,
            f"                    ::DiagnosticTiming::Scope diagnosticScope(::DiagnosticTiming::Block::{block});\n" + anchor)
    for anchor, block in (
        ('void ValidateMeans(const EquationDofs &state) const\n        {', 'MeanValidation'),
        ('const auto measure = [&](auto &norms)\n        {', 'DefectAssembly'),
        ('const NonlinearDetail::PhaseTimer timer(control.profile, NonlinearProfile::PseudoTimePP);', 'PseudoTimePP'),
        ('std::pair<index, real> EquationNonlinearSolver::ApplyCorrection(EquationDofs &state, const EquationDofs &increment, real scale,\n                                                                  const std::vector<index> &updateComponents)\n    {', 'CorrectionLocal')):
        text = replace_once(text, anchor, anchor +
            f'\n        ::DiagnosticTiming::Scope detailScope(::DiagnosticTiming::Block::{block});')
    # End local correction at the existing publication boundary. Collective
    # acceptance and PP statistics remain in their exact original order.
    text = replace_once(text, '// Candidate checks precede all state writers.',
        'detailScope.Finish();\n        // Candidate checks precede all state writers.')
    if 'const auto validateCandidate = [&]' in text:
        # Separate local validation from the existing global minimum. These
        # clocks only observe the production sequence: no barrier, reordered
        # collective, or production timer participates in the measurement.
        text = replace_once(text, 'const auto validateCandidate = [&]\n        {',
            'const auto validateCandidate = [&]\n        {\n'
            '            ::DiagnosticTiming::Scope validationScope(::DiagnosticTiming::Block::CandidateValidation);')
        text = replace_once(text, 'MPI_Allreduce(MPI_IN_PLACE, agreement, 2, MPI_DOUBLE, MPI_MIN, data.mpi.comm);',
            '{ ::DiagnosticTiming::Scope agreementScope(::DiagnosticTiming::Block::CandidateAgreement);\n'
            '            MPI_Allreduce(MPI_IN_PLACE, agreement, 2, MPI_DOUBLE, MPI_MIN, data.mpi.comm); }')
        text = replace_once(text, 'const auto validateCandidate = [&]',
            '{ ::DiagnosticTiming::Scope acceptanceScope(::DiagnosticTiming::Block::CandidateAcceptance);\n'
            'const auto validateCandidate = [&]')
        text = replace_once(text, 'else\n            MPI::LocalStage(data.mpi, validateCandidate);',
            'else\n            MPI::LocalStage(data.mpi, validateCandidate); }')
    else:
        text = replace_once(text, 'MPI::LocalStage(data.mpi, [&]\n                        {\n            DNDS_check_throw_info(frozen,',
            '{ ::DiagnosticTiming::Scope detailScope(::DiagnosticTiming::Block::CandidateAcceptance);\n'
            'MPI::LocalStage(data.mpi, [&]\n                        {\n            DNDS_check_throw_info(frozen,')
        text = replace_once(text, 'data.ValidateMeans(data.candidate); });',
            'data.ValidateMeans(data.candidate); }); }')
    text = replace_once(text, 'if (data.positivity)\n        {\n            MPI_Allreduce(MPI_IN_PLACE, &limited,',
        'if (data.positivity)\n        {\n            ::DiagnosticTiming::Scope detailScope(::DiagnosticTiming::Block::CorrectionStatistics);\n'
        '            MPI_Allreduce(MPI_IN_PLACE, &limited,')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.write_text("// Generated nonlinear diagnostic; original SHA256: " + digest +
                           '\n#include "DiagnosticTiming.hpp"\n' + text)


if __name__ == "__main__":
    main()
