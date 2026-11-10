"""Add external controller scopes to a private unified nonlinear source copy."""
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
    text = replace_once(
        text, anchor,
        "        const auto &diagnosticNumerics = data.discrete.Settings().numerics;\n"
        "        ::DiagnosticTiming::Level diagnosticLevel(diagnosticNumerics.firstOrderConvection ? 2 : diagnosticNumerics.reconstruction == \"variational\" ? 0 : 1);\n" + anchor)
    for phase, block in (("Jacobian", "Matrix"), ("LinearSolve", "Linear")):
        anchor = f"                    const NonlinearDetail::PhaseTimer timer(profile, NonlinearProfile::{phase});"
        text = replace_once(
            text, anchor,
            f"                    ::DiagnosticTiming::Scope diagnosticScope(::DiagnosticTiming::Block::{block});\n" + anchor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.write_text(
        "// Generated unified nonlinear diagnostic; original SHA256: " + digest +
        '\n#include "DiagnosticTiming.hpp"\n' + text)


if __name__ == "__main__":
    main()
