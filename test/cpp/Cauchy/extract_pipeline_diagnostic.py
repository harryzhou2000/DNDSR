"""Add external diagnostic scopes to a private copy of the real FV pipeline.

No call is removed or moved. The executable supplies the scope implementation;
production objects and their built-in timer counters are not modified.
"""
import argparse
import hashlib
import re
from pathlib import Path


def replace_once(text, old, new):
    # Hook formatting may change indentation/line wrapping, never tokens.
    pattern = r"\s+".join(re.escape(part) for part in old.strip().split())
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"pipeline scope anchor changed: {old!r}")
    match = matches[0]
    return text[:match.start()] + new + text[match.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()
    text = replace_once(text,
        "        const EquationWorkspace::PhaseTime timing(work, EquationWorkspace::TotalPhase);",
        "        ::DiagnosticTiming::Level diagnosticLevel(_data->settings.numerics.firstOrderConvection ? 2 : _data->settings.numerics.reconstruction == \"variational\" ? 0 : 1);\n"
        "        ::DiagnosticTiming::Scope diagnosticFV(::DiagnosticTiming::Block::Spatial);\n"
        "        const EquationWorkspace::PhaseTime timing(work, EquationWorkspace::TotalPhase);")
    text = replace_once(text,
        "local([&] { auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers)); });",
        "local([&] { ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryPrepare); auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers)); });")
    text = replace_once(text,
        "        for (auto &halo : work.meanHalo)\n            meanOperations.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
        "        { ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryExchange);\n"
        "        for (auto &halo : work.meanHalo)\n            meanOperations.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));\n        }")
    text = replace_once(text,
        "submit([&] { auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });",
        "submit([&] { ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute); auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });")
    text = replace_once(text,
        "        wait(meanOperations);",
        "        { ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryExchange); wait(meanOperations); }")
    text = replace_once(text,
        "                auxiliaryPreparation->Apply(overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);",
        "                ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute);\n"
        "                auxiliaryPreparation->Apply(overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);")
    text = replace_once(text,
        "            PrepareAuxValues(state, *auxiliary, *work.meanGradients, engine, stream, time, work.workers);",
        "            ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute);\n"
        "            PrepareAuxValues(state, *auxiliary, *work.meanGradients, engine, stream, time, work.workers);")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.write_text('// Generated FV diagnostic; original SHA256: ' + digest + '\n' + text)


if __name__ == "__main__":
    main()
