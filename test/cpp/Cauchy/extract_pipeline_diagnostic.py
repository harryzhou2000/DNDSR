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
    # Chronology is disabled unless the diagnostic explicitly opts in. Insert
    # statements only; stripping them must reproduce the input byte for byte.
    trace_insertions = []

    def insert_trace(anchor, addition, after=False):
        nonlocal text
        pattern = r"\s+".join(re.escape(part) for part in anchor.strip().split())
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ValueError(f"auxiliary trace anchor changed: {anchor!r}")
        match = matches[0]
        position = text.find("\n", match.end()) + 1 if after else text.rfind("\n", 0, match.start()) + 1
        text = text[:position] + addition + text[position:]
        trace_insertions.append(addition)

    insert_trace("void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, Workspace &work, AuxiliaryValues &auxiliary,\n"
                 "R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const\n    {",
                 "        ::DiagnosticTiming::AuxiliaryTrace correctionTrace(!work.meanGradients, 1,\n"
                 "            _data->settings.numerics.firstOrderConvection ? 2 : _data->settings.numerics.reconstruction == \"variational\" ? 0 : 1);\n", True)
    insert_trace("        for (auto &halo : work.meanHalo)\n"
                 "            transfers.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
                 "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchBegin);\n")
    insert_trace("            transfers.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
                 "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchEnd);\n"
                 "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::BoundaryEnd);\n", True)
    insert_trace("        for (auto &transfer : transfers)\n            transfer.Wait();",
                 "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitBegin);\n")
    insert_trace("            transfer.Wait();",
                 "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitEnd);\n", True)
    insert_trace("                preparation->Apply(_data->settings.execution.overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);",
                 "                correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeBegin);\n")
    insert_trace("                preparation->Apply(_data->settings.execution.overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);",
                 "                correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeEnd);\n", True)
    insert_trace("        const Workspace::PhaseTime timing(work, Workspace::TotalPhase);",
                 "        ::DiagnosticTiming::AuxiliaryTrace evaluationTrace(auxiliary && !work.meanGradients, 0,\n"
                 "            _data->settings.numerics.firstOrderConvection ? 2 : _data->settings.numerics.reconstruction == \"variational\" ? 0 : 1);\n")
    insert_trace("        for (auto &halo : work.meanHalo)\n"
                 "            meanOperations.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
                 "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchBegin);\n")
    insert_trace("            meanOperations.push_back(halo.Begin(work.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
                 "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchEnd);\n", True)
    insert_trace("        wait(meanOperations);",
                 "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::BoundaryEnd);\n"
                 "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitBegin);\n")
    insert_trace("        wait(meanOperations);",
                 "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitEnd);\n", True)
    insert_trace("                auxiliaryPreparation->Apply(overlap ? CFV::ReconstructionRegion::HaloDependent : CFV::ReconstructionRegion::All);",
                 "                evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeBegin);\n")
    insert_trace("                auxiliaryPreparation->Finish(*auxiliary); });",
                 "                evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeEnd);\n")
    # Both end marks measure Apply completion. Finish publishes leases;
    # existing parent clocks still enclose publication.
    restored = text
    for addition in reversed(trace_insertions):
        if restored.count(addition) != 1:
            raise ValueError("ambiguous auxiliary trace insertion")
        restored = restored.replace(addition, "", 1)
    if restored != args.source.read_text():
        raise ValueError("auxiliary tracing changed production bytes")
    # Count the two correction-only source refresh entrances, once per call.
    # The production guard and all numerical calls remain byte-for-byte intact.
    # This counter is not a measurement of source-kernel execution time.
    refresh = r'if\s*\(work\.SourceSamplesCurrent\(state,\s*time\)\)\s*return;'
    text, count = re.subn(refresh, lambda match: match.group(0) +
        '\n                ::DiagnosticTiming::ObserveCorrectionSource(work.sourceOffsets.empty() ? 0 : work.sourceOffsets.back());', text)
    if count != 2:
        raise ValueError('expected two correction source refresh guards')
    # Phase-local scopes retain all original calls. Source sampling nests
    # inside volume-source evaluation and is reported separately as such.
    for phase, block in (('Gradient', 'Gradient'), ('VR', 'Reconstruction'),
                         ('PP', 'PolynomialPP'), ('RHS', 'FaceRHS'),
                         ('Sampling', 'SourceSampling'), ('Source', 'VolumeSource')):
        pattern = (r'const\s+(?:Workspace::)?PhaseTime\s+timing\('
                   r'(\*this|work),\s*(?:Workspace::)?' + phase + r'Phase\);')
        text, count = re.subn(pattern,
            lambda match: f'::DiagnosticTiming::Scope detailScope(::DiagnosticTiming::Block::{block});\n' + match.group(0), text)
        if count != 1:
            raise ValueError(f'FV detail scope {phase} expected once, got {count}')
    text = replace_once(text,
        "        const Workspace::PhaseTime timing(work, Workspace::TotalPhase);",
        "        ::DiagnosticTiming::Level diagnosticLevel(_data->settings.numerics.firstOrderConvection ? 2 : _data->settings.numerics.reconstruction == \"variational\" ? 0 : 1);\n"
        "        ::DiagnosticTiming::Scope diagnosticFV(::DiagnosticTiming::Block::Spatial);\n"
        "        const Workspace::PhaseTime timing(work, Workspace::TotalPhase);")
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
