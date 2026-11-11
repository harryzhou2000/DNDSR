"""Add external diagnostic scopes to a private copy of the unified FV pipeline.

The generated translation unit preserves every production statement and adds
only clocks/counters. Unique anchors deliberately fail when orchestration is
restructured so the performance diagnostic cannot silently measure a stale
path.
"""
import argparse
import hashlib
import re
from pathlib import Path


def pattern(text):
    return r"\s+".join(re.escape(part) for part in text.strip().split())


def insert_after_once(text, anchor, addition):
    matches = list(re.finditer(pattern(anchor), text))
    if len(matches) != 1:
        raise ValueError(f"pipeline anchor changed: {anchor!r}")
    end = text.find("\n", matches[0].end()) + 1
    return text[:end] + addition + text[end:]


def insert_before_once(text, anchor, addition):
    matches = list(re.finditer(pattern(anchor), text))
    if len(matches) != 1:
        raise ValueError(f"pipeline anchor changed: {anchor!r}")
    start = text.rfind("\n", 0, matches[0].start()) + 1
    return text[:start] + addition + text[start:]


def replace_once(text, old, new):
    matches = list(re.finditer(pattern(old), text))
    if len(matches) != 1:
        raise ValueError(f"pipeline anchor changed: {old!r}")
    match = matches[0]
    return text[:match.start()] + new + text[match.end():]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text = args.source.read_text()

    correction = (
        "void FiniteVolume::PrepareCorrectionAux(EquationDofs &state, ReconstructionState &reconstruction,\n"
        "Workspace &work, AuxiliaryValues &auxiliary,\n"
        "R::EagerEngine &engine, const R::ExecutionStream &stream, real time) const\n    {")
    text = insert_after_once(
        text, correction,
        "        ::DiagnosticTiming::Scope diagnosticAuxiliary(::DiagnosticTiming::Block::Auxiliary);\n"
        "        ::DiagnosticTiming::AuxiliaryTrace correctionTrace(!work.meanGradients, 1, ::DiagnosticTiming::currentLevel);\n")
    text = insert_before_once(
        text, "        for (auto &halo : reconstruction.meanHalo)\n"
              "            transfers.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchBegin);\n")
    text = insert_after_once(
        text, "            transfers.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchEnd);\n"
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::BoundaryEnd);\n"
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitBegin);\n")
    text = insert_after_once(
        text, "        OperatorSchedule::Wait(transfers);",
        "        correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitEnd);\n")
    text = insert_before_once(
        text, "                preparation->Apply(rest);",
        "                correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeBegin);\n")
    text = insert_after_once(
        text, "                preparation->Finish(auxiliary);",
        "                correctionTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeEnd);\n")

    evaluation = (
        "void FiniteVolume::EvaluateImpl(EquationDofs &state,\n"
        "OptionalRef<EquationDofs> residual,\n"
        "ReconstructionState &reconstruction, Workspace &work, R::EagerEngine &engine,\n"
        "const R::ExecutionStream &stream, real time,\n"
        "OptionalRef<AuxiliaryValues> auxiliary) const\n    {")
    text = insert_after_once(
        text, evaluation,
        "        ::DiagnosticTiming::Level diagnosticLevel(_data->settings.numerics.firstOrderConvection ? 2 : _data->settings.numerics.reconstruction == \"variational\" ? 0 : 1);\n"
        "        ::DiagnosticTiming::Scope diagnosticFV(::DiagnosticTiming::Block::Spatial);\n"
        "        ::DiagnosticTiming::AuxiliaryTrace evaluationTrace(auxiliary && !work.meanGradients, 0, ::DiagnosticTiming::currentLevel);\n")
    text = replace_once(
        text,
        "                      auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers, inputs)); });",
        "                      ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryPrepare);\n"
        "                      auxiliaryPreparation.emplace(BeginAuxValues(state, *auxiliary, time, work.workers, inputs)); });")
    text = insert_before_once(
        text, "        for (auto &halo : reconstruction.meanHalo)\n"
              "            meanOperations.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchBegin);\n"
        "        ::DiagnosticTiming::Scope auxiliaryExchange(::DiagnosticTiming::Block::AuxiliaryExchange);\n")
    text = insert_after_once(
        text, "            meanOperations.push_back(halo.Begin(reconstruction.meanEpoch, engine, stream, _data->settings.CheckHaloFreshness()));",
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::LaunchEnd);\n"
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::BoundaryEnd);\n"
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitBegin);\n")
    text = insert_after_once(
        text, "        OperatorSchedule::Wait(meanOperations);",
        "        auxiliaryExchange.Finish();\n"
        "        evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::WaitEnd);\n")
    text = replace_once(
        text,
        "                            { auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });",
        "                            { ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute);\n"
        "                              auxiliaryPreparation->Apply(CFV::ReconstructionRegion::Core); });")
    text = insert_before_once(
        text, "                auxiliaryPreparation->Apply(rest);",
        "                ::DiagnosticTiming::Scope scope(::DiagnosticTiming::Block::AuxiliaryCompute);\n"
        "                evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeBegin);\n")
    text = insert_after_once(
        text, "                auxiliaryPreparation->Finish(*auxiliary); });",
        "                evaluationTrace.Mark(::DiagnosticTiming::AuxiliaryTrace::ComputeEnd);\n")

    refresh = r"if\s*\(work\.SourceSamplesCurrent\(state,\s*time\)\)\s*return;"
    text, count = re.subn(
        refresh,
        lambda match: match.group(0) +
        "\n                ::DiagnosticTiming::ObserveCorrectionSource(work.sourceOffsets.empty() ? 0 : work.sourceOffsets.back());",
        text)
    if count != 3:
        raise ValueError(f"expected three correction source refresh guards, got {count}")

    # Direct-gradient and variational PP are mutually exclusive production
    # leaves but intentionally share one diagnostic block. Keep both anchors so
    # neither path can disappear from the generated instrumentation silently.
    for phase, block, expected in (("Gradient", "Gradient", 1),
                                   ("PP", "PolynomialPP", 2),
                                   ("RHS", "FaceRHS", 1),
                                   ("Sampling", "SourceSampling", 1),
                                   ("Source", "VolumeSource", 1)):
        expression = (r"const\s+(?:Workspace::)?PhaseTime\s+timing\("
                      r"(\*this|work),\s*(?:Workspace::)?" + phase + r"Phase\);")
        text, count = re.subn(
            expression,
            lambda match: f"::DiagnosticTiming::Scope detailScope(::DiagnosticTiming::Block::{block});\n" + match.group(0),
            text)
        if count != expected:
            raise ValueError(
                f"FV detail scope {phase} expected {expected}, got {count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    args.output.write_text(
        "// Generated unified FV diagnostic; original SHA256: " + digest +
        '\n#include "DiagnosticTiming.hpp"\n' + text)


if __name__ == "__main__":
    main()
