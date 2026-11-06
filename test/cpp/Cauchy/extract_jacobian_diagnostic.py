"""Observe four disjoint assembly batches in a private production source copy.

Only external scope declarations/Finish calls are inserted. The round-trip
check below rejects any accidental change to the production text. Unique
anchors fail explicitly when the implementation structure changes.
"""
import argparse
import hashlib
from pathlib import Path


def extract(source):
    text = source
    insertions = []

    def before(anchor, addition):
        nonlocal text
        if text.count(anchor) != 1:
            raise ValueError(f"expected one Jacobian batch anchor: {anchor!r}")
        text = text.replace(anchor, addition + anchor, 1)
        insertions.append(addition)

    before("        ValidateAuxValues(state, auxiliary, time);",
           "        ::DiagnosticTiming::Scope assemblySetup(::DiagnosticTiming::Block::JacobianSetup);\n")
    before("            // Only O(N) scalar thermodynamic linearization data are frozen per\n",
           "            assemblySetup.Finish();\n"
           "            ::DiagnosticTiming::Scope assemblyFreeze(::DiagnosticTiming::Block::JacobianFreeze);\n")
    before("            if ((chemical || rans) && sourceReconstruction)\n",
           "            assemblyFreeze.Finish();\n"
           "            ::DiagnosticTiming::Scope assemblySource(::DiagnosticTiming::Block::JacobianSource);\n")
    before("            for (index iCell = 0; iCell < owned; ++iCell)\n            {\n                if (output.DiagonalMode() == DiagonalStorage::Block)",
           "            assemblySource.Finish();\n"
           "            ::DiagnosticTiming::Scope assemblyFaces(::DiagnosticTiming::Block::JacobianFaces);\n")
    restored = text
    for addition in reversed(insertions):
        if restored.count(addition) != 1:
            raise ValueError("ambiguous diagnostic insertion")
        restored = restored.replace(addition, "", 1)
    if restored != source:
        raise ValueError("Jacobian extraction changed production text")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("// Generated assembly diagnostic; original SHA256: " +
                           hashlib.sha256(source).hexdigest() +
                           '\n#include "DiagnosticTiming.hpp"\n' + extract(source.decode()))
