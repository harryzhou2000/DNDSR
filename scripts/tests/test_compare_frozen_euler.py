"""Critical oracle-verifier checks without DNDSR extensions or solver execution."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "compare_frozen_euler", Path(__file__).resolve().parents[1] / "benchmark/compare_frozen_euler.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def bits(value):
    return struct.unpack("=Q", struct.pack("=d", value))[0]


class FrozenProbeComparison(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.old, self.new = self.root / "old", self.root / "new"
        # The RHS whose name contains 'jacobian' must remain a vector; only the
        # explicit Jacobian fields have square width. Use a shuffled ID order.
        fields = {"u_bits": [bits(1.), bits(2.)], "rec_bits": [bits(0.)] * 4,
                  "rhs_no_reactive_jacobian_bits": [bits(3.), bits(4.)],
                  "source_jacobian_bits": [bits(5.)] * 4, "volume_bits": bits(.5)}
        self.record = {"model": 1, "np": 1, "rank": 0, "polynomial": 1,
                       "energy_convention": "test", "scales": [1.7, 230, 2.3, 300],
                       "input_bits_preserved": True,
                       "cells": [{"id": cell, **fields} for cell in (9, 3)]}
        self.write(self.old, self.record)
        self.record["cells"].reverse()
        self.write(self.new, self.record)

    def write(self, prefix, record):
        prefix.with_suffix(".rank0.json").write_text(json.dumps(record))

    def test_exact_id_join_and_visible_numeric_mismatch(self):
        self.assertTrue(probe.compare(self.old, self.new)["passed"])
        self.record["cells"][0]["rhs_no_reactive_jacobian_bits"] = [bits(3.1), bits(4.)]
        self.write(self.new, self.record)
        result = probe.compare(self.old, self.new)
        self.assertFalse(result["passed"])
        self.assertEqual(result["fields"]["rhs_no_reactive_jacobian_bits"][0]["bit_differences"], 1)

    def test_missing_ids_and_nonfinite_payload_are_rejected(self):
        self.record["cells"].pop()
        self.write(self.new, self.record)
        with self.assertRaisesRegex(ValueError, "inventory"):
            probe.compare(self.old, self.new)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe.decode(bits(float("nan")))


if __name__ == "__main__":
    unittest.main()
