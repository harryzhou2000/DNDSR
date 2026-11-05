"""CMake registration/build coverage, without a compiler or DNDS extensions."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TestAggregateCoverage(unittest.TestCase):
    def configure(self, dependencies):
        cmake = os.environ.get("DNDS_HARNESS_CMAKE") or shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake is not installed")
        with tempfile.TemporaryDirectory(prefix="dnds-test-targets-") as directory:
            source = Path(directory)
            (source / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.21)
project(aggregate_contract NONE)
include("{ROOT / 'cmake/DndsTestTargets.cmake'}")
add_custom_target(all_unit_tests)
add_custom_target(module_tests)
add_custom_target(registered_test)
set_property(GLOBAL APPEND PROPERTY DNDS_CPP_TEST_TARGETS registered_test)
add_dependencies(all_unit_tests module_tests frontend)
{dependencies}
dnds_check_registered_test_targets(all_unit_tests)
# A parent project may declare its frontend after its test subdirectory.
add_custom_target(frontend)
""")
            return subprocess.run(
                [cmake, "-S", str(source), "-B", str(source / "build")],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=30,
            )

    def test_transitive_aggregate_and_later_frontend(self):
        result = self.configure("add_dependencies(module_tests registered_test)")
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_registered_orphan_is_rejected(self):
        result = self.configure("")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("registered_test is missing from all_unit_tests", result.stdout)


if __name__ == "__main__":
    unittest.main()
