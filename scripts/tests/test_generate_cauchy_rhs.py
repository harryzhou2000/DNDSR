"""Exercise the RHS generator's CMake contract without a CUDA compilation."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("rhs_codegen", ROOT / "scripts/generate_cauchy_rhs.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class RHSCodeGenerationTests(unittest.TestCase):
    def test_catalog_and_unchanged_output(self):
        manifest = ROOT / "scripts/manifests/cauchy_rhs_shapes.json"
        shapes = GENERATOR.read_shapes(manifest)
        self.assertEqual(len(shapes), 24)
        acm = GENERATOR.read_shapes(manifest, mechanical=True)
        self.assertEqual(len(acm), 24)
        self.assertEqual({c for c, _, _ in acm}, {4, 5, 6})
        self.assertEqual(sum(m == 0 for _, m, _ in acm), 6)
        for components in (5, 6, 7):
            self.assertEqual([(m, d) for c, m, d in shapes if c == components],
                             [(0, 2), (2, 2), (5, 2), (9, 2), (0, 3), (3, 3), (9, 3), (19, 3)])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "generated"
            command = [sys.executable, str(ROOT / "scripts/generate_cauchy_rhs.py"),
                       "--manifest", str(manifest), "--output-dir", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
            self.assertEqual(len(before), 71)
            for c, m, d in acm:
                for mapping in (("Thread", "CTASegments") if m else ("Thread",)):
                    name = f"FiniteVolumeACMFluxKernels_C{c}_M{m}_D{d}_{mapping}.cu"
                    self.assertIn(f"LaunchACMRHSShape<{c}, {m}, {d}, RHSFaceMapping::{mapping}>", before[name][0].decode())
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()})

    def test_cpu_catalog_distinguishes_physics_form_and_width(self):
        manifest = ROOT / "scripts/manifests/cauchy_rhs_shapes.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            subprocess.run([sys.executable, str(ROOT / "scripts/generate_cauchy_cpu_flux.py"),
                            "--manifest", str(manifest), "--output-dir", str(output)], check=True)
            generated = list(output.glob("FiniteVolumeCPUFluxCatalog_Flux_*.cpp"))
            self.assertEqual(len(generated), 60)
            for profile in ("ACMSA", "PerfectGas"):
                for mode, form in ((0, "MeanOnly"), (0, "DirectGradient"), (19, "Modal")):
                    name = f"FiniteVolumeCPUFluxCatalog_Flux_{profile}_{form}_C5_M{mode}_D3.cpp"
                    text = (output / name).read_text()
                    self.assertIn(f"EquationPhysicsProfile::{profile}, ReconstructionForm::{form}, 5, 3, {mode}", text)
                    self.assertIn("a.remote.CompactRows<5>()", text)

    def test_filtered_registries_only_reference_selected_leaves(self):
        manifest = ROOT / "scripts/manifests/cauchy_rhs_shapes.json"
        with tempfile.TemporaryDirectory() as temporary:
            for generator in ("cauchy_cpu_flux", "cauchy_rhs"):
                output = Path(temporary) / generator
                base = [sys.executable, str(ROOT / f"scripts/generate_{generator}.py"),
                        "--manifest", str(manifest), "--output-dir", str(output)]
                for family, dimension in (("ACM", 3), ("PerfectGas", 2)):
                    subprocess.run(base + ["--families", family, "--dimensions", str(dimension)], check=True)
                    cpu = generator == "cauchy_cpu_flux"
                    leaves = sorted(output.glob("*_C*_M*_D*"))
                    self.assertEqual(len(leaves), 15 if cpu else 21 if family == "ACM" else 12)
                    for leaf in leaves:
                        self.assertIn(f"_D{dimension}", leaf.name)
                        self.assertEqual("ACM" in leaf.name, family == "ACM")
                    if cpu:
                        registry = (output / "FiniteVolumeCPUFluxCatalog.cpp").read_text()
                        for leaf in leaves:
                            self.assertIn(leaf.stem.removeprefix("FiniteVolumeCPUFluxCatalog_"), registry)
                    else:
                        dispatchers = list(output.glob("*_Instantiations.cpp"))
                        self.assertEqual(len(dispatchers), 1)
                        self.assertNotIn(f"case {5 - dimension}:\n                switch (modes)",
                                         dispatchers[0].read_text())

    @unittest.skipUnless(shutil.which("cmake"), "CMake is unavailable")
    def test_cmake_feature_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            build = source / "build"
            (source / "CMakeLists.txt").write_text(
                'cmake_minimum_required(VERSION 3.21)\nproject(Features NONE)\n'
                f'include("{ROOT.as_posix()}/cmake/DndsCauchyRHS.cmake")\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/selected" "${_cauchy_families}:${_cauchy_dimensions}")\n')
            configure = ["cmake", "-S", str(source), "-B", str(build)]
            subprocess.run(configure, check=True, capture_output=True)
            self.assertEqual((build / "selected").read_text(), "PerfectGas;ACM:2;3")
            subprocess.run(configure + ["-DDNDS_CAUCHY_ENABLE_COMPRESSIBLE=OFF", "-DDNDS_CAUCHY_ENBALE_3D=OFF"],
                           check=True, capture_output=True)
            self.assertEqual((build / "selected").read_text(), "ACM:2")
            for disabled in ("ACM", "2D"):
                result = subprocess.run(configure + [f"-DDNDS_CAUCHY_ENABLE_{disabled}=OFF"],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("at least one equation family and one mesh dimension", result.stderr)
                subprocess.run(configure + [f"-DDNDS_CAUCHY_ENABLE_{disabled}=ON"], check=True, capture_output=True)

    @unittest.skipUnless(shutil.which("cmake"), "CMake is unavailable")
    def test_cmake_tracks_inputs_and_recovers_every_output(self):
        generators = ["Unix Makefiles"] if shutil.which("make") else []
        if shutil.which("ninja"):
            generators.append("Ninja")
        if not generators:
            self.skipTest("Make and Ninja are unavailable")
        for backend in generators:
            with self.subTest(generator=backend), tempfile.TemporaryDirectory() as temporary:
                source, build = Path(temporary) / "source", Path(temporary) / "build"
                for name in ("scripts/generate_cauchy_rhs.py", "scripts/manifests/cauchy_rhs_shapes.json", "cmake/DndsCauchyRHS.cmake"):
                    destination = source / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, destination)
                (source / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.21)
project(RHSCodeGeneration NONE)
include("${CMAKE_SOURCE_DIR}/cmake/DndsCauchyRHS.cmake")
dnds_generate_cauchy_rhs(rhs_sources rhs_include rhs_codegen cauchy_rhs)
add_custom_target(static_consumer ALL)
add_custom_target(shared_consumer ALL)
add_dependencies(static_consumer ${rhs_codegen})
add_dependencies(shared_consumer ${rhs_codegen})
file(APPEND "${CMAKE_BINARY_DIR}/configure_count" "configured\\n")
''')

                def run(*arguments):
                    result = subprocess.run(["cmake", *map(str, arguments)], stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout)

                run("-S", source, "-B", build, "-G", backend)
                run("--build", build, "--parallel", "4")
                output = build / "generated/cauchy_rhs"
                stable = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
                count = (build / "configure_count").read_text().count("configured")
                run("--build", build, "--parallel", "4")
                self.assertEqual(count, (build / "configure_count").read_text().count("configured"))
                self.assertEqual(stable, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})
                # Cover the first and secondary custom-command output, plus
                # the include and CMake index needed before source compilation.
                for name in ("FiniteVolumeRHSKernels_Instantiations.cpp", "FiniteVolumeRHSKernels_C7_M19_D3.cu",
                             "FiniteVolumeACMFluxKernels_C4_M19_D3_Thread.cu",
                             "FiniteVolumeACMFluxKernels_C4_M19_D3_CTASegments.cu",
                             "FiniteVolumeRHSKernels_Instantiations.hpp", "FiniteVolumeRHSKernels_Sources.cmake"):
                    (output / name).unlink()
                    run("--build", build, "--parallel", "4")
                    self.assertTrue((output / name).is_file())
                manifest = source / "scripts/manifests/cauchy_rhs_shapes.json"
                data = json.loads(manifest.read_text())
                data["component_counts"].append(8)
                manifest.write_text(json.dumps(data))
                run("--build", build, "--parallel", "4")
                self.assertTrue((output / "FiniteVolumeRHSKernels_C8_M19_D3.cu").is_file())
                self.assertIn("case 8:", (output / "FiniteVolumeRHSKernels_Instantiations.cpp").read_text())
                retired = output / "FiniteVolumeRHSKernels_C8_M19_D3.cu"
                edited = output / "FiniteVolumeRHSKernels_C8_M2_D2.cu"
                edited.write_text("// local investigation\n")
                data["component_counts"].remove(8)
                manifest.write_text(json.dumps(data))
                run("--build", build, "--parallel", "4")
                self.assertFalse(retired.exists())
                self.assertEqual(edited.read_text(), "// local investigation\n")
                # A generator-only change must also trigger reconfiguration,
                # even when it emits byte-identical numerical wrappers.
                count = (build / "configure_count").read_text().count("configured")
                with (source / "scripts/generate_cauchy_rhs.py").open("a") as stream:
                    stream.write("\n# Dependency tracking probe.\n")
                run("--build", build, "--parallel", "4")
                self.assertGreater((build / "configure_count").read_text().count("configured"), count)


if __name__ == "__main__":
    unittest.main()
