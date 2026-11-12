#!/usr/bin/env python3
"""Build the pinned, non-MPI AmgX dependency outside the DNDSR source tree.

This recipe does not alter cfd_externals, install system packages, select a GPU
or launch solver jobs. CUDA architectures and the host compiler are explicit.
An existing source checkout must already match the pin and be clean.
"""

import argparse
import json
from pathlib import Path
import subprocess


AMGX_REVISION = "cc1cebdbb32b14d33762d4ddabcb2e23c1669f47"
AMGX_TAG = "v2.5.0"


def run(arguments, **kwargs):
    subprocess.run([str(value) for value in arguments], check=True, **kwargs)


def git_output(source, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(source), *arguments], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Dedicated dependency directory containing source/build/install")
    parser.add_argument("--architectures", required=True,
                        help="CMake CUDA architecture list, for example 80;86 or 86")
    parser.add_argument("--cuda-compiler", type=Path)
    parser.add_argument("--host-compiler", type=Path,
                        help="CUDA-compatible C++ compiler for both C++ and nvcc host compilation")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1 or not args.architectures:
        parser.error("jobs and architecture list must be nonempty/positive")

    root = args.root.expanduser().resolve()
    source, build, install = (root / name for name in ("source", "build", "install"))
    root.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        run(["git", "clone", "--depth=1", "--branch", AMGX_TAG,
             "https://github.com/NVIDIA/AMGX.git", source])
    if git_output(source, "status", "--porcelain"):
        raise RuntimeError("AmgX source has local changes; preserve them and select a clean dependency root")
    if git_output(source, "rev-parse", "HEAD") != AMGX_REVISION:
        raise RuntimeError("Existing AmgX source differs from the pinned revision; it was not modified")
    run(["git", "-C", source, "submodule", "update", "--init", "--recursive", "--depth=1"])

    configure = ["cmake", "-S", source, "-B", build,
                 "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_NO_MPI=ON",
                 # Upstream's prefix-only RUNPATH masks the application's
                 # transitive CUDA RPATH and can load a different toolkit from
                 # ldconfig. DNDSR already supplies its resolved provider/CUDA
                 # library directories with DT_RPATH; let that policy apply.
                 "-DAMGX_NO_RPATH=ON",
                 f"-DCMAKE_CUDA_ARCHITECTURES={args.architectures}",
                 f"-DCMAKE_INSTALL_PREFIX={install}"]
    if args.cuda_compiler:
        configure.append(f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler.expanduser().resolve()}")
    if args.host_compiler:
        compiler = args.host_compiler.expanduser().resolve()
        configure.extend([f"-DCMAKE_CXX_COMPILER={compiler}",
                          f"-DCMAKE_CUDA_HOST_COMPILER={compiler}"])
    run(configure)
    # Upstream installs its small serial C-API example unconditionally. Build
    # it with the libraries so cmake --install has every declared artifact.
    # Do not build upstream's full test target or execute device examples.
    run(["cmake", "--build", build, "--target", "amgx", "amgxsh", "amgx_capi",
         "--parallel", args.jobs])
    run(["cmake", "--install", build])
    provenance = dict(revision=AMGX_REVISION, tag=AMGX_TAG, mpi=False,
                      architectures=args.architectures,
                      configure=[str(value) for value in configure])
    (install / "dndsr-amgx-build.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Configure DNDSR with -DDNDS_USE_AMGX=ON -DDNDS_AMGX_ROOT={install}")


if __name__ == "__main__":
    main()
