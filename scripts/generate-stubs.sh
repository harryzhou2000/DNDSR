#!/bin/bash
# Generate type stubs for the DNDSR Python package.
#
# This script is called automatically by `cmake --install build --component py`
# after the pybind11 .so files are placed into python/DNDSR/.
#
# Manual usage (if needed):
#   PYTHONPATH=python ./scripts/generate-stubs.sh
#
# Prerequisites:
#   - pip install pybind11-stubgen
#   - pybind11 .so files installed into python/DNDSR/<Module>/_ext/
#   - PYTHONPATH must include the python/ directory (set automatically
#     when called from CMake install, or pass it explicitly)
#
# Output:
#   stubs/DNDSR/           — raw pybind11-stubgen output
#   python/DNDSR/**/*.pyi  — normalized, syntax-checked stubs for PEP 561

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUBS_DIR="${PROJECT_ROOT}/stubs"
PYTHON_DIR="${PROJECT_ROOT}/python"

# Resolve pybind11-stubgen and its matching Python interpreter, supporting
# PATH, PYTHON_EXECUTABLE, local venv/, and canonical parent venv/ for worktrees.
find_stubgen() {
    if [ -n "${STUBGEN:-}" ] && [ -x "${STUBGEN}" ]; then
        echo "${STUBGEN}"
        return 0
    fi
    if command -v pybind11-stubgen >/dev/null 2>&1; then
        command -v pybind11-stubgen
        return 0
    fi
    local candidates=()
    if [ -n "${PYTHON_EXECUTABLE:-}" ]; then
        candidates+=("$(dirname "${PYTHON_EXECUTABLE}")/pybind11-stubgen")
    fi
    candidates+=(
        "${PROJECT_ROOT}/venv/bin/pybind11-stubgen"
        "${PROJECT_ROOT}/../../../venv/bin/pybind11-stubgen"
    )
    for c in "${candidates[@]}"; do
        if [ -x "${c}" ]; then
            echo "${c}"
            return 0
        fi
    done
    return 1
}

STUBGEN_BIN="$(find_stubgen)" || {
    echo "Error: pybind11-stubgen not found in PATH or project venv." >&2
    exit 1
}
PYTHON_BIN="${PYTHON_EXECUTABLE:-$(dirname "${STUBGEN_BIN}")/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

# Stub generation only introspects type signatures; it must not call
# MPI_Init because MPI_Finalize (via atexit) would run before pybind11
# and C++ static destructors, causing a double-free crash.
export DNDSR_SKIP_MPI_INIT=1
export PYTHONPATH="${PYTHON_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Clean previous output
rm -rf "${STUBS_DIR}"
mkdir -p "${STUBS_DIR}"

echo "=== Generating stubs for DNDSR.DNDS ==="
"${STUBGEN_BIN}" DNDSR.DNDS -o "${STUBS_DIR}" || exit 1

echo "=== Generating stubs for DNDSR.Geom ==="
"${STUBGEN_BIN}" DNDSR.Geom -o "${STUBS_DIR}" || exit 1

echo "=== Generating stubs for DNDSR.CFV ==="
"${STUBGEN_BIN}" DNDSR.CFV -o "${STUBS_DIR}" || exit 1

echo "=== Generating stubs for DNDSR.EulerP ==="
"${STUBGEN_BIN}" DNDSR.EulerP -o "${STUBS_DIR}" || exit 1

# Cauchy is optional for installations building only the historical targets.
MODULES=(DNDS Geom CFV EulerP)
if compgen -G "${PYTHON_DIR}/DNDSR/Cauchy/_ext/cauchy_pybind11*.so" > /dev/null; then
    echo "=== Generating stubs for DNDSR.Cauchy ==="
    "${STUBGEN_BIN}" DNDSR.Cauchy -o "${STUBS_DIR}" || exit 1
    MODULES+=(Cauchy)
fi

# For each module, flatten the _ext/<binding>/*.pyi submodule stubs
# into the package-level directory so that IDEs see them.
for MODULE in "${MODULES[@]}"; do
    EXT_DIR=$(find "${STUBS_DIR}/DNDSR/${MODULE}/_ext" -mindepth 1 -maxdepth 1 -type d 2>/dev/null || true)
    if [ -n "${EXT_DIR}" ]; then
        for stub in "${EXT_DIR}"/*.pyi; do
            [ -f "$stub" ] || continue
            base=$(basename "$stub")
            dest="${STUBS_DIR}/DNDSR/${MODULE}/${base}"
            cp -v "$stub" "$dest"
        done
    fi
done

# Copy stubs into the python package tree for PEP 561 compliance
echo "=== Copying stubs into python/DNDSR/ ==="
for MODULE in "${MODULES[@]}"; do
    SRC="${STUBS_DIR}/DNDSR/${MODULE}"
    DST="${PYTHON_DIR}/DNDSR/${MODULE}"
    if [ -d "${SRC}" ]; then
        find "${SRC}" -name '*.pyi' | while read -r pyi; do
            rel="${pyi#${SRC}/}"
            mkdir -p "$(dirname "${DST}/${rel}")"
            cp -v "$pyi" "${DST}/${rel}"
        done
    fi
done

# Also copy the top-level DNDSR/__init__.pyi if generated
if [ -f "${STUBS_DIR}/DNDSR/__init__.pyi" ]; then
    cp -v "${STUBS_DIR}/DNDSR/__init__.pyi" "${PYTHON_DIR}/DNDSR/__init__.pyi"
fi

# Hoist and deduplicate `from __future__ import ...` directives to the top of
# every generated .pyi file and gate on Python syntax compilation.
echo "=== Normalizing and syntax-checking .pyi stubs ==="
"${PYTHON_BIN}" - "${STUBS_DIR}/DNDSR" "${PYTHON_DIR}/DNDSR" <<'PYEOF'
import pathlib
import py_compile
import re
import sys

FUTURE_RE = re.compile(r"^\s*from\s+__future__\s+import\s+.+$")

def normalize_pyi(path: pathlib.Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Preserve leading comments / blank lines / module docstring before future imports.
    idx = 0
    n = len(lines)
    prefix = []
    while idx < n and (lines[idx].strip() == "" or lines[idx].lstrip().startswith("#")):
        prefix.append(lines[idx])
        idx += 1
    if idx < n:
        stripped = lines[idx].lstrip()
        for quote in ('"""', "'''"):
            if stripped.startswith(quote):
                prefix.append(lines[idx])
                if stripped.count(quote) >= 2 and len(stripped.rstrip()) >= 2 * len(quote):
                    idx += 1
                else:
                    idx += 1
                    while idx < n:
                        prefix.append(lines[idx])
                        if quote in lines[idx]:
                            idx += 1
                            break
                        idx += 1
                break

    futures = []
    seen_futures = set()
    body = []
    for line in lines[idx:]:
        if FUTURE_RE.match(line):
            canonical = line.strip()
            if canonical not in seen_futures:
                seen_futures.add(canonical)
                futures.append(canonical + "\n")
        else:
            body.append(line)

    if futures:
        new_text = "".join(prefix) + "".join(futures) + "".join(body)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")

checked = 0
for root_arg in sys.argv[1:]:
    root = pathlib.Path(root_arg)
    if not root.exists():
        continue
    for pyi in sorted(root.rglob("*.pyi")):
        normalize_pyi(pyi)
        py_compile.compile(str(pyi), doraise=True)
        checked += 1

print(f"Verified {checked} .pyi stub files with zero syntax errors.")
PYEOF

echo "=== Done. Stubs in ${STUBS_DIR}/ and ${PYTHON_DIR}/DNDSR/ ==="
