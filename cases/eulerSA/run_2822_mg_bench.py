import argparse
import importlib
import os
import shlex
import subprocess
import sys
import pprint

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirname, "..", "..", "scripts"))

GraceExit = importlib.import_module("utils.GraceExit").GraceExit

handler = GraceExit(max_attempts=5)

config_name = os.path.join(dirname, "config_2822_mg_bench.json")

out_base = "../data/out/RAE2822_M0-MGtest_R0.dir"

name_prefix = ""

name_prefix = "x0-"

os.makedirs(out_base, exist_ok=True)

nProc = 32

opt_0 = {}
opt_0["n_iter"] = 4000
opt_0["gC_base"] = 1
opt_0["jC_base"] = 2
opt_0["name_base"] = "gmres5x1ilu"
opt_0["jC_append"] = 2
opt_0["name_append"] = "ilu"  # smoother

opt_1 = {}
opt_1["n_iter"] = 4000
opt_1["gC_base"] = 0
opt_1["jC_base"] = 2
opt_1["name_base"] = "ilu"
opt_1["jC_append"] = 2
opt_1["name_append"] = "ilu"  # smoother

opt_2 = {}
opt_2["n_iter"] = 8000
opt_2["gC_base"] = 0
opt_2["jC_base"] = 1
opt_2["name_base"] = "lusgs"
opt_2["jC_append"] = 1
opt_2["name_append"] = "lusgs"  # smoother

opt_3 = {}
opt_3["n_iter"] = 4000
opt_3["gC_base"] = 1
opt_3["jC_base"] = 1
opt_3["name_base"] = "gmres5x1lusgs"
opt_3["jC_append"] = 1
opt_3["name_append"] = "lusgs"  # smoother

opts = [
    opt_0,  # GMRES-ILU
    # opt_1,  # ILU
    # opt_3,  # GMRES-LUSGS
    opt_2,  # LUSGS
]

mg_seqs = [
    (0, 0, 0),
    # # ! 1 level
    (1, 2, 0),
    (1, 4, 0),
    (1, 8, 0),
    # (1, 16, 0),
    # ! 2 level thin
    (2, 1, 2),
    (2, 1, 4),
    (2, 1, 8),
    # (2, 1, 16),
    # ! 2 level mul 2
    (2, 2, 2),
    (2, 2, 4),
    (2, 2, 8),
    # (2, 2, 16),
    # ! 2 level mul 4
    (2, 4, 2),
    (2, 4, 4),
    (2, 4, 8),
    # (2, 4, 16),
    # ! 2 level mul 6
    (2, 8, 2),
    (2, 8, 4),
    (2, 8, 8),
    # (2, 8, 16),
]

# mg_seqs = [(0, 0, 0)] #! overwrite: only no mg cases


def get_options(
    n_iter,
    gC_base,
    jC_base,
    name_base,
    jC_append,
    name_append,
):
    options_list = {}
    for mg_set in mg_seqs:
        options = []
        options.append(
            (
                "/convergenceControl/nTimeStepInternal",
                n_iter * 2 if mg_set[0] == 0 else n_iter,
            )
        )
        options.append(("/linearSolverControl/jacobiCode", jC_base))
        options.append(("/linearSolverControl/gmresCode", gC_base))
        options.append(
            (
                "/linearSolverControl/coarseGridLinearSolverControlList/1/jacobiCode",
                jC_append,
            )
        )
        options.append(
            (
                "/linearSolverControl/coarseGridLinearSolverControlList/2/jacobiCode",
                jC_append,
            )
        )
        name = name_prefix + name_base
        for i in (1, 2):
            name += f"-{mg_set[i]}{name_append}" if i <= mg_set[0] else ""
        options.append(
            (
                "/dataIOControl/outPltName",
                f'"{os.path.join(out_base, name)}"',
            )
        )
        options.extend(
            [
                ("/linearSolverControl/multiGridLP", mg_set[0]),
                (
                    "/linearSolverControl/coarseGridLinearSolverControlList/1/multiGridNIter",
                    mg_set[1],
                ),
                (
                    "/linearSolverControl/coarseGridLinearSolverControlList/2/multiGridNIter",
                    mg_set[2],
                ),
            ]
        )
        options_list[name] = options
    return options_list


options_list = {}

for opt in opts:
    options_list.update(get_options(**opt))


def build_command(options):
    command = ["mpirun", "-np", str(nProc), "app/eulerSA.exe", config_name]
    for key, value in options:
        command.extend(["-k", key, "-v", str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run the RAE 2822 MG benchmark sweep")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every command without launching the solver",
    )
    args = parser.parse_args()

    pprint.pprint(options_list)
    for name, options in options_list.items():
        command = build_command(options)
        stdout_path = os.path.join(out_base, name) + "-stdout.txt"
        print(f"Command::: {name}\n\n {shlex.join(command)} > {stdout_path}\n")
        if args.dry_run:
            continue
        with open(stdout_path, "w") as stdout:
            subprocess.run(command, stdout=stdout, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
