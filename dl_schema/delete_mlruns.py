"""Tool to clean up deleted mlflow runs, since UI only changes a given runs lifecycle
status metadata"""
from pathlib import Path
import shutil

import mlflow


def get_run_dir(artifacts_uri):
    return artifacts_uri[7:-10]


def remove_run_dir(run_dir):
    shutil.rmtree(run_dir, ignore_errors=True)


def find_experiments(base_dir):
    exp_ids = []
    for p in Path(base_dir).iterdir():
        if "trash" not in p.stem and p.stem.isnumeric():
            exp_ids.append(p.stem)
    return exp_ids


def purge_runs(exp_id, dry_run=False):
    deleted_runs = 2
    exp = mlflow.tracking.MlflowClient(tracking_uri="./mlruns")
    runs = exp.search_runs(str(exp_id), run_view_type=deleted_runs)
    run_dirs = [get_run_dir(run.info.artifact_uri) for run in runs]

    print("deleting runs:")
    print(*run_dirs, sep="\n")

    if not dry_run:
        _ = [remove_run_dir(d) for d in run_dirs]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tool to clean up deleted mlflow runs, since UI only changes a given"
        + " runs lifecycle status metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--exp_id",
        type=str,
        default="test_input.png or dir",
        help="input image path",
    )
    parser.add_argument(
        "--all_runs",
        action="store_true",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )
    base_dir = "./mlruns"
    exp_ids = find_experiments(base_dir)
    args = parser.parse_args()

    if args.all_runs:
        for exp in exp_ids:
            purge_runs(exp, args.dry_run)
    else:
        purge_runs(args.exp_id, args.dry_run)
