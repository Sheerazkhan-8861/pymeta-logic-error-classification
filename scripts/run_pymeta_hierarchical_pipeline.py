#!/usr/bin/env python3
"""Fail-fast, resumable PyMETA hierarchical experiment orchestrator.

Training requires BOTH --execute and --confirm-training YES.
This script does not modify the training engine, evaluator, datasets or baseline.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / "pymeta_project"
sys.path.insert(0, str(ROOT))

from scripts import pymeta_hierarchical_training_engine as engine


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fail_if_paths_exist(paths, label):
    found = [str(p) for p in paths if p.exists()]
    if found:
        raise RuntimeError(
            f"{label}; refusing to proceed:\n  " + "\n  ".join(found)
        )


def recovery_sidecars(recovery):
    return [
        recovery.with_name(recovery.name + suffix)
        for suffix in (
            ".previous", ".temporary", ".corrupt",
            ".previous.corrupt", ".failed_restore",
        )
    ]


def snapshot_sidecars(output_dir, epoch):
    base = output_dir / f"epoch_{epoch}"
    return [
        base.with_name(base.name + suffix)
        for suffix in (".temporary", ".previous", ".corrupt", ".failed")
    ]


def verified_stage_artifact_names(stage, output_dir, recovery_epoch):
    """Verify completed snapshots and permit only the immediate next-epoch orphan."""
    expected_names = {"recovery"}

    for completed_epoch in range(1, recovery_epoch + 1):
        snapshot = output_dir / f"epoch_{completed_epoch}"
        engine.verify_epoch_snapshot(snapshot, stage, completed_epoch)
        expected_names.add(snapshot.name)

    next_epoch = recovery_epoch + 1
    if next_epoch <= engine.EPOCHS:
        orphan = output_dir / f"epoch_{next_epoch}"
        if orphan.exists():
            engine.verify_epoch_snapshot(orphan, stage, next_epoch)
            expected_names.add(orphan.name)

    return expected_names


def inspect_stage(stage):
    cfg = engine.STAGES[stage]
    output_dir = cfg["output_dir"]
    recovery = output_dir / "recovery"

    fail_if_paths_exist(
        recovery_sidecars(recovery),
        f"{stage}: recovery sidecar detected",
    )

    for epoch in (1, 2):
        fail_if_paths_exist(
            snapshot_sidecars(output_dir, epoch),
            f"{stage}: epoch snapshot sidecar detected",
        )

    train_hash, val_hash = engine.verify_stage_inputs(stage)

    train_records = engine.load_records(
        cfg["train_path"], cfg["train_rows"], cfg["classes"]
    )
    val_records = engine.load_records(
        cfg["validation_path"], cfg["validation_rows"], cfg["classes"]
    )

    if (
        len(train_records) != cfg["train_rows"]
        or len(val_records) != cfg["validation_rows"]
    ):
        raise RuntimeError(f"{stage}: loaded row counts differ from configuration")

    if not recovery.is_dir():
        raise RuntimeError(f"{stage}: recovery checkpoint missing: {recovery}")

    state = engine.verify_recovery_checkpoint(
        recovery, cfg, stage, train_hash, val_hash
    )

    epoch = state["epoch"]
    if not 0 <= epoch <= engine.EPOCHS:
        raise RuntimeError(f"{stage}: invalid recovered epoch {epoch}")

    expected_names = verified_stage_artifact_names(
        stage, output_dir, epoch
    )

    summary_path = output_dir / "training_summary.json"

    if epoch == engine.EPOCHS:
        if summary_path.exists():
            summary = read_json(summary_path)
            if (
                summary.get("stage") != stage
                or summary.get("completed_epochs") != engine.EPOCHS
                or summary.get("successful_optimizer_updates") != cfg["total_steps"]
                or summary.get("test_records_accessed") is not False
                or not isinstance(summary.get("history"), list)
                or len(summary["history"]) != engine.EPOCHS
            ):
                raise RuntimeError(f"{stage}: training summary failed checks")
            expected_names.add(summary_path.name)
        # A missing summary can be finalized by the engine from completed state.
    elif summary_path.exists():
        raise RuntimeError(
            f"{stage}: summary exists although recovery state is incomplete"
        )

    actual_names = {p.name for p in output_dir.iterdir()}
    unexpected = sorted(actual_names - expected_names)

    if unexpected:
        raise RuntimeError(
            f"{stage}: unexpected stage-output artifacts: {unexpected}"
        )

    return {
        "stage": stage,
        "epoch": epoch,
        "global_step": state["global_step"],
        "next_position": state["next_position"],
        "train_rows": len(train_records),
        "validation_rows": len(val_records),
        "train_sha256": train_hash,
        "validation_sha256": val_hash,
        "summary_exists": summary_path.is_file(),
    }


def verify_stage_complete(stage):
    info = inspect_stage(stage)
    cfg = engine.STAGES[stage]

    if (
        info["epoch"] != engine.EPOCHS
        or info["global_step"] != cfg["total_steps"]
        or info["next_position"] != cfg["train_rows"]
        or not (cfg["output_dir"] / "training_summary.json").is_file()
    ):
        raise RuntimeError(f"{stage}: completion criteria not met: {info}")

    return info


def cascade_destinations():
    output_dir = (
        ROOT / "outputs" / "hierarchical_validation" / "cascade_selection"
    )
    best_dir = ROOT / "checkpoints" / "hierarchical_best_pair"

    fail_if_paths_exist(
        [
            output_dir,
            output_dir.with_name(output_dir.name + ".temporary"),
            best_dir,
            best_dir.with_name(best_dir.name + ".temporary"),
        ],
        "Cascade destination already exists",
    )
    return output_dir, best_dir


def preflight():
    engine.verify_environment()
    stage1 = inspect_stage("stage1")
    stage2 = inspect_stage("stage2")
    output_dir, best_dir = cascade_destinations()

    return {
        "stage1": stage1,
        "stage2": stage2,
        "cascade_output_dir": str(output_dir),
        "best_pair_dir": str(best_dir),
        "test_records_accessed": False,
    }


def write_audit(path, audit):
    temporary = path.with_suffix(".json.tmp")

    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

    os.replace(temporary, path)


def stop_child_process(process, timeout=10):
    """Terminate an active child; kill it if graceful termination times out."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_logged(command, env, log_path, label):
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {label} =====\nCOMMAND: {command!r}\n")
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            assert process.stdout is not None

            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

            code = process.wait()
            log.write(f"\nEXIT_CODE: {code}\n")
            log.flush()

        except BaseException:
            log.write("\nPARENT INTERRUPTED; stopping child process.\n")
            log.flush()
            stop_child_process(process)
            raise

        finally:
            if process.stdout is not None:
                process.stdout.close()

    if code != 0:
        raise RuntimeError(
            f"{label} failed with exit code {code}; see {log_path}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")

    parser.add_argument(
        "--confirm-training",
        choices=["YES"],
        help="Required with --execute to authorize training.",
    )

    args = parser.parse_args()

    if args.execute and args.confirm_training != "YES":
        parser.error("--execute requires --confirm-training YES")

    if args.dry_run and args.confirm_training is not None:
        parser.error("--confirm-training is only valid with --execute")

    print("========== PyMETA HIERARCHICAL PIPELINE PREFLIGHT ==========")

    report = preflight()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nPREFLIGHT: PASS")

    if args.dry_run:
        print("\nDRY RUN ONLY — no logs, checkpoints or model outputs created.")
        print(
            "Planned order: Stage 1 -> verify -> Stage 2 -> verify "
            "-> full-dev cascade -> final audit."
        )
        print("Training has NOT started.")
        return

    run_root = ROOT / "outputs" / "hierarchical_pipeline_runs"
    run_root.mkdir(parents=True, exist_ok=True)

    run_dir = run_root / datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )

    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    run_dir.mkdir()
    log_path = run_dir / "pipeline.log"
    audit_path = run_dir / "pipeline_audit.json"

    audit = {
        "project": "PyMETA",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "preflight": report,
        "events": [],
        "test_records_accessed": False,
    }
    write_audit(audit_path, audit)

    def event(name, details):
        audit["events"].append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": name,
            "details": details,
        })
        write_audit(audit_path, audit)

    try:
        train_env = os.environ.copy()
        train_env["PYMETA_START_TRAINING"] = "YES"

        for stage in ("stage1", "stage2"):
            before = inspect_stage(stage)

            if (
                before["epoch"] == engine.EPOCHS
                and before["summary_exists"]
            ):
                print(f"\n{stage}: already complete and verified; skipping.")
                event(f"{stage}_skipped_already_complete", before)
            else:
                command = [
                    sys.executable,
                    str(
                        ROOT / "scripts"
                        / "pymeta_hierarchical_training_engine.py"
                    ),
                    "--stage",
                    stage,
                ]

                event(f"{stage}_started_or_resumed", before)
                run_logged(command, train_env, log_path, f"TRAIN {stage}")

                completed = verify_stage_complete(stage)
                event(f"{stage}_completed_and_verified", completed)
                print(f"\n{stage.upper()} COMPLETED AND VERIFIED.")

        # Recheck both stages immediately before cascade evaluation.
        stage1_final = verify_stage_complete("stage1")
        stage2_final = verify_stage_complete("stage2")
        output_dir, best_dir = cascade_destinations()

        cascade_env = os.environ.copy()
        cascade_env.pop("PYMETA_START_TRAINING", None)

        command = [
            sys.executable,
            str(
                ROOT / "scripts"
                / "pymeta_hierarchical_cascade_evaluator.py"
            ),
            "--output-dir",
            str(output_dir),
            "--best-pair-dir",
            str(best_dir),
        ]

        event("cascade_started", {
            "stage1": stage1_final,
            "stage2": stage2_final,
            "output_dir": str(output_dir),
            "best_pair_dir": str(best_dir),
        })

        run_logged(command, cascade_env, log_path, "FULL DEVELOPMENT CASCADE")

        report_path = output_dir / "cascade_selection_report.json"
        manifest_path = best_dir / "best_pair_manifest.json"

        if not report_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Cascade report or best-pair manifest is missing")

        cascade_report = read_json(report_path)
        best_manifest = read_json(manifest_path)

        expected_pairs = {
            f"stage1_epoch_{a}_stage2_epoch_{b}"
            for a in (1, 2)
            for b in (1, 2)
        }

        observed_pairs = {
            item["pair"]
            for item in cascade_report.get("candidate_pairs", [])
        }

        if (
            cascade_report.get("evaluation")
            != "full_original_development_set_hierarchical_cascade"
            or cascade_report.get("development_rows") != 4863
            or observed_pairs != expected_pairs
            or cascade_report.get("test_records_accessed") is not False
            or best_manifest.get("test_records_accessed") is not False
            or best_manifest.get("best_pair") != cascade_report.get("best_pair")
        ):
            raise RuntimeError("Cascade report failed final integrity checks")

        for required_dir in (
            best_dir / "stage1_adapter",
            best_dir / "stage1_tokenizer",
            best_dir / "stage2_adapter",
            best_dir / "stage2_tokenizer",
        ):
            if (
                not required_dir.is_dir()
                or not any(p.is_file() for p in required_dir.rglob("*"))
            ):
                raise RuntimeError(
                    f"Best-pair artifact missing or empty: {required_dir}"
                )

        event("cascade_completed_and_verified", {
            "report_sha256": sha256(report_path),
            "best_pair_manifest_sha256": sha256(manifest_path),
            "best_pair": cascade_report["best_pair"],
            "candidate_pairs": sorted(observed_pairs),
            "test_records_accessed": False,
        })

        audit["completed_utc"] = datetime.now(timezone.utc).isoformat()
        audit["status"] = "PASS"
        write_audit(audit_path, audit)

        print("\nPIPELINE COMPLETED. BOTH STAGES AND CASCADE VERIFIED.")
        print("Run directory:", run_dir)
        print("Audit:", audit_path)

    except (Exception, KeyboardInterrupt) as exc:
        audit["status"] = "FAILED_OR_INTERRUPTED"
        audit["failure"] = repr(exc)
        audit["failed_utc"] = datetime.now(timezone.utc).isoformat()
        write_audit(audit_path, audit)

        print(f"\nPIPELINE STOPPED: {exc}", file=sys.stderr)
        print(
            "Existing verified checkpoints and outputs have been preserved "
            "where the underlying engine permits."
        )
        print("Audit:", audit_path, file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
