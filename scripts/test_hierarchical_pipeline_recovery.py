#!/usr/bin/env python3
"""Targeted tests for hierarchical pipeline recovery and interruption."""

import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import pymeta_hierarchical_training_engine as engine
from scripts import run_pymeta_hierarchical_pipeline as pipeline


class FakeModel:
    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter.bin").write_bytes(b"test-adapter")


class FakeTokenizer:
    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(
            '{"test": true}', encoding="utf-8"
        )


def create_snapshot(directory, stage, epoch):
    engine.save_epoch_adapter(
        FakeModel(),
        FakeTokenizer(),
        directory,
        stage,
        epoch,
        {"eval_loss": 0.5},
    )


def test_valid_next_epoch_orphan():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        create_snapshot(output, "stage1", 1)

        allowed = pipeline.verified_stage_artifact_names(
            "stage1", output, 0
        )
        assert allowed == {"recovery", "epoch_1"}
        print("PASS: valid next-epoch orphan accepted")


def test_corrupt_next_epoch_orphan_rejected():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        create_snapshot(output, "stage1", 1)
        (output / "epoch_1" / "adapter" / "adapter.bin").write_bytes(
            b"corrupted"
        )

        try:
            pipeline.verified_stage_artifact_names(
                "stage1", output, 0
            )
        except RuntimeError:
            print("PASS: corrupted orphan rejected")
        else:
            raise AssertionError("Corrupted orphan was accepted")


def test_future_snapshot_remains_unexpected():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        create_snapshot(output, "stage1", 2)

        allowed = pipeline.verified_stage_artifact_names(
            "stage1", output, 0
        )
        actual = {p.name for p in output.iterdir()}
        unexpected = actual - allowed

        assert "epoch_2" in unexpected
        print("PASS: future snapshot remains unexpected")


def test_completed_epoch_snapshot_verified():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp)
        create_snapshot(output, "stage1", 1)

        allowed = pipeline.verified_stage_artifact_names(
            "stage1", output, 1
        )
        assert "epoch_1" in allowed
        print("PASS: completed epoch snapshot verified")


def test_stop_child_terminates_process():
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

        def kill(self):
            raise AssertionError("Kill should not be needed")

    process = FakeProcess()
    pipeline.stop_child_process(process)

    assert process.terminated
    assert process.waited
    print("PASS: child process terminated and waited for")


def test_stop_child_kills_after_timeout():
    class FakeProcess:
        def __init__(self):
            self.wait_calls = 0
            self.terminated = False
            self.killed = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("test-child", timeout)
            return 0

        def kill(self):
            self.killed = True

    process = FakeProcess()
    pipeline.stop_child_process(process, timeout=1)

    assert process.terminated
    assert process.killed
    assert process.wait_calls == 2
    print("PASS: child process killed after termination timeout")


def test_run_logged_normal_child():
    with tempfile.TemporaryDirectory() as temp:
        log_path = Path(temp) / "child.log"

        pipeline.run_logged(
            [
                __import__("sys").executable,
                "-c",
                "print('PYMETA_CHILD_TEST_OUTPUT')",
            ],
            dict(__import__("os").environ),
            log_path,
            "TEST_CHILD",
        )

        content = log_path.read_text(encoding="utf-8")
        assert "PYMETA_CHILD_TEST_OUTPUT" in content
        assert "EXIT_CODE: 0" in content
        print("PASS: normal child output and exit code logged")


def test_run_logged_interrupt_stops_child():
    class InterruptingOutput:
        def __iter__(self):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self):
            self.stdout = InterruptingOutput()
            self.terminated = False
            self.waited = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.waited = True
            return 0

        def kill(self):
            raise AssertionError("Kill should not be needed")

    with tempfile.TemporaryDirectory() as temp:
        log_path = Path(temp) / "interrupt.log"
        process = FakeProcess()

        with patch.object(
            pipeline.subprocess, "Popen", return_value=process
        ):
            try:
                pipeline.run_logged(
                    ["fake-child"],
                    {},
                    log_path,
                    "INTERRUPTION_TEST",
                )
            except KeyboardInterrupt:
                pass
            else:
                raise AssertionError("KeyboardInterrupt was not propagated")

        assert process.terminated
        assert process.waited
        assert process.stdout.closed

        content = log_path.read_text(encoding="utf-8")
        assert "PARENT INTERRUPTED" in content
        print("PASS: interruption stops child and records log")


def main():
    tests = [
        test_valid_next_epoch_orphan,
        test_corrupt_next_epoch_orphan_rejected,
        test_future_snapshot_remains_unexpected,
        test_completed_epoch_snapshot_verified,
        test_stop_child_terminates_process,
        test_stop_child_kills_after_timeout,
        test_run_logged_normal_child,
        test_run_logged_interrupt_stops_child,
    ]

    for test in tests:
        test()

    print(f"\nALL {len(tests)} TARGETED TESTS PASSED")


if __name__ == "__main__":
    main()
