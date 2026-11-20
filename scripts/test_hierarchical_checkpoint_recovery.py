#!/usr/bin/env python3
"""Isolated tests for PyMETA hierarchical checkpoint and epoch-snapshot recovery."""

import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

import pymeta_hierarchical_training_engine as engine


def create_fake_checkpoint(directory, stage, train_hash, validation_hash, format_version=3):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter").mkdir()
    (directory / "tokenizer").mkdir()

    (directory / "adapter" / "adapter.bin").write_bytes(b"fake-adapter-weights")
    (directory / "tokenizer" / "tokenizer.json").write_text(
        '{"test": true}',
        encoding="utf-8",
    )

    state = {
        "optimizer": {},
        "scheduler": {"last_epoch": 0},
        "scaler": {},
        "epoch": 0,
        "permutation": torch.randperm(
            engine.STAGES[stage]["train_rows"],
            generator=torch.Generator().manual_seed(42),
        ),
        "next_position": 0,
        "global_step": 0,
        "skipped_overflow_attempts": 0,
        "history": [],
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [],
            "generator": torch.Generator().manual_seed(42).get_state(),
        },
    }
    if format_version == 3:
        state["elapsed_training_seconds"] = 0.0
    torch.save(state, directory / "training_state.pt")

    inventory = engine.checkpoint_inventory(directory)
    manifest = {
        "project": "PyMETA",
        "checkpoint_format_version": format_version,
        "stage": stage,
        "experiment_spec_sha256": engine.sha256_file(engine.SPEC_PATH),
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "epoch": 0,
        "next_position": 0,
        "global_step": 0,
        "test_records_accessed": False,
        "training_started": (False if format_version == 3 else True),
        "checkpoint_files": inventory,
    }
    if format_version == 3:
        manifest["elapsed_training_seconds"] = 0.0
    with open(directory / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


class FakeModel:
    def __init__(self, payload):
        self.payload = payload

    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter.bin").write_bytes(self.payload)


class FakeTokenizer:
    def save_pretrained(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text(
            '{"fake_tokenizer": true}',
            encoding="utf-8",
        )


def main():
    stage = "stage1"
    train_hash = "test-train-hash"
    validation_hash = "test-validation-hash"

    with tempfile.TemporaryDirectory(prefix="pymeta_checkpoint_test_") as temp:
        root = Path(temp)
        recovery = root / "recovery"
        backup = root / "recovery.previous"
        corrupt = root / "recovery.corrupt"

        # TEST 1 — Valid checkpoint passes full verification.
        create_fake_checkpoint(recovery, stage, train_hash, validation_hash)
        engine.verify_recovery_checkpoint(
            recovery, {}, stage, train_hash, validation_hash
        )
        print("TEST 1 — Valid checkpoint: PASS")

        # TEST 2 — A changed file is rejected by SHA-256 verification.
        with open(recovery / "adapter" / "adapter.bin", "ab") as handle:
            handle.write(b"CORRUPTION")

        try:
            engine.verify_recovery_checkpoint(
                recovery, {}, stage, train_hash, validation_hash
            )
        except RuntimeError as exc:
            assert "SHA-256 verification failed" in str(exc), str(exc)
        else:
            raise AssertionError("Verifier accepted a corrupted checkpoint.")
        print("TEST 2 — SHA-256 corruption detection: PASS")

        # TEST 3 — A verified .previous is restored; corrupt current is preserved.
        create_fake_checkpoint(backup, stage, train_hash, validation_hash)
        restored = engine.recover_interrupted_checkpoint(
            recovery, {}, stage, train_hash, validation_hash
        )
        assert restored is True
        assert recovery.is_dir()
        assert not backup.exists()
        assert corrupt.is_dir(), "Corrupt checkpoint was not preserved."
        engine.verify_recovery_checkpoint(
            recovery, {}, stage, train_hash, validation_hash
        )
        print("TEST 3 — Previous checkpoint restoration: PASS")

        # TEST 4 — A valid current checkpoint is retained over a valid backup.
        create_fake_checkpoint(backup, stage, train_hash, validation_hash)
        engine.recover_interrupted_checkpoint(
            recovery, {}, stage, train_hash, validation_hash
        )
        assert recovery.is_dir()
        assert not backup.exists()
        engine.verify_recovery_checkpoint(
            recovery, {}, stage, train_hash, validation_hash
        )
        print("TEST 4 — Valid current checkpoint retained: PASS")

        # TEST 5 — An invalid previous checkpoint is preserved, not deleted.
        create_fake_checkpoint(backup, stage, train_hash, validation_hash)
        with open(backup / "adapter" / "adapter.bin", "ab") as handle:
            handle.write(b"CORRUPTION")
        engine.recover_interrupted_checkpoint(
            recovery, {}, stage, train_hash, validation_hash
        )
        invalid_backup = root / "recovery.previous.corrupt"
        assert invalid_backup.is_dir()
        assert recovery.is_dir()
        engine.verify_recovery_checkpoint(
            recovery, {}, stage, train_hash, validation_hash
        )
        print("TEST 5 — Invalid previous checkpoint preservation: PASS")

        # TEST 6 — Epoch snapshots can be safely replaced after an interrupted save.
        output_dir = root / "stage1_snapshots"
        fake_model = FakeModel(b"epoch-one")
        fake_tokenizer = FakeTokenizer()

        engine.save_epoch_adapter(
            fake_model, fake_tokenizer, output_dir, stage, 1,
            {"validation_loss": 1.0},
        )
        epoch_dir = output_dir / "epoch_1"
        engine.verify_epoch_snapshot(epoch_dir, stage, 1)

        # Simulate a crash after the old snapshot was moved to .previous.
        epoch_backup = output_dir / "epoch_1.previous"
        epoch_dir.replace(epoch_backup)

        fake_model.payload = b"recomputed-epoch-one"
        engine.save_epoch_adapter(
            fake_model, fake_tokenizer, output_dir, stage, 1,
            {"validation_loss": 0.9},
        )
        snapshot_manifest = engine.verify_epoch_snapshot(epoch_dir, stage, 1)
        assert not epoch_backup.exists()
        assert snapshot_manifest["validation_metrics"]["validation_loss"] == 0.9
        print("TEST 6 — Epoch snapshot interrupted-replacement recovery: PASS")

        # TEST 7 — A full initial recovery checkpoint is created and verified.
        class FakeStateObject:
            def __init__(self, state):
                self.state = state

            def state_dict(self):
                return self.state

        new_recovery = root / "new_recovery"
        generator = torch.Generator().manual_seed(42)
        initial_state = {
            "epoch": 0,
            "permutation": torch.randperm(
                engine.STAGES[stage]["train_rows"],
                generator=torch.Generator().manual_seed(42),
            ),
            "next_position": 0,
            "global_step": 0,
            "skipped_overflow_attempts": 0,
            "history": [],
            "elapsed_training_seconds": 0.0,
        }
        engine.save_recovery_checkpoint(
            new_recovery,
            FakeModel(b"initial-adapter"),
            FakeTokenizer(),
            FakeStateObject({}),
            FakeStateObject({"last_epoch": 0}),
            FakeStateObject({}),
            initial_state,
            generator,
            {},
            stage,
            train_hash,
            validation_hash,
        )
        saved_state = engine.verify_recovery_checkpoint(
            new_recovery, {}, stage, train_hash, validation_hash
        )
        assert saved_state["global_step"] == 0
        assert saved_state["elapsed_training_seconds"] == 0.0
        print("TEST 7 — Full initial recovery checkpoint creation: PASS")

        # TEST 8 — Existing format-v2 recovery checkpoints remain resumable.
        legacy_recovery = root / "legacy_recovery"
        create_fake_checkpoint(
            legacy_recovery,
            stage,
            train_hash,
            validation_hash,
            format_version=2,
        )
        legacy_state = engine.verify_recovery_checkpoint(
            legacy_recovery, {}, stage, train_hash, validation_hash
        )
        assert legacy_state["elapsed_training_seconds"] == 0.0
        print("TEST 8 — Version-2 checkpoint backward compatibility: PASS")

    print("\nALL CHECKPOINT AND EPOCH SNAPSHOT TESTS PASSED")


if __name__ == "__main__":
    main()
