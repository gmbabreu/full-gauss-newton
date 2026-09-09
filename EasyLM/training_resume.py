"""Validation for complete, reproducible raw-HF Adam checkpoints."""
import os
import re


def resume_companion_paths(checkpoint_path):
    match = re.fullmatch(r"streaming_train_state(_[0-9]+)?", os.path.basename(checkpoint_path))
    if match is None:
        raise ValueError("Full resume requires a streaming_train_state checkpoint")
    suffix = match.group(1) or ""
    directory = os.path.dirname(checkpoint_path)
    return {name: os.path.join(directory, f"{name}{suffix}.pkl")
            for name in ("metadata", "dataset", "complete")}


def validate_resume_metadata(metadata, complete, flags):
    if not isinstance(metadata, dict) or metadata.get("resume_version") != 1:
        raise ValueError("Checkpoint lacks complete RNG/data resume metadata")
    if (not metadata.get("optimizer_saved")
            or metadata.get("checkpoint_float_dtype") != "fp32"):
        raise ValueError("Exact Adam resume requires optimizer state saved in fp32")
    if "train_rng" not in metadata or "step" not in metadata:
        raise ValueError("Checkpoint lacks RNG or completed-step count")
    if (not isinstance(complete, dict) or complete.get("resume_version") != 1
            or not metadata.get("snapshot_id")
            or complete.get("snapshot_id") != metadata["snapshot_id"]
            or complete.get("step") != metadata["step"]):
        raise ValueError("Checkpoint is incomplete or its completion record does not match")
    saved = metadata["flags"]
    keys = ("seed", "mesh_dim", "dtype", "param_dtype", "total_steps", "tokenizer",
            "llama", "optimizer", "train_dataset", "train_dataset_batch_size",
            "eval_dataset", "eval_steps", "eval_freq", "log_freq", "weight_average",
            "weight_average_decay", "target_loss")
    changed = [key for key in keys if saved.get(key) != flags.get(key)]
    if changed:
        raise ValueError("Resume changes trajectory settings: " + ", ".join(changed))


def validate_dataset_snapshot(state, metadata):
    if not isinstance(state, dict) or state.get("packed_state_version") != 1:
        raise ValueError("Exact resume requires the saved token-packing remainder")
    if (state.get("training_step") != metadata["step"]
            or state.get("snapshot_id") != metadata["snapshot_id"]):
        raise ValueError("Dataset and model metadata belong to different snapshots")
    cfg = metadata["flags"]["train_dataset"]["huggingface_dataset"]
    expected_tokens = (cfg["tokens_count_at_start"] + metadata["step"]
                       * cfg["batch_size"] * cfg["seq_length"])
    if state["metadata"]["dataset_total_tokens"] != expected_tokens:
        raise ValueError("Packed dataset token count does not match completed updates")
