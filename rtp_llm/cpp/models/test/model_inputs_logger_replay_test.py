"""Tests for loading and validating model-input dumps.

Run the self-contained synthetic dump test:

    bazel test --config=cpu \
        //rtp_llm/cpp/models/test:model_inputs_logger_replay_test \
        --test_output=all

To generate and validate dumps from a real model, start the server in one shell:

    mkdir -p /tmp/rtp_llm_model_inputs
    CHECKPOINT_PATH=/mnt/nas1/hf/Qwen2-0.5B \
    TOKENIZER_PATH=/mnt/nas1/hf/Qwen2-0.5B \
    MODEL_TYPE=qwen_2 \
    ENABLE_MODEL_INPUTS_LOG=1 \
    LOG_PATH=/tmp/rtp_llm_model_inputs \
    START_PORT=8088 \
    python3 -m rtp_llm.start_server

Then send a request and run the dump validation in another shell:

    curl -sS -X POST http://127.0.0.1:8088 \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"Hello","generate_config":{"max_new_tokens":8}}'
    bazel test --config=cpu \
        //rtp_llm/cpp/models/test:model_inputs_logger_replay_test \
        --test_env=MODEL_INPUTS_DUMP_PATH=/tmp/rtp_llm_model_inputs \
        --test_output=all

CHECKPOINT_PATH and TOKENIZER_PATH select the model. LOG_PATH is where the
server writes model_inputs_*.pt, while MODEL_INPUTS_DUMP_PATH tells this test
which dump file or directory to validate.
"""

import io
import os
import shlex
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch


def _iter_model_inputs_dumps(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        [*path.rglob("*.pt"), *path.rglob("*.ptlog")],
        key=lambda p: str(p.relative_to(path)),
    )


def _load_model_inputs_dump_records(path: Path) -> list[dict]:
    def load_framed_records() -> list[dict]:
        records = []
        with path.open("rb") as f:
            while True:
                length_bytes = f.read(8)
                if not length_bytes:
                    break
                if len(length_bytes) != 8:
                    raise ValueError(f"incomplete record length in {path}")
                (record_size,) = struct.unpack("<Q", length_bytes)
                record = f.read(record_size)
                if len(record) != record_size:
                    raise ValueError(f"incomplete record payload in {path}")
                records.append(
                    torch.load(
                        io.BytesIO(record), map_location="cpu", weights_only=False
                    )
                )
        return records

    try:
        records = load_framed_records()
        if records:
            return records
    except Exception:
        if path.suffix == ".ptlog":
            raise

    return [torch.load(path, map_location="cpu", weights_only=False)]


def _runfile_path(relative_path: str) -> str:
    test_srcdir = Path(os.environ["TEST_SRCDIR"])
    workspace = os.environ.get("TEST_WORKSPACE", "rtp_llm")
    candidates = [
        test_srcdir / workspace / relative_path,
        test_srcdir / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"runfile not found: {relative_path}")


def _is_tensor(value) -> bool:
    return isinstance(value, torch.Tensor)


def _is_replayable_payload(payload: dict) -> bool:
    return (
        not bool(payload.get("warmup", False))
        and not bool(payload.get("skip_run", False))
        and _is_tensor(payload.get("kv_cache_block_id"))
        and _is_tensor(payload.get("combo_tokens"))
        and _is_tensor(payload.get("input_lengths"))
        and _is_tensor(payload.get("sequence_lengths"))
    )


def _to_cpu_i32(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device="cpu", dtype=torch.int32).contiguous()


def _expand_kernel_block_ids(
    block_ids: torch.Tensor, kernel_blocks_per_kv_block: int
) -> torch.Tensor:
    if kernel_blocks_per_kv_block <= 1:
        return block_ids.clone()
    offsets = torch.arange(kernel_blocks_per_kv_block, dtype=torch.int32).view(
        *([1] * block_ids.dim()), kernel_blocks_per_kv_block
    )
    expanded = block_ids.unsqueeze(-1) * kernel_blocks_per_kv_block + offsets
    expanded = torch.where(
        block_ids.unsqueeze(-1) == -1, torch.full_like(expanded, -1), expanded
    )
    return expanded.reshape(
        *block_ids.shape[:-1], block_ids.shape[-1] * kernel_blocks_per_kv_block
    )


def _pad_last_dim(tensor: torch.Tensor, size: int) -> torch.Tensor:
    if tensor.size(-1) == size:
        return tensor
    padded = torch.zeros(*tensor.shape[:-1], size, dtype=tensor.dtype)
    padded[..., : tensor.size(-1)] = tensor
    return padded


def _derive_kv_cache_kernel_block_id(payload: dict) -> torch.Tensor:
    block_id = payload["kv_cache_block_id"]
    if not _is_tensor(block_id) or block_id.numel() == 0:
        return torch.empty(0, dtype=torch.int32)

    block_id = _to_cpu_i32(block_id)
    seq_size_per_block = int(payload["seq_size_per_block"])
    kernel_seq_size_per_block = int(payload["kernel_seq_size_per_block"])
    full_group_bpk = 1
    if kernel_seq_size_per_block > 0:
        full_group_bpk = max(1, seq_size_per_block // kernel_seq_size_per_block)

    group_types = payload.get("kv_cache_group_types")
    group_types = (
        _to_cpu_i32(group_types).flatten()
        if _is_tensor(group_types)
        else torch.empty(0, dtype=torch.int32)
    )

    if block_id.dim() != 3:
        group_type = int(group_types[0].item()) if group_types.numel() else 1
        return _expand_kernel_block_ids(
            block_id, full_group_bpk if group_type == 1 else 1
        )

    groups = []
    kernel_block_count = block_id.size(-1) * full_group_bpk
    for group_idx in range(block_id.size(0)):
        group_type = (
            int(group_types[group_idx].item()) if group_idx < group_types.numel() else 1
        )
        group_kernel_blocks = _expand_kernel_block_ids(
            block_id[group_idx], full_group_bpk if group_type == 1 else 1
        )
        groups.append(_pad_last_dim(group_kernel_blocks, kernel_block_count))
    return torch.stack(groups, dim=0)


def _validate_model_inputs_dump(path: Path) -> dict:
    records = _load_model_inputs_dump_records(path)
    assert len(records) > 0
    payload = records[0]
    expected_keys = {
        "trace_ids",
        "combo_tokens",
        "input_lengths",
        "sequence_lengths",
        "lm_output_indexes",
        "prefix_lengths",
        "combo_tokens_type_ids",
        "combo_position_ids",
        "last_hidden_states",
        "attention_mask",
        "kv_cache_block_id",
        "kv_cache_layer_to_group",
        "kv_cache_group_types",
        "kv_cache_update_mapping",
        "request_id",
        "request_pd_separation",
        "kv_block_stride_bytes",
        "kv_scale_stride_bytes",
        "seq_size_per_block",
        "kernel_seq_size_per_block",
        "pd_separation",
        "decode_entrance",
        "need_all_logits",
        "need_moe_gating",
        "warmup",
        "skip_run",
        "is_fake_stream",
        "is_target_verify",
    }
    assert set(payload.keys()) == expected_keys
    assert payload["combo_tokens"].tolist() == [11, 12, 13]
    assert payload["last_hidden_states"] is None
    assert payload["attention_mask"] is None
    assert tuple(payload["kv_cache_block_id"].shape) == (1, 1, 2)
    kernel_block_id = _derive_kv_cache_kernel_block_id(payload)
    assert tuple(kernel_block_id.shape) == (1, 1, 4)
    assert kernel_block_id.tolist() == [[[14, 15, 16, 17]]]
    return payload


class ModelInputsLoggerReplayTest(unittest.TestCase):
    def test_generated_dump_can_be_loaded_by_torch(self) -> None:
        tool = _runfile_path("rtp_llm/cpp/models/test/model_inputs_logger_dump_tool")
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.check_call([tool, temp_dir, "2"])
            dumps = sorted(Path(temp_dir).glob("model_inputs_r2_s3*.pt"))
            self.assertEqual(1, len(dumps))
            records = _load_model_inputs_dump_records(dumps[0])
            self.assertEqual(2, len(records))
            payload = _validate_model_inputs_dump(dumps[0])
            self.assertEqual(["trace-a", "trace-b"], payload["trace_ids"])

    def test_optional_external_dump_forward_replay(self) -> None:
        replay_path = os.environ.get("MODEL_INPUTS_DUMP_PATH")
        if not replay_path:
            self.skipTest("MODEL_INPUTS_DUMP_PATH is not set")

        path = Path(replay_path)
        dumps = _iter_model_inputs_dumps(path)
        self.assertGreater(len(dumps), 0)
        replayable_count = 0
        for dump in dumps:
            for payload in _load_model_inputs_dump_records(dump):
                if not _is_replayable_payload(payload):
                    continue
                replayable_count += 1
                self.assertTrue(_is_tensor(_derive_kv_cache_kernel_block_id(payload)))
        self.assertGreater(replayable_count, 0)

        replay_cmd = os.environ.get("MODEL_INPUTS_FORWARD_REPLAY_CMD")
        if replay_cmd:
            env = os.environ.copy()
            env["MODEL_INPUTS_DUMP_PATH"] = str(path)
            subprocess.check_call(shlex.split(replay_cmd), env=env)


if __name__ == "__main__":
    unittest.main()
