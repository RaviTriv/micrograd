import json
import os
import struct
import subprocess

import pytest
import torch
from transformers import GPT2LMHeadModel

from conftest import EVAL_DIR
from export_hf import export
from test_scoring import direct_logprob

GPT_SCORE = os.environ.get("GPT_SCORE", str(EVAL_DIR.parent / "build" / "gpt_score"))
N_LAYER, N_EMBD, N_HEAD, BLOCK, VOCAB, PADDED = 2, 32, 4, 64, 50257, 50304
PAIRS = [
    ([464], [2068, 7586]),
    ([464, 2068, 7586, 21831], [18045, 625, 262, 16931, 3290]),
    ([40, 1842, 284], [4483, 13]),
    ([50256, 15496], [11, 995, 0]),
    ([(i * 37) % VOCAB for i in range(50)], [11, 290, 262]),
]


def architecture(**overrides):
    metadata = {
        "format": "gpt-arch-1",
        "n_layer": str(N_LAYER),
        "n_embd": str(N_EMBD),
        "n_head": str(N_HEAD),
        "block_size": str(BLOCK),
        "vocab_size": str(VOCAB),
        "padded_vocab_size": str(PADDED),
        "rotary": "false",
        "qk_norm": "false",
        "relu_squared": "false",
    }
    return metadata | overrides


def parameters():
    generator = torch.Generator().manual_seed(0)

    def normal(*shape, mean=0.0, std=0.3):
        return mean + std * torch.randn(*shape, generator=generator)

    params = {
        "token_embedding.weight": normal(PADDED, N_EMBD),
        "position_embedding.weight": normal(BLOCK, N_EMBD),
    }
    for i in range(N_LAYER):
        for norm in ("ln1", "ln2"):
            params[f"h.{i}.{norm}.gain"] = normal(N_EMBD, mean=1.0)
            params[f"h.{i}.{norm}.bias"] = normal(N_EMBD)
        for part in ("query", "key", "value", "out_proj"):
            params[f"h.{i}.attn.{part}.weight"] = normal(N_EMBD, N_EMBD)
            params[f"h.{i}.attn.{part}.bias"] = normal(N_EMBD)
        params[f"h.{i}.mlp_fc.weight"] = normal(N_EMBD, 4 * N_EMBD)
        params[f"h.{i}.mlp_fc.bias"] = normal(4 * N_EMBD)
        params[f"h.{i}.mlp_proj.weight"] = normal(4 * N_EMBD, N_EMBD)
        params[f"h.{i}.mlp_proj.bias"] = normal(N_EMBD)
    params["ln_f.gain"] = normal(N_EMBD, mean=1.0)
    params["ln_f.bias"] = normal(N_EMBD)
    return params


def write_checkpoint(path, metadata, params, version=2):
    def string(text):
        data = text.encode()
        return struct.pack("<I", len(data)) + data

    out = struct.pack("<II", 0x4D47534E, version) + struct.pack("<I", len(metadata))
    for key, value in metadata.items():
        out += string(key) + string(value)
    out += struct.pack("<I", len(params))
    for name, tensor in params.items():
        out += string(name) + struct.pack("<II", 0, tensor.dim())
        out += struct.pack(f"<{tensor.dim()}Q", *tensor.shape)
        out += tensor.contiguous().numpy().astype("<f4").tobytes()
    path.write_bytes(out)
    return path


def gpt_score(checkpoint):
    if not os.access(GPT_SCORE, os.X_OK):
        pytest.skip(f"no gpt_score at {GPT_SCORE}; build it or set GPT_SCORE")
    requests = "".join(json.dumps({"context": c, "continuation": t}) + "\n" for c, t in PAIRS)
    result = subprocess.run(
        [GPT_SCORE, "--checkpoint", str(checkpoint), "--device", "cpu"],
        input=requests,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return [json.loads(line)["logprob"] for line in result.stdout.splitlines()]


def test_exported_model_matches_gpt_score(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "tiny.bin", architecture(), parameters())
    export(checkpoint, tmp_path / "hf")
    model = GPT2LMHeadModel.from_pretrained(tmp_path / "hf", dtype=torch.float32).eval()
    assert model.config.vocab_size == VOCAB
    exported = [direct_logprob(model, c + t, len(t)) for c, t in PAIRS]
    assert exported == pytest.approx(gpt_score(checkpoint), abs=1e-4)


@pytest.mark.parametrize("switch", ["rotary", "qk_norm", "relu_squared"])
def test_refuses_recipe_switches(tmp_path, switch):
    checkpoint = write_checkpoint(tmp_path / "tiny.bin", architecture(**{switch: "true"}), parameters())
    with pytest.raises(ValueError, match=switch):
        export(checkpoint, tmp_path / "hf")


def test_refuses_unknown_format_version(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "tiny.bin", architecture(), parameters(), version=3)
    with pytest.raises(ValueError, match="version 3"):
        export(checkpoint, tmp_path / "hf")


def test_refuses_unknown_architecture_format(tmp_path):
    checkpoint = write_checkpoint(tmp_path / "tiny.bin", architecture(format="gpt-arch-2"), parameters())
    with pytest.raises(ValueError, match="gpt-arch-2"):
        export(checkpoint, tmp_path / "hf")
