"""Convert a micrograd GPT checkpoint into a Hugging Face GPT-2 model folder.

Writes config.json, model.safetensors (float32) and GPT-2's tokenizer files,
loadable with transformers.GPT2LMHeadModel.from_pretrained.

Usage:
    python export_hf.py <checkpoint.bin> <output dir>
"""

import argparse
import struct
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

MAGIC = 0x4D47534E
VERSION = 2
FLOAT32 = 0
ARCH_FORMAT = "gpt-arch-1"
GPT2_VOCAB_SIZE = 50257
RECIPE_SWITCHES = ("rotary", "qk_norm", "relu_squared")
SIZE_KEYS = ("n_layer", "n_embd", "n_head", "block_size", "vocab_size", "padded_vocab_size")


class Reader:
    def __init__(self, data, path):
        self.data = data
        self.path = path
        self.pos = 0

    def take(self, count):
        if self.pos + count > len(self.data):
            self.fail("file is truncated")
        chunk = self.data[self.pos : self.pos + count]
        self.pos += count
        return chunk

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u32()).decode()

    def fail(self, message):
        raise ValueError(f"{self.path}: {message}")


def read_checkpoint(path):
    reader = Reader(Path(path).read_bytes(), path)
    if reader.u32() != MAGIC:
        reader.fail("not a micrograd state dict file")
    version = reader.u32()
    if version != VERSION:
        reader.fail(f"unsupported state dict version {version}, expected {VERSION}")

    metadata = {}
    for _ in range(reader.u32()):
        key = reader.string()
        metadata[key] = reader.string()

    tensors = {}
    for _ in range(reader.u32()):
        name = reader.string()
        dtype = reader.u32()
        if dtype != FLOAT32:
            reader.fail(f"parameter {name} has dtype code {dtype}, expected float32")
        shape = [reader.u64() for _ in range(reader.u32())]
        count = 1
        for dim in shape:
            count *= dim
        values = torch.frombuffer(bytearray(reader.take(4 * count)), dtype=torch.float32)
        tensors[name] = values.reshape(shape)
    if reader.pos != len(reader.data):
        reader.fail("unexpected data after the last parameter")
    return metadata, tensors


def gpt2_config(metadata, path):
    if metadata.get("format") != ARCH_FORMAT:
        raise ValueError(f"{path}: architecture format is {metadata.get('format')!r}, expected {ARCH_FORMAT}")
    for key in RECIPE_SWITCHES:
        if metadata.get(key) != "false":
            raise ValueError(f"{path}: {key} is {metadata.get(key)!r}; GPT-2 needs it false")
    sizes = {key: int(metadata[key]) for key in SIZE_KEYS}
    if sizes["vocab_size"] != GPT2_VOCAB_SIZE:
        raise ValueError(f"{path}: vocab_size is {sizes['vocab_size']}, GPT-2's tokenizer has {GPT2_VOCAB_SIZE}")
    return GPT2Config(
        vocab_size=sizes["vocab_size"],
        n_positions=sizes["block_size"],
        n_embd=sizes["n_embd"],
        n_layer=sizes["n_layer"],
        n_head=sizes["n_head"],
        activation_function="gelu_new",
        layer_norm_epsilon=1e-5,
    )


def hf_state_dict(tensors, config):
    def take(name):
        return tensors.pop(name)

    state = {
        "transformer.wte.weight": take("token_embedding.weight")[: config.vocab_size],
        "transformer.wpe.weight": take("position_embedding.weight"),
    }
    for i in range(config.n_layer):
        src = f"h.{i}."
        dst = f"transformer.h.{i}."
        attn = [take(f"{src}attn.{part}.weight") for part in ("query", "key", "value")]
        attn_bias = [take(f"{src}attn.{part}.bias") for part in ("query", "key", "value")]
        state |= {
            f"{dst}ln_1.weight": take(f"{src}ln1.gain"),
            f"{dst}ln_1.bias": take(f"{src}ln1.bias"),
            f"{dst}attn.c_attn.weight": torch.cat(attn, dim=1),
            f"{dst}attn.c_attn.bias": torch.cat(attn_bias),
            f"{dst}attn.c_proj.weight": take(f"{src}attn.out_proj.weight"),
            f"{dst}attn.c_proj.bias": take(f"{src}attn.out_proj.bias"),
            f"{dst}ln_2.weight": take(f"{src}ln2.gain"),
            f"{dst}ln_2.bias": take(f"{src}ln2.bias"),
            f"{dst}mlp.c_fc.weight": take(f"{src}mlp_fc.weight"),
            f"{dst}mlp.c_fc.bias": take(f"{src}mlp_fc.bias"),
            f"{dst}mlp.c_proj.weight": take(f"{src}mlp_proj.weight"),
            f"{dst}mlp.c_proj.bias": take(f"{src}mlp_proj.bias"),
        }
    state["transformer.ln_f.weight"] = take("ln_f.gain")
    state["transformer.ln_f.bias"] = take("ln_f.bias")
    if tensors:
        raise ValueError(f"checkpoint has parameters GPT-2 cannot hold: {sorted(tensors)}")
    return state


def export(checkpoint, output):
    metadata, tensors = read_checkpoint(checkpoint)
    config = gpt2_config(metadata, checkpoint)
    state = hf_state_dict(tensors, config)
    state["lm_head.weight"] = state["transformer.wte.weight"]
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state, strict=True)
    model.save_pretrained(output)
    GPT2TokenizerFast.from_pretrained("gpt2").save_pretrained(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
