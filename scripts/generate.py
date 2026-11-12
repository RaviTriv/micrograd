import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "eval" / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

EXPORT_DIR = ROOT / "thunder-results" / "long-a100x8-20261112" / "hf-final"


def continuation(source, prompt, device, max_new_tokens, temperature, top_k, seed):
    tokenizer = GPT2TokenizerFast.from_pretrained(source)
    model = GPT2LMHeadModel.from_pretrained(source, dtype=torch.float32).to(device).eval()
    torch.manual_seed(seed)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True,
                             max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:])


def main():
    parser = argparse.ArgumentParser(description="Generate text from the exported model for one prompt.")
    parser.add_argument("--export-dir", type=Path, default=EXPORT_DIR)
    parser.add_argument("--prompt", default="The cat and dog went on a walk\n")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=950709)
    args = parser.parse_args()
    text = continuation(str(args.export_dir), args.prompt, args.device, args.max_new_tokens,
                        args.temperature, args.top_k, args.seed)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
