"""Part B generation: one fixed prompt per model type, greedy decoding, fixed stops."""

import json

BASE_TEMPLATE = "Question: {question}\nAnswer:"
BLANK_LINE = "\n\n"
MAX_NEW_TOKENS = 200
SETTINGS = {"decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "base_stop": "blank line or end of text",
            "chat_stop": "end of turn token"}


def kind_of(tokenizer):
    return "chat" if getattr(tokenizer, "chat_template", None) else "base"


def prompt_for(tokenizer, question):
    if kind_of(tokenizer) == "chat":
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return BASE_TEMPLATE.format(question=question)


def finish(text, kind):
    if kind == "base" and BLANK_LINE in text:
        return text.split(BLANK_LINE, 1)[0].strip(), "blank_line"
    return text.strip(), None


def stop_ids(model, tokenizer):
    configured = model.generation_config.eos_token_id
    ids = set(configured if isinstance(configured, (list, tuple)) else [configured])
    return (ids | {tokenizer.eos_token_id}) - {None}


def answer(generator, question):
    tokenizer = generator.tokenizer
    kind = kind_of(tokenizer)
    prompt = prompt_for(tokenizer, question)
    stops = [BLANK_LINE] if kind == "base" else None
    ids = generator.generate_ids(prompt, MAX_NEW_TOKENS, 0, 0, 0, stop_strings=stops)
    text, stop = finish(tokenizer.decode(ids, skip_special_tokens=True), kind)
    if stop is None:
        stop = "eos" if ids and ids[-1] in stop_ids(generator.model, tokenizer) else "max_new_tokens"
    return {"kind": kind, "prompt": prompt, "answer": text, "new_tokens": len(ids), "stop": stop}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
