from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from chat import answer
from serve import Generator

CHAT_DIR = Path(answer.__file__).parent

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    return GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)


def constant_model(tokenizer, text, directory, chat=False, stops=None):
    token = tokenizer.convert_tokens_to_ids(text) if text == tokenizer.eos_token else tokenizer(text).input_ids[0]
    config = GPT2Config(n_layer=1, n_head=1, n_embd=8, n_positions=512, tie_word_embeddings=False)
    model = GPT2LMHeadModel(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transformer.ln_f.bias[0] = 1.0
        model.lm_head.weight[token, 0] = 1.0
    if stops:
        model.generation_config.eos_token_id = [tokenizer(t).input_ids[0] for t in stops]
    model.save_pretrained(directory)
    saved = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    if chat:
        saved.chat_template = CHAT_TEMPLATE
    saved.save_pretrained(directory)
    return Generator(str(directory), "cpu")


def test_base_model_gets_plain_question_answer_prompt(tokenizer):
    assert answer.kind_of(tokenizer) == "base"
    assert answer.prompt_for(tokenizer, "What is a cat?") == "Question: What is a cat?\nAnswer:"


def test_chat_model_gets_its_chat_template_with_generation_prompt(tokenizer):
    chat = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    chat.chat_template = CHAT_TEMPLATE
    assert answer.kind_of(chat) == "chat"
    assert answer.prompt_for(chat, "What is a cat?") == "<|user|>What is a cat?\n<|assistant|>"


def test_base_answer_is_cut_at_the_first_blank_line():
    assert answer.finish(" Paris.\n\nQuestion: next", "base") == ("Paris.", "blank_line")
    assert answer.finish(" Paris.\nIt is big.", "base") == ("Paris.\nIt is big.", None)


def test_chat_answer_is_not_cut_at_a_blank_line():
    assert answer.finish("Step one.\n\nStep two. ", "chat") == ("Step one.\n\nStep two.", None)


def test_base_generation_stops_at_a_blank_line(tokenizer, tmp_path):
    reply = answer.answer(constant_model(tokenizer, "\n\n", tmp_path), "What is a cat?")
    assert reply == {"kind": "base", "prompt": "Question: What is a cat?\nAnswer:", "answer": "",
                     "new_tokens": 1, "stop": "blank_line"}


def test_generation_stops_at_end_of_text(tokenizer, tmp_path):
    reply = answer.answer(constant_model(tokenizer, tokenizer.eos_token, tmp_path), "What is a cat?")
    assert (reply["answer"], reply["new_tokens"], reply["stop"]) == ("", 1, "eos")


def test_generation_without_a_stop_runs_to_the_token_limit(tokenizer, tmp_path):
    reply = answer.answer(constant_model(tokenizer, " cat", tmp_path), "What is a cat?")
    assert reply["new_tokens"] == answer.MAX_NEW_TOKENS
    assert reply["stop"] == "max_new_tokens"
    assert reply["answer"] == " ".join(["cat"] * answer.MAX_NEW_TOKENS)


def test_chat_generation_ignores_blank_lines(tokenizer, tmp_path):
    reply = answer.answer(constant_model(tokenizer, "\n\n", tmp_path, chat=True), "What is a cat?")
    assert reply["kind"] == "chat"
    assert reply["prompt"] == "<|user|>What is a cat?\n<|assistant|>"
    assert (reply["new_tokens"], reply["stop"]) == (answer.MAX_NEW_TOKENS, "max_new_tokens")


def test_question_sets_are_balanced_and_disjoint():
    held_out = answer.read_jsonl(CHAT_DIR / "questions.jsonl")
    dev = answer.read_jsonl(CHAT_DIR / "dev.jsonl")
    for rows, per_group in ((held_out, 20), (dev, 6)):
        groups = [row["group"] for row in rows]
        assert {g: groups.count(g) for g in set(groups)} == dict.fromkeys(
            ("definition", "fact", "howto", "reasoning", "unanswerable"), per_group)
        assert all(row["reference"].strip() and row["question"].strip() for row in rows)
        assert len({row["id"] for row in rows}) == len(rows)
    assert not {r["question"].lower() for r in held_out} & {r["question"].lower() for r in dev}



def test_a_configured_end_token_other_than_the_tokenizer_eos_is_labelled_eos(tokenizer, tmp_path):
    reply = answer.answer(constant_model(tokenizer, " end", tmp_path, stops=[" end", " stop"]), "What is a cat?")
    assert (reply["answer"], reply["new_tokens"], reply["stop"]) == ("end", 1, "eos")
