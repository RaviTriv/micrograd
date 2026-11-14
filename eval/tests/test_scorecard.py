import json

import huggingface_hub
import pytest

import scorecard
from chat import answer, judge
from evaluate import sha256
from test_chat_answer import constant_model

QUESTION = {"group": "fact", "question": "What is the capital of France?", "reference": "The capital of France is Paris."}


def question_set(path, count):
    answer.write_jsonl(path, [{"id": f"q{i}", **QUESTION} for i in range(count)])
    return path


def generations(path, model, answers, questions=None):
    questions = questions or question_set(path.parent / "set.jsonl", len(answers))
    rows = [{"id": f"q{i}", **QUESTION, "model": model, "answer": text} for i, text in enumerate(answers)]
    answer.write_jsonl(path, rows)
    meta = {"question_set": str(questions), "question_set_sha256": sha256(questions)}
    path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return path


def test_summary_prefers_acc_norm_and_falls_back_to_acc():
    task = {"acc,none": 0.3, "acc_stderr,none": 0.01, "acc_norm,none": 0.35, "acc_norm_stderr,none": 0.02}
    plain = {"acc,none": 0.5, "acc_stderr,none": 0.03}
    results = {name: (plain if name in ("winogrande", "mmlu") else task) for name in scorecard.KNOWLEDGE_TASKS}
    summary = scorecard.summarize(results)
    assert summary["hellaswag"] == {"metric": "acc_norm", "value": 0.35, "stderr": 0.02, "acc": 0.3, "acc_stderr": 0.01}
    assert summary["mmlu"] == {"metric": "acc", "value": 0.5, "stderr": 0.03, "acc": 0.5, "acc_stderr": 0.03}


def test_hub_ids_name_themselves_and_local_paths_need_a_name(tmp_path):
    assert scorecard.result_name("HuggingFaceTB/SmolLM2-135M-Instruct", None) == "smollm2-135m-instruct"
    assert scorecard.result_name(str(tmp_path / "hf-final"), "taby-mm") == "taby-mm"
    (tmp_path / "hf-final").mkdir()
    with pytest.raises(SystemExit, match="--name"):
        scorecard.result_name(str(tmp_path / "hf-final"), None)


def test_hub_download_is_limited_to_model_and_tokenizer_files(tmp_path, monkeypatch):
    calls = []

    def download(repo_id, **options):
        calls.append(options)
        return str(tmp_path / "snapshots" / "abc123")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    path, source = scorecard.resolve("org/model")
    assert source == {"hf_id": "org/model", "hf_revision": "abc123"}
    patterns = calls[0]["allow_patterns"]
    assert "*.safetensors" in patterns and "config.json" in patterns and "generation_config.json" in patterns
    assert not any(p in patterns for p in ("*", "*.bin", "*.pt", "*.gguf", "*.onnx"))


def test_generate_writes_answers_and_provenance(tmp_path, monkeypatch):
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
    model_dir = tmp_path / "tiny"
    constant_model(tokenizer, tokenizer.eos_token, model_dir)
    monkeypatch.setattr(scorecard, "CHAT_RESULTS", tmp_path / "chat")
    scorecard.main(["generate", str(model_dir), "--name", "tiny", "--set", "dev", "--device", "cpu"])
    rows = answer.read_jsonl(tmp_path / "chat" / "tiny-dev.jsonl")
    meta = json.loads((tmp_path / "chat" / "tiny-dev.meta.json").read_text())
    dev = scorecard.QUESTION_SETS["dev"]
    assert [r["id"] for r in rows] == [q["id"] for q in answer.read_jsonl(dev)]
    assert all(r["stop"] == "eos" and r["kind"] == "base" and r["model"] == "tiny" for r in rows)
    assert (meta["kind"], meta["device"], meta["question_set"]) == ("base", "cpu", "eval/chat/dev.jsonl")
    assert meta["question_set_sha256"] == sha256(dev)
    assert meta["weights_sha256"]["model.safetensors"] == sha256(model_dir / "model.safetensors")
    assert meta["settings"] == answer.SETTINGS and meta["base_template"] == answer.BASE_TEMPLATE
    with pytest.raises(SystemExit, match="--overwrite"):
        scorecard.main(["generate", str(model_dir), "--name", "tiny", "--set", "dev", "--device", "cpu"])


def test_judge_command_pins_judge_prompt_and_provenance(tmp_path):
    source = generations(tmp_path / "m-dev.jsonl", "m", ["The capital of France is Paris.", "No idea."])
    scorecard.main(["judge", str(source), "--judge", "fake"])
    record = json.loads((tmp_path / "m-dev-judge-fake-word-overlap.json").read_text())
    assert record["judge_id"] == judge.FakeJudge.id
    assert record["judge_prompt"] == judge.GRADE_PROMPT
    assert isinstance(record["repo_dirty"], bool)
    assert (record["stale"], record["question_set_sha256"]) == (False, sha256(tmp_path / "set.jsonl"))
    assert record["summary"]["headline_pct_both_at_least_4"] == 50.0
    assert [g["id"] for g in record["grades"]] == ["q0", "q1"]


def test_judge_refuses_a_changed_question_set_unless_allowed(tmp_path):
    source = generations(tmp_path / "m-dev.jsonl", "m", ["Paris."])
    question_set(tmp_path / "set.jsonl", 2)
    with pytest.raises(SystemExit, match="--allow-stale"):
        scorecard.main(["judge", str(source), "--judge", "fake"])
    scorecard.main(["judge", str(source), "--judge", "fake", "--allow-stale"])
    record = json.loads((tmp_path / "m-dev-judge-fake-word-overlap.json").read_text())
    assert record["stale"] is True


def test_judge_rows_are_written_as_they_come_and_resumed(tmp_path, monkeypatch):
    source = generations(tmp_path / "m-dev.jsonl", "m", ["Paris."] * 3)
    calls = []

    class Flaky:
        id = "flaky"

        def complete(self, prompt):
            calls.append(prompt)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return '{"correctness": 5, "helpfulness": 5}'

    monkeypatch.setattr(judge, "make_judge", lambda spec: Flaky())
    with pytest.raises(KeyboardInterrupt):
        scorecard.main(["judge", str(source), "--judge", "x"])
    progress = tmp_path / "m-dev-judge-flaky.rows.jsonl"
    assert [r["id"] for r in answer.read_jsonl(progress)] == ["q0"]
    scorecard.main(["judge", str(source), "--judge", "x"])
    record = json.loads((tmp_path / "m-dev-judge-flaky.json").read_text())
    assert len(calls) == 4
    assert [g["id"] for g in record["grades"]] == ["q0", "q1", "q2"]


def test_new_answers_are_graded_again_not_resumed(tmp_path):
    output = tmp_path / "m-dev-judge-fake-word-overlap.json"
    source = generations(tmp_path / "m-dev.jsonl", "m", ["Bananas.", "Bananas."])
    scorecard.main(["judge", str(source), "--judge", "fake"])
    assert json.loads(output.read_text())["summary"]["headline_pct_both_at_least_4"] == 0.0
    generations(source, "m", ["The capital of France is Paris."] * 2, questions=tmp_path / "set.jsonl")
    scorecard.main(["judge", str(source), "--judge", "fake"])
    record = json.loads(output.read_text())
    assert record["summary"]["headline_pct_both_at_least_4"] == 100.0
    assert all(g["answers_sha256"] == sha256(source) for g in record["grades"])


def test_pairwise_command_records_the_seed_and_counts(tmp_path):
    a = generations(tmp_path / "new-dev.jsonl", "new", ["The capital of France is Paris."] * 3)
    b = generations(tmp_path / "old-dev.jsonl", "old", ["Bananas."] * 3)
    scorecard.main(["pairwise", str(a), str(b), "--judge", "fake", "--seed", "7"])
    record = json.loads((tmp_path / "new-dev-vs-old-dev-pairwise-fake-word-overlap.json").read_text())
    assert (record["seed"], record["judge_prompt"]) == (7, judge.PAIRWISE_PROMPT)
    assert record["summary"]["a_wins"] == 3
    assert len(record["results"]) == 3


def test_pairwise_refuses_answers_to_different_question_sets(tmp_path):
    a = generations(tmp_path / "new-dev.jsonl", "new", ["Paris."] * 2)
    other = question_set(tmp_path / "other.jsonl", 3)
    b = generations(tmp_path / "old-dev.jsonl", "old", ["Paris."] * 2, questions=other)
    with pytest.raises(SystemExit, match="different question sets"):
        scorecard.main(["pairwise", str(a), str(b), "--judge", "fake"])


def test_judge_command_refuses_an_unconfigured_judge(tmp_path):
    source = generations(tmp_path / "m-dev.jsonl", "m", ["Paris."])
    with pytest.raises(judge.JudgeNotConfigured):
        scorecard.main(["judge", str(source), "--judge", "llm:any"])
    assert not list(tmp_path.glob("*judge*"))
