"""Run the taby-mm scorecard: Part A (multiple choice) and Part B (question answering).

Usage:
    python scorecard.py knowledge <model> --device mps
    python scorecard.py generate <model> --set questions --device mps
    python scorecard.py judge results/chat/<model>-questions.jsonl --judge <id>
    python scorecard.py pairwise <a.jsonl> <b.jsonl> --judge <id> --seed 0

<model> is an exported folder or a Hugging Face id. A folder needs --name; an id
defaults to its last part, lower-cased. Existing results are kept unless --overwrite.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from chat import answer as chat_answer
from chat import judge as chat_judge
from evaluate import REPO, git, portable, recorded_path, sha256

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
CHAT_RESULTS = RESULTS / "chat"
QUESTION_SETS = {"questions": EVAL_DIR / "chat" / "questions.jsonl", "dev": EVAL_DIR / "chat" / "dev.jsonl"}
KNOWLEDGE_TASKS = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu"]
BACKEND = "lm-eval hf, float32, batch size 1, 0-shot, no chat template"
HF_FILES = ["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json", "vocab.json",
            "merges.txt", "*.safetensors"]


def result_name(model, name):
    if name:
        return name
    if Path(model).exists():
        raise SystemExit(f"{model} is a local path: pass --name to name its results")
    return model.split("/")[-1].lower()


def fresh(path, overwrite):
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists: pass --overwrite to replace it")
    return path


def resolve(model):
    if Path(model).is_dir():
        path = Path(model).resolve()
        return path, {"model_path": recorded_path(path)}
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model, allow_patterns=HF_FILES))
    return path, {"hf_id": model, "hf_revision": path.name}


def provenance(model, device):
    path, source = resolve(model)
    weights = sorted(path.glob("*.safetensors"))
    record = {
        **source,
        "weights_sha256": {w.name: sha256(w) for w in weights},
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain")),
        "lm_eval_version": version("lm_eval"),
        "transformers_version": version("transformers"),
        "torch_version": version("torch"),
        "device": device,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return path, record


def summarize(results):
    summary = {}
    for task in KNOWLEDGE_TASKS:
        metrics = results[task]
        name = "acc_norm" if "acc_norm,none" in metrics else "acc"
        summary[task] = {
            "metric": name,
            "value": metrics[f"{name},none"],
            "stderr": metrics[f"{name}_stderr,none"],
            "acc": metrics["acc,none"],
            "acc_stderr": metrics["acc_stderr,none"],
        }
    return summary


def knowledge(args):
    import lm_eval

    name = result_name(args.model, args.name)
    output = fresh(Path(args.output or RESULTS / f"{name}-scorecard-a.json"), args.overwrite)
    path, record = provenance(args.model, args.device)
    record["backend"] = BACKEND
    record["tasks"] = KNOWLEDGE_TASKS
    record["limit"] = args.limit
    start = time.time()
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained": str(path), "dtype": "float32"},
        tasks=KNOWLEDGE_TASKS,
        device=args.device,
        batch_size=1,
        limit=args.limit,
        log_samples=False,
    )
    record["wall_time_seconds"] = round(time.time() - start, 1)
    record["summary"] = summarize(results["results"])
    record["harness"] = portable(json.loads(json.dumps(results, default=str)))
    write_json(output, record)
    for task, row in record["summary"].items():
        print(f"{task:14} {row['metric']:8} {row['value']:.4f} +- {row['stderr']:.4f}")
    print(f"wrote {output}")


def generate(args):
    from serve import Generator

    name = result_name(args.model, args.name)
    output = fresh(CHAT_RESULTS / f"{name}-{args.set}.jsonl", args.overwrite)
    path, record = provenance(args.model, args.device)
    questions_path = QUESTION_SETS[args.set]
    generator = Generator(str(path), args.device)
    questions = chat_answer.read_jsonl(questions_path)
    start = time.time()
    rows = []
    for question in questions:
        reply = chat_answer.answer(generator, question["question"])
        rows.append({**question, "model": name, **reply})
        print(f"{question['id']} {reply['stop']:14} {reply['answer'][:70]!r}", flush=True)
    record |= {
        "model": name,
        "kind": rows[0]["kind"] if rows else chat_answer.kind_of(generator.tokenizer),
        "question_set": recorded_path(questions_path),
        "question_set_sha256": sha256(questions_path),
        "settings": chat_answer.SETTINGS,
        "base_template": chat_answer.BASE_TEMPLATE,
        "wall_time_seconds": round(time.time() - start, 1),
    }
    chat_answer.write_jsonl(output, rows)
    write_json(output.with_suffix(".meta.json"), record)
    print(f"wrote {output}")


def slug(text):
    return "".join(c if c.isalnum() else "-" for c in text.lower())


def load_generations(path, allow_stale):
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} is missing: cannot tell which question set {path} answers")
    meta = json.loads(meta_path.read_text())
    questions_path = Path(meta["question_set"])
    questions_path = questions_path if questions_path.is_absolute() else REPO / questions_path
    current = sha256(questions_path)
    stale = current != meta["question_set_sha256"]
    if stale and not allow_stale:
        raise SystemExit(f"{path} was generated from {meta['question_set']} at sha256 "
                         f"{meta['question_set_sha256'][:12]}, which is now {current[:12]}: "
                         "regenerate, or pass --allow-stale to grade against the current references")
    questions = {q["id"]: q for q in chat_answer.read_jsonl(questions_path)}
    rows = chat_answer.read_jsonl(path)
    missing = [row["id"] for row in rows if row["id"] not in questions]
    if missing:
        raise SystemExit(f"{path}: ids not in {meta['question_set']}: {', '.join(missing[:10])}")
    stamp = {"question_set": meta["question_set"], "question_set_sha256": current,
             "generated_from_sha256": meta["question_set_sha256"], "stale": stale}
    return rows, questions, stamp


def judge_record(judge, prompt, files, stamp):
    return {
        "judge_id": judge.id,
        "judge_prompt": prompt,
        "inputs": {recorded_path(f): sha256(f) for f in files},
        **stamp,
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain")),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def progress_file(output, key):
    path = output.with_suffix(".rows.jsonl")
    latest = {}
    if path.is_file():
        for row in chat_answer.read_jsonl(path):
            if any(row.get(k) != v for k, v in key.items()):
                continue
            if "error" in row and "error" not in latest.get(row["id"], {"error": True}):
                continue
            latest[row["id"]] = row

    def append(row):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps({**key, **row}, ensure_ascii=False) + "\n")
        latest[row["id"]] = {**key, **row}

    done = {qid for qid, row in latest.items() if "error" not in row}
    return latest, done, append


def judge(args):
    grader = chat_judge.make_judge(args.judge)
    source = Path(args.generations)
    rows, questions, stamp = load_generations(source, args.allow_stale)
    record = judge_record(grader, chat_judge.GRADE_PROMPT, [source], stamp)
    output = Path(args.output or source.with_name(f"{source.stem}-judge-{slug(grader.id)}.json"))
    key = {"judge_id": grader.id, "question_set_sha256": stamp["question_set_sha256"], "answers_sha256": sha256(source)}
    latest, done, append = progress_file(output, key)
    chat_judge.grade_rows(grader, rows, questions, done=done, sink=append)
    graded = [latest[row["id"]] for row in rows if row["id"] in latest]
    record |= {"summary": chat_judge.grade_summary(graded), "grades": graded}
    write_json(output, record)
    print(json.dumps(record["summary"], indent=2))
    print(f"wrote {output}")


def pairwise(args):
    grader = chat_judge.make_judge(args.judge)
    a, b = Path(args.a), Path(args.b)
    rows_a, questions, stamp = load_generations(a, args.allow_stale)
    rows_b, _, stamp_b = load_generations(b, args.allow_stale)
    if (stamp["question_set"], stamp["generated_from_sha256"]) != (stamp_b["question_set"], stamp_b["generated_from_sha256"]):
        raise SystemExit(f"{a} and {b} answer different question sets: {stamp['question_set']} at "
                         f"{stamp['generated_from_sha256'][:12]} against {stamp_b['question_set']} at "
                         f"{stamp_b['generated_from_sha256'][:12]}")
    record = judge_record(grader, chat_judge.PAIRWISE_PROMPT, [a, b], stamp)
    output = Path(args.output or a.with_name(f"{a.stem}-vs-{b.stem}-pairwise-{slug(grader.id)}.json"))
    key = {"judge_id": grader.id, "question_set_sha256": stamp["question_set_sha256"], "seed": args.seed,
           "a_sha256": sha256(a), "b_sha256": sha256(b)}
    latest, done, append = progress_file(output, key)
    chat_judge.pairwise_rows(grader, rows_a, rows_b, questions, args.seed, done=done, sink=append)
    results = [latest[row["id"]] for row in rows_a if row["id"] in latest]
    record |= {"a": a.stem, "b": b.stem, "seed": args.seed,
               "summary": chat_judge.pairwise_summary(results), "results": results}
    write_json(output, record)
    print(json.dumps(record["summary"], indent=2))
    print(f"wrote {output}")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    part_a = commands.add_parser("knowledge", help="Part A: multiple choice through lm-eval")
    part_a.add_argument("model")
    part_a.add_argument("--name")
    part_a.add_argument("--device", default="mps")
    part_a.add_argument("--limit", type=int)
    part_a.add_argument("--output")
    part_a.add_argument("--overwrite", action="store_true")
    part_a.set_defaults(run=knowledge)

    part_b = commands.add_parser("generate", help="Part B: answer a question set with fixed settings")
    part_b.add_argument("model")
    part_b.add_argument("--name")
    part_b.add_argument("--set", choices=sorted(QUESTION_SETS), default="questions")
    part_b.add_argument("--device", default="mps")
    part_b.add_argument("--overwrite", action="store_true")
    part_b.set_defaults(run=generate)

    grade = commands.add_parser("judge", help="Part B: grade saved answers 1 to 5 against the references")
    grade.add_argument("generations")
    grade.add_argument("--judge", required=True, help="'fake' or 'llm:<model id>'")
    grade.add_argument("--output")
    grade.add_argument("--allow-stale", action="store_true", help="grade although the question set changed since generation")
    grade.set_defaults(run=judge)

    versus = commands.add_parser("pairwise", help="Part B: judge picks the better of two models per question")
    versus.add_argument("a")
    versus.add_argument("b")
    versus.add_argument("--judge", required=True, help="'fake' or 'llm:<model id>'")
    versus.add_argument("--seed", type=int, default=0)
    versus.add_argument("--output")
    versus.add_argument("--allow-stale", action="store_true", help="judge although the question set changed since generation")
    versus.set_defaults(run=pairwise)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
