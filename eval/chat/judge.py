"""Part B judging: 1 to 5 grades against the reference, and pairwise preference.

A judge is any object with an `id` string and a `complete(prompt) -> str` method.
`FakeJudge` is deterministic and used by the tests. `LLMJudge` is the slot for the
real judge; it raises until the owner picks a model and who pays for it.
"""

import json
import random
import re

GRADE_PROMPT = """You are grading an answer to a question. Compare it with the reference answer.

<question>
{question}
</question>

<reference>
{reference}
</reference>

<answer>
{answer}
</answer>

Score two things from 1 (worst) to 5 (best):
- correctness: does the answer agree with the reference and contain no false claims? If the question is nonsense or cannot be answered, a correct answer says so.
- helpfulness: is the answer on topic, direct, clear, and free of rambling or repetition?

Reply with only a JSON object: {{"correctness": <1-5>, "helpfulness": <1-5>, "reason": "<one sentence>"}}"""

PAIRWISE_PROMPT = """You are comparing two answers to the same question. Use the reference answer to judge correctness.

<question>
{question}
</question>

<reference>
{reference}
</reference>

<answer_1>
{answer_1}
</answer_1>

<answer_2>
{answer_2}
</answer_2>

Which answer is better overall: more correct first, then more helpful? Reply with only a JSON object: {{"winner": "1", "2" or "tie", "reason": "<one sentence>"}}"""

PASS_SCORE = 4
WORD = re.compile(r"[a-z0-9]+")


class JudgeNotConfigured(RuntimeError):
    pass


class LLMJudge:
    def __init__(self, model):
        self.id = f"llm:{model}"

    def complete(self, prompt):
        raise JudgeNotConfigured(
            f"{self.id}: no judge backend is configured. The judge model and who pays for its API calls are "
            "an open owner decision (docs/plans/active/2026-11-13-taby-mm-useful.md)."
        )


def tagged(prompt, tag):
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.S)
    return match.group(1) if match else ""


def overlap(answer, reference):
    words = set(WORD.findall(reference.lower()))
    return len(words & set(WORD.findall(answer.lower()))) / max(1, len(words))


class FakeJudge:
    id = "fake:word-overlap"

    def complete(self, prompt):
        reference = tagged(prompt, "reference")
        if "<answer_1>" in prompt:
            first = overlap(tagged(prompt, "answer_1"), reference)
            second = overlap(tagged(prompt, "answer_2"), reference)
            winner = "tie" if first == second else ("1" if first > second else "2")
            return json.dumps({"winner": winner, "reason": "word overlap with the reference"})
        answer = tagged(prompt, "answer")
        score = 1 + round(4 * overlap(answer, reference))
        helpful = score if answer.strip() else 1
        return f"Grade: {json.dumps({'correctness': score, 'helpfulness': helpful, 'reason': 'word overlap'})}"


def make_judge(spec):
    if spec == "fake":
        return FakeJudge()
    if spec.startswith("llm:"):
        return LLMJudge(spec.removeprefix("llm:"))
    raise ValueError(f"unknown judge {spec!r}: use 'fake' or 'llm:<model id>'")


def json_object(text):
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    reply, _ = json.JSONDecoder().raw_decode(text, start)
    if not isinstance(reply, dict):
        raise ValueError(f"judge reply is not a JSON object: {text[:200]!r}")
    return reply


def parse_grade(text):
    reply = json_object(text)
    grade = {}
    for key in ("correctness", "helpfulness"):
        value = reply.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
            raise ValueError(f"{key} is not a whole number: {value!r}")
        if not 1 <= value <= 5:
            raise ValueError(f"{key} out of range 1 to 5: {value!r}")
        grade[key] = int(value)
    grade["reason"] = str(reply.get("reason", ""))
    return grade


def parse_winner(text):
    winner = str(json_object(text).get("winner", "")).strip().lower()
    if winner not in ("1", "2", "tie"):
        raise ValueError(f"winner must be 1, 2 or tie: {winner!r}")
    return winner


def ask(judge, prompt, parse):
    raw = None
    try:
        raw = judge.complete(prompt)
        return {**parse(raw), "raw": raw}
    except JudgeNotConfigured:
        raise
    except Exception as error:
        return {"raw": raw, "error": f"{type(error).__name__}: {error}"}


def passes(grade):
    return grade["correctness"] >= PASS_SCORE and grade["helpfulness"] >= PASS_SCORE


def headline(grades):
    return round(100 * sum(passes(g) for g in grades) / len(grades), 1) if grades else 0.0


def grade_rows(judge, rows, questions, done=(), sink=None):
    graded = []
    for row in rows:
        if row["id"] in done:
            continue
        question = questions[row["id"]]
        prompt = GRADE_PROMPT.format(question=question["question"], reference=question["reference"],
                                     answer=row["answer"])
        result = {"id": row["id"], "group": question["group"], **ask(judge, prompt, parse_grade)}
        if sink:
            sink(result)
        graded.append(result)
    return graded


def grade_summary(graded):
    valid = [g for g in graded if "error" not in g]
    groups = sorted({g["group"] for g in valid})
    return {
        "headline_pct_both_at_least_4": headline(valid),
        "mean_correctness": round(sum(g["correctness"] for g in valid) / len(valid), 3) if valid else None,
        "mean_helpfulness": round(sum(g["helpfulness"] for g in valid) / len(valid), 3) if valid else None,
        "by_group": {name: headline([g for g in valid if g["group"] == name]) for name in groups},
        "count": len(valid),
        "errors": len(graded) - len(valid),
    }


def pairwise_rows(judge, rows_a, rows_b, questions, seed, done=(), sink=None):
    by_id = {row["id"]: row for row in rows_b}
    rng = random.Random(seed)
    results = []
    for a in rows_a:
        order = "ba" if rng.random() < 0.5 else "ab"
        if a["id"] in done:
            continue
        b = by_id[a["id"]]
        question = questions[a["id"]]
        first, second = (b, a) if order == "ba" else (a, b)
        prompt = PAIRWISE_PROMPT.format(question=question["question"], reference=question["reference"],
                                        answer_1=first["answer"], answer_2=second["answer"])
        reply = ask(judge, prompt, lambda raw: {"pick": parse_winner(raw)})
        result = {"id": a["id"], "group": question["group"], "order": order, "raw": reply["raw"]}
        if "error" in reply:
            result["error"] = reply["error"]
        else:
            result["winner"] = "tie" if reply["pick"] == "tie" else order[int(reply["pick"]) - 1]
        if sink:
            sink(result)
        results.append(result)
    return results


def pairwise_summary(results):
    valid = [r for r in results if "error" not in r]
    wins = {side: sum(r["winner"] == side for r in valid) for side in ("a", "b", "tie")}
    return {"a_wins": wins["a"], "b_wins": wins["b"], "ties": wins["tie"],
            "a_wins_more_than_loses": wins["a"] > wins["b"], "errors": len(results) - len(valid)}
