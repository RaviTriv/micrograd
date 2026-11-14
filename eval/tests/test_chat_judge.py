import json

import pytest

from chat import judge

REFERENCE = "The capital of France is Paris."


def row(qid, answer, group="fact", reference=REFERENCE):
    return {"id": qid, "group": group, "question": "What is the capital of France?",
            "reference": reference, "answer": answer}


def questions_for(rows):
    return {r["id"]: {k: r[k] for k in ("id", "group", "question", "reference")} for r in rows}


def graded(rows, grader=None, **options):
    return judge.grade_rows(grader or judge.FakeJudge(), rows, questions_for(rows), **options)


class Scripted:
    id = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def grade(correctness, helpfulness):
    return {"correctness": correctness, "helpfulness": helpfulness}


def test_grade_reply_is_parsed_from_surrounding_text():
    text = 'Sure.\n{"correctness": 4, "helpfulness": 5, "reason": "right"}\nDone.'
    assert judge.parse_grade(text) == {"correctness": 4, "helpfulness": 5, "reason": "right"}


def test_only_the_first_json_object_is_read():
    text = '{"correctness": 4, "helpfulness": 5, "reason": "uses {braces}"} and later {"correctness": 1}'
    assert judge.parse_grade(text) == {"correctness": 4, "helpfulness": 5, "reason": "uses {braces}"}


@pytest.mark.parametrize("text", [
    "no json here",
    '{"correctness": 6, "helpfulness": 3}',
    '{"correctness": 0, "helpfulness": 3}',
    '{"correctness": 3.5, "helpfulness": 3}',
    '{"correctness": "4", "helpfulness": 3}',
    '{"correctness": true, "helpfulness": 3}',
    '{"helpfulness": 3}',
])
def test_bad_grade_replies_are_rejected(text):
    with pytest.raises(ValueError):
        judge.parse_grade(text)


def test_winner_reply_is_parsed():
    assert judge.parse_winner('{"winner": "2", "reason": "x"}') == "2"
    assert judge.parse_winner('{"winner": "Tie"}') == "tie"
    with pytest.raises(ValueError):
        judge.parse_winner('{"winner": "A"}')


def test_headline_counts_answers_with_both_scores_at_least_four():
    grades = [grade(5, 5), grade(4, 4), grade(4, 3), grade(2, 5)]
    assert judge.headline(grades) == 50.0
    assert judge.headline([]) == 0.0


def test_fake_judge_is_deterministic_and_rewards_the_reference():
    rows = [row("q1", REFERENCE), row("q2", ""), row("q3", "Bananas are yellow.")]
    first = graded(rows)
    assert first == graded(rows)
    assert [(g["correctness"], g["helpfulness"]) for g in first] == [(5, 5), (1, 1), (1, 1)]
    assert all(g["raw"].startswith("Grade: ") for g in first)


def test_grade_summary_reports_headline_overall_and_by_group():
    rows = [row("q1", REFERENCE), row("q2", "", group="howto"), row("q3", REFERENCE, group="howto")]
    summary = judge.grade_summary(graded(rows))
    assert summary["headline_pct_both_at_least_4"] == pytest.approx(66.7)
    assert summary["by_group"] == {"fact": 100.0, "howto": 50.0}
    assert (summary["count"], summary["errors"]) == (3, 0)


def test_grading_uses_the_current_reference_not_the_one_in_the_row():
    stale = row("q1", "Paris is the capital.", reference="Old reference text.")
    current = {"q1": {"id": "q1", "group": "fact", "question": "New question?", "reference": "New reference."}}
    scripted = Scripted('{"correctness": 5, "helpfulness": 5}')
    result = judge.grade_rows(scripted, [stale], current)
    assert judge.tagged(scripted.prompts[0], "reference") == "New reference."
    assert judge.tagged(scripted.prompts[0], "question") == "New question?"
    assert result[0]["group"] == "fact"


def test_a_bad_reply_or_backend_error_is_recorded_and_grading_continues():
    rows = [row("q1", REFERENCE), row("q2", REFERENCE), row("q3", REFERENCE)]
    scripted = Scripted("not json", RuntimeError("timeout"), '{"correctness": 5, "helpfulness": 4}')
    sunk = []
    results = graded(rows, scripted, sink=sunk.append)
    assert sunk == results
    assert results[0]["raw"] == "not json" and "ValueError" in results[0]["error"]
    assert results[1]["raw"] is None and "timeout" in results[1]["error"]
    assert results[2]["correctness"] == 5 and "error" not in results[2]
    summary = judge.grade_summary(results)
    assert (summary["headline_pct_both_at_least_4"], summary["count"], summary["errors"]) == (100.0, 1, 2)


def test_rows_already_done_are_skipped():
    rows = [row("q1", REFERENCE), row("q2", REFERENCE)]
    assert [g["id"] for g in graded(rows, done={"q1"})] == ["q2"]


def test_grade_prompt_pins_question_reference_and_answer():
    prompt = judge.GRADE_PROMPT.format(question="Q?", reference="R.", answer="A.")
    assert "<question>\nQ?\n</question>" in prompt
    assert "<reference>\nR.\n</reference>" in prompt
    assert "<answer>\nA.\n</answer>" in prompt
    assert '{"correctness": <1-5>' in prompt


def test_real_judge_is_a_stub_until_configured():
    stub = judge.make_judge("llm:some-model")
    assert stub.id == "llm:some-model"
    with pytest.raises(judge.JudgeNotConfigured):
        graded([row("q1", REFERENCE)], stub)
    with pytest.raises(ValueError):
        judge.make_judge("gpt")


def pair(count):
    good = [row(f"q{i}", REFERENCE) for i in range(count)]
    bad = [row(f"q{i}", "I like trains.") for i in range(count)]
    return good, bad


def versus(a, b, seed, grader=None, **options):
    return judge.pairwise_rows(grader or judge.FakeJudge(), a, b, questions_for(a), seed, **options)


def test_pairwise_order_is_random_but_fixed_by_the_seed():
    good, bad = pair(40)
    orders = [r["order"] for r in versus(good, bad, 0)]
    assert set(orders) == {"ab", "ba"}
    assert orders == [r["order"] for r in versus(good, bad, 0)]
    assert orders != [r["order"] for r in versus(good, bad, 1)]


def test_pairwise_winner_is_mapped_back_through_the_order():
    good, bad = pair(20)
    results = versus(good, bad, 0)
    assert all(r["winner"] == "a" for r in results)
    swapped = versus(bad, good, 0)
    assert all(r["winner"] == "b" for r in swapped)
    assert judge.pairwise_summary(results) == {"a_wins": 20, "b_wins": 0, "ties": 0,
                                               "a_wins_more_than_loses": True, "errors": 0}


def test_pairwise_tie_and_the_answer_seen_first():
    scripted = Scripted(json.dumps({"winner": "tie"}))
    results = versus([row("q1", "AAA")], [row("q1", "BBB")], 0, scripted)
    first = judge.tagged(scripted.prompts[0], "answer_1")
    assert first == ("AAA" if results[0]["order"] == "ab" else "BBB")
    assert judge.pairwise_summary(results) == {"a_wins": 0, "b_wins": 0, "ties": 1,
                                               "a_wins_more_than_loses": False, "errors": 0}


def test_pairwise_errors_are_recorded_and_skipped_rows_keep_their_order():
    good, bad = pair(4)
    full = [r["order"] for r in versus(good, bad, 5)]
    scripted = Scripted('{"winner": "A"}', '{"winner": "1"}', '{"winner": "2"}')
    results = versus(good, bad, 5, scripted, done={"q0"})
    assert [r["order"] for r in results] == full[1:]
    assert "error" in results[0] and results[0]["raw"] == '{"winner": "A"}'
    assert judge.pairwise_summary(results)["errors"] == 1
