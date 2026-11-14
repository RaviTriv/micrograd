import json

import pytest

from chat import answer
from serve import BLIND_PAGE, Blind


def saved(path, model, count=30):
    rows = [{"id": f"q{i:03d}", "group": "fact", "question": f"Question {i}?", "reference": f"Reference {i}.",
             "model": model, "answer": f"{model} answer {i}"} for i in range(count)]
    answer.write_jsonl(path, rows)
    return path


@pytest.fixture
def files(tmp_path):
    return saved(tmp_path / "new.jsonl", "new"), saved(tmp_path / "old.jsonl", "old"), tmp_path / "votes.jsonl"


def test_items_hide_the_model_names_and_shuffle_sides(files):
    a, b, votes = files
    blind = Blind(a, b, votes, count=20, seed=0)
    public = blind.public()
    assert len(public) == 20
    assert all(set(item) == {"index", "question", "reference", "left", "right", "voted"} for item in public)
    assert "new" not in json.dumps([{k: v for k, v in item.items() if k not in ("left", "right")} for item in public])
    lefts = {item["left"].split()[0] for item in public}
    assert lefts == {"new", "old"}


def test_same_seed_gives_same_questions_and_sides(files):
    a, b, votes = files
    first = Blind(a, b, votes, count=20, seed=3).public()
    assert first == Blind(a, b, votes, count=20, seed=3).public()
    assert first != Blind(a, b, votes, count=20, seed=4).public()


def test_vote_is_recorded_against_the_hidden_model(files):
    a, b, votes = files
    blind = Blind(a, b, votes, count=20, seed=0)
    item = blind.public()[0]
    left_model = item["left"].split()[0]
    vote = blind.record(0, "left")
    assert (vote["winner"], vote["left"], vote["a"], vote["b"], vote["seed"]) == (left_model, left_model, "new", "old", 0)
    blind.record(1, "tie")
    rows = answer.read_jsonl(votes)
    assert [r["choice"] for r in rows] == ["left", "tie"]
    assert rows[1]["winner"] == "tie"
    assert rows[0]["question_id"] == blind.items[0]["id"]


def test_votes_survive_a_restart_and_bad_choices_are_refused(files):
    a, b, votes = files
    Blind(a, b, votes, count=20, seed=0).record(2, "right")
    again = Blind(a, b, votes, count=20, seed=0)
    assert [item["voted"] for item in again.public()][:3] == [False, False, True]
    with pytest.raises(ValueError):
        again.record(0, "both")
    assert len(answer.read_jsonl(votes)) == 1


def test_an_item_takes_one_vote_only(files):
    a, b, votes = files
    blind = Blind(a, b, votes, count=20, seed=0)
    blind.record(0, "left")
    with pytest.raises(ValueError, match="already"):
        blind.record(0, "right")
    assert len(answer.read_jsonl(votes)) == 1


@pytest.mark.parametrize("index", [-1, 20, "0", 1.0, None, True])
def test_bad_indexes_are_refused(files, index):
    a, b, votes = files
    blind = Blind(a, b, votes, count=20, seed=0)
    with pytest.raises(ValueError):
        blind.record(index, "left")
    assert not votes.exists()


def test_page_ignores_votes_while_one_is_in_flight():
    assert "if (!current || busy) return;" in BLIND_PAGE
    assert BLIND_PAGE.index("busy = true;") < BLIND_PAGE.index('await fetch("/vote"')


def test_two_files_from_the_same_model_are_refused(tmp_path):
    a, b = saved(tmp_path / "one.jsonl", "same"), saved(tmp_path / "two.jsonl", "same")
    with pytest.raises(ValueError, match="same"):
        Blind(a, b, tmp_path / "votes.jsonl", count=20, seed=0)
