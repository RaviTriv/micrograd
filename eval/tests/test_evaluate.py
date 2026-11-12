from pathlib import Path

import evaluate


def test_path_inside_repo_is_recorded_relative_to_repo():
    path = evaluate.REPO / "thunder-results" / "run" / "long.bin"
    assert evaluate.recorded_path(path) == "thunder-results/run/long.bin"


def test_path_outside_repo_keeps_only_its_name(tmp_path):
    assert evaluate.recorded_path(tmp_path / "build-rel" / "gpt_score") == "<outside-repo>/gpt_score"


def test_harness_output_has_no_absolute_paths():
    task_yaml = evaluate.SITE_PACKAGES / "lm_eval" / "tasks" / "hellaswag" / "hellaswag.yaml"
    results = {
        "configs": {"hellaswag": {"task_path": str(task_yaml)}},
        "paths": [str(evaluate.REPO / "eval" / ".cache"), str(Path.home() / "hf")],
        "n": 3,
    }
    assert evaluate.portable(results) == {
        "configs": {"hellaswag": {"task_path": "lm_eval/tasks/hellaswag/hellaswag.yaml"}},
        "paths": ["eval/.cache", "~/hf"],
        "n": 3,
    }
