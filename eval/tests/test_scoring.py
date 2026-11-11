import pytest
import torch
from lm_eval.api.instance import Instance

EOT = 50256


def request(context, continuation):
    return Instance("loglikelihood", {}, (context, continuation), 0)


def direct_logprob(gpt2, tokens, n_continuation):
    with torch.no_grad():
        logits = gpt2(torch.tensor([tokens[:-1]])).logits[0].double()
    total = 0.0
    for position in range(len(tokens) - n_continuation, len(tokens)):
        row = logits[position - 1]
        total += float(row[tokens[position]] - torch.logsumexp(row, dim=0))
    return total


def test_round_trip(scorer):
    response = scorer.score([464, 2068], [3290, 13])
    assert set(response) == {"logprob", "greedy", "scored"}
    assert response["scored"] == 2
    assert response["logprob"] < 0
    assert response["greedy"] is False


def test_logprob_matches_direct_computation(scorer, gpt2):
    context, continuation = [464, 2068, 7586], [21831, 18045]
    expected = direct_logprob(gpt2, context + continuation, len(continuation))
    assert scorer.score(context, continuation)["logprob"] == pytest.approx(expected, abs=1e-4)


def test_truncates_long_context_from_the_left(scorer):
    context = [(i * 37) % 50000 for i in range(1100)]
    continuation = [11, 290]
    long = scorer.score(context, continuation)
    kept = scorer.score(context[-(1024 - len(continuation)) :], continuation)
    assert long == kept
    assert long["scored"] == 2


def test_empty_context_skips_first_token(scorer):
    tokens = [464, 3797, 3332]
    empty = scorer.score([], tokens)
    shifted = scorer.score(tokens[:1], tokens[1:])
    assert empty == shifted
    assert empty["scored"] == 2


def test_wrapper_empty_context_uses_end_of_text_prefix(lm, scorer):
    (logprob, _), = lm.loglikelihood([request("", " The cat sat")], disable_tqdm=True)
    tokens = lm.tok_encode(" The cat sat")
    assert logprob == scorer.score([EOT], tokens)["logprob"]


def test_wrapper_moves_boundary_whitespace_to_continuation(lm):
    trailing, leading = lm.loglikelihood(
        [request("The cat ", "sat"), request("The cat", " sat")], disable_tqdm=True
    )
    assert trailing == leading


def test_error_line_raises_with_request(lm):
    with pytest.raises(RuntimeError, match="99999"):
        lm.score([([1], [99999])])
    assert len(lm.score([([464], [2068])])) == 1


def test_greedy_true_for_argmax_continuation(scorer, gpt2):
    tokens = [464, 2068, 7586]
    for _ in range(3):
        with torch.no_grad():
            tokens.append(int(gpt2(torch.tensor([tokens])).logits[0, -1].argmax()))
    response = scorer.score(tokens[:3], tokens[3:])
    assert response["greedy"] is True
    assert response["scored"] == 3


def test_generation_not_supported(lm):
    with pytest.raises(NotImplementedError):
        lm.generate_until([])
