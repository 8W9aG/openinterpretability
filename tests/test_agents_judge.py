"""The judge agent asks its fuzzing question with the fuzzing prompt (agents/judge.py)."""

from openinterpretability.agents.judge import ExplanationJudge, FuzzingAnswer, TaskWindow


def test_fuzz_sends_the_marked_tokens_and_returns_the_judges_answer(mocker):
    windows = [TaskWindow(text="ab", tokens=["a", "b"], highlighted=[1])]
    client = mocker.Mock()
    client.chat.completions.create.return_value = FuzzingAnswer(correct=[1])
    answer = ExplanationJudge("openai/judge", client=client).fuzz("Fires on b.", windows, "head")
    assert answer.correct == [1]
    sent = str(client.chat.completions.create.call_args)
    assert "a<<b>>" in sent and "Explanation of the head" in sent
