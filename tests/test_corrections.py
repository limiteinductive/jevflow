"""Corrections: splitting the local model's fix into word changes, Jev's per-change gate, and deciding each sentence once when pauses overlap.

Jev and the local model are replaced by fakes, so these run offline and give the same answer every time.
"""

import asyncio
import re

import pytest

from flow import decide, generator
from flow.decide import Change, apply, changes, describe


@pytest.fixture(autouse=True)
def empty_caches():
    decide.sentence_fixes.clear()
    decide.sentence_notes.clear()


class FakeJev:
    """Answers the fix gate from a table of word changes: `verdicts[(old, new)]` is (P(fixes a real mistake), P(on purpose)); unlisted changes are good fixes."""

    def __init__(self, verdicts: dict[tuple[str, str], tuple[float, float]] | None = None, lowercase: float = 0.05, hurts: float = 0.1):
        self.verdicts, self.lowercase, self.hurts = verdicts or {}, lowercase, hurts
        self.calls = []

    async def __call__(self, model, prompt):
        self.calls.append(model)
        answers = {}
        for name, field in model.model_fields.items():
            description = field.description or ""
            if name == "lowercase":
                answers[name] = self.lowercase
            elif name.startswith("hurts_"):
                answers[name] = self.hurts
            elif name.startswith(("fixes_", "on_purpose_")):
                fixes, on_purpose = self.verdicts.get(self.edit(description), (0.95, 0.02))
                answers[name] = fixes if name.startswith("fixes_") else on_purpose
        return answers

    @staticmethod
    def edit(description: str) -> tuple[str, str]:
        if found := re.search(r"changing '(.*?)' to '(.*?)'", description):
            return found.group(1), found.group(2)
        if found := re.search(r"adding '(.*?)'", description):
            return "", found.group(1)
        found = re.search(r"removing '(.*?)'", description)
        return found.group(1), ""


def use(monkeypatch, correction: str, jev: FakeJev) -> FakeJev:
    async def fix(sentence: str) -> str:
        return correction

    monkeypatch.setattr(generator, "fix", fix)
    monkeypatch.setattr(decide, "run", jev)
    return jev


# Splitting a correction into word changes


def test_an_unchanged_sentence_has_no_changes():
    assert changes("This is fine.", "This is fine.") == []


def test_a_typo_is_one_change():
    assert changes("i realy like it", "i really like it") == [Change(1, 2, ["really"])]


def test_look_alike_words_split_one_by_one():
    assert changes("The team have finish it.", "The team has finished it.") == [Change(2, 3, ["has"]), Change(3, 4, ["finished"])]


def test_words_that_do_not_pair_up_stay_one_change():
    found = changes("Leboncoin is french second handed marketplace.", "Leboncoin is a French second-hand marketplace.")
    assert found == [Change(2, 5, ["a", "French", "second-hand"])]


def test_applying_every_change_gives_the_correction():
    sentence, correction = "Leboncoin is french second handed marketplace, it help users.", "Leboncoin is a French second-hand marketplace, it helps users."
    assert apply(sentence.split(" "), changes(sentence, correction)) == correction


def test_applying_keeps_the_writers_spacing():
    sentence = "We  shipped teh  build."
    assert apply(sentence.split(" "), changes(sentence, "We  shipped the  build.")) == "We  shipped the  build."


def test_applying_some_changes_leaves_the_others_as_written():
    sentence = "The team have finish it."
    found = changes(sentence, "The team has finished it.")
    assert apply(sentence.split(" "), [found[1]]) == "The team have finished it."


def test_a_change_is_described_before_and_after_in_place():
    words = "The team have finish the report and more".split(" ")
    described = describe(words, Change(3, 4, ["finished"]))
    assert described.startswith("changing 'finish' to 'finished'")
    assert "'... The team have finish the report and ...' reads '... The team have finished the report and ...'" in described


def test_an_insertion_is_described_as_adding():
    assert describe("it is fast".split(" "), Change(1, 1, ["very"])).startswith("adding 'very'")


# Jev's per-change gate


def test_every_good_change_is_applied(monkeypatch):
    use(monkeypatch, "We received the report.", FakeJev())
    assert asyncio.run(decide.decide_fix("We recieved teh report.", "", "", [])) == ("We received the report.", None)


def test_one_bad_change_no_longer_sinks_the_good_ones(monkeypatch):
    jev = FakeJev({("Leboncoin", "LeBonCoin"): (0.2, 0.9)})
    use(monkeypatch, "LeBonCoin is a French second-hand marketplace.", jev)
    fixed, _ = asyncio.run(decide.decide_fix("Leboncoin is french second handed marketplace.", "", "", []))
    assert fixed == "Leboncoin is a French second-hand marketplace."


def test_a_change_below_the_threshold_is_dropped(monkeypatch):
    use(monkeypatch, "The team has finished it.", FakeJev({("have", "has"): (0.4, 0.1)}))
    assert asyncio.run(decide.decide_fix("The team have finish it.", "", "", []))[0] == "The team have finished it."


def test_a_word_chosen_on_purpose_is_kept(monkeypatch):
    use(monkeypatch, "going to ship it tomorrow.", FakeJev({("gonna", "going to"): (0.9, 0.95), ("tmrw.", "tomorrow."): (0.9, 0.9)}))
    assert asyncio.run(decide.decide_fix("gonna ship it tmrw.", "", "", [])) == (None, None)


def test_capitalizing_is_dropped_when_the_draft_is_lowercase_on_purpose(monkeypatch):
    use(monkeypatch, "I really like it.", FakeJev(lowercase=0.9))
    assert asyncio.run(decide.decide_fix("i realy like it.", "", "", []))[0] == "i really like it."


def test_capitalizing_is_kept_in_a_normal_draft(monkeypatch):
    use(monkeypatch, "I really like it.", FakeJev(lowercase=0.05))
    assert asyncio.run(decide.decide_fix("i realy like it.", "", "", []))[0] == "I really like it."


def test_a_sentence_the_model_leaves_alone_never_asks_jev(monkeypatch):
    jev = use(monkeypatch, "This is fine.", FakeJev())
    assert asyncio.run(decide.decide_fix("This is fine.", "", "", [])) == (None, None)
    assert jev.calls == []


def test_every_change_rides_in_one_jev_request(monkeypatch):
    jev = use(monkeypatch, "I really like the new build.", FakeJev())
    asyncio.run(decide.decide_fix("I realy like teh new build.", "", "", []))
    assert len(jev.calls) == 1
    assert {"fixes_0", "on_purpose_0", "fixes_1", "on_purpose_1", "lowercase"} <= set(jev.calls[0].model_fields)


# Deciding each sentence once


def test_overlapping_callers_share_one_decision():
    runs = 0

    async def decide_once():
        nonlocal runs
        runs += 1
        await asyncio.sleep(0.01)
        return "done"

    async def main():
        cache = {}
        return await asyncio.gather(*(decide.once(cache, "key", decide_once) for _ in range(3)))

    assert asyncio.run(main()) == ["done"] * 3
    assert runs == 1


def test_a_failed_decision_is_retried_by_the_next_caller():
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError
        return "done"

    async def main():
        cache = {}
        with pytest.raises(TimeoutError):
            await decide.once(cache, "key", flaky)
        await asyncio.sleep(0)
        return await decide.once(cache, "key", flaky)

    assert asyncio.run(main()) == "done"
    assert attempts == 2


def test_a_caller_that_gives_up_does_not_cancel_the_decision_others_wait_on():
    async def slow():
        await asyncio.sleep(0.05)
        return "done"

    async def main():
        cache = {}
        impatient = asyncio.create_task(decide.once(cache, "key", slow))
        patient = asyncio.create_task(decide.once(cache, "key", slow))
        await asyncio.sleep(0.01)
        impatient.cancel()
        return await patient

    assert asyncio.run(main()) == "done"


def test_overlapping_pauses_decide_each_finished_sentence_once(monkeypatch):
    decided = []

    async def decide_fix(sentence, draft, goal, categories):
        decided.append(sentence)
        await asyncio.sleep(0.01)
        return (sentence.replace("teh", "the"), None) if "teh" in sentence else (None, None)

    monkeypatch.setattr(decide, "decide_fix", decide_fix)
    text = "We shipped teh build. It is fast. and still typi"
    limit = text.index("and")

    async def main():
        return await asyncio.gather(decide.find_fixes(text, limit, "", []), decide.find_fixes(text, limit, "", []))

    (fixes, _), (again, _) = asyncio.run(main())
    assert sorted(decided) == ["It is fast.", "We shipped teh build."]
    assert [fix.replacement for fix in fixes] == [fix.replacement for fix in again] == ["We shipped the build."]


def test_overlapping_pauses_ask_about_each_note_sentence_once(monkeypatch):
    asked = []

    async def decide_note(sentence, comment, goal, enabled):
        asked.append(sentence)
        await asyncio.sleep(0.01)
        return None

    monkeypatch.setattr(decide, "decide_note", decide_note)
    text = "im writing a blog post. We shipped it."

    async def main():
        return await asyncio.gather(*(decide.find_notes(text, "", [], "", frozenset()) for _ in range(3)))

    assert asyncio.run(main()) == [[], [], []]
    assert sorted(asked) == ["We shipped it", "im writing a blog post"]
