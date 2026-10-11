import json
import random

from najdi.chatdata.config import Config
from najdi.chatdata.filters import RuleFilter, bad_numbers, filler_opener
from najdi.chatdata.judge import Judge
from najdi.chatdata.places import Places, short_summary
from najdi.chatdata.prompts import PromptBuilder
from najdi.chatdata.seeds import KEYS, SITUATIONS, SeedBuilder, pick_followups, split_tags


def test_seeds_are_unique_and_deterministic():
    a, b = SeedBuilder(1234).build("helpful", 300), SeedBuilder(1234).build("helpful", 300)
    assert a == b
    assert len({KEYS["helpful"](s) for s in a}) == 300
    assert {s["situation"] for s in a[: len(SITUATIONS)]} == {split_tags(raw)[1] for _, raw in SITUATIONS}
    identity = SeedBuilder(1234).build("identity", 50)
    assert all(s["turns"] >= 2 for s in identity if s["topic"].endswith("_then_help"))


def test_followups_never_repeat_and_thanks_last():
    rng = random.Random(0)
    for _ in range(200):
        out = pick_followups(rng, 4)
        assert all(x != y for x, y in zip(out, out[1:]))
        assert "thanks" not in out[:-1]


def test_places_and_summary(places_file):
    places = Places(places_file, 40, 3)
    assert {"روضه خريم", "منتزه الثمامه", "الثمامه"} <= places.names
    picked = places.for_persona({"region": "الغربية"}, random.Random(1))
    assert len(picked) == 3
    assert short_summary("نص (بين قوسين) قصير جدا", 3) == "نص قصير جدا..."


def test_prompts(places_file, anchors):
    cfg = Config(creator_name="عبدالرحمن")
    builder = PromptBuilder(cfg, anchors, Places(places_file, 40, 3))
    seed = next(s for s in SeedBuilder(1234).build("helpful", 400) if s["topic"] == "outings_trips")
    task = builder.task(seed)
    assert task["places"] and task == builder.task(seed)
    system, user = builder.messages(task, ["ابشر الحين"])
    assert system["role"] == "system" and "«ابشر الحين»" in user["content"]
    identity = SeedBuilder(1234).build("identity", 5)[0]
    assert "عبدالرحمن" in builder.messages(builder.task(identity))[1]["content"]


def test_rule_filter(places_file):
    cfg = Config()
    rules = RuleFilter(cfg, Places(places_file, 40, 3))
    t = {"subset": "helpful", "topic": "outings_trips"}

    def chat(reply):
        return [{"role": "user", "content": "وش تقترح علي بالويكند؟"}, {"role": "assistant", "content": reply}]

    good = "روح روضة خريم الحين عشان الجو فيها زين، وخذ معك شي تاكله وموية كفاية لأن المكان بعيد شوي."
    assert rules.check(t, chat(good)) == (None, None)
    assert rules.check(t, chat("يا قلبي " + good))[0] == "flirting"
    assert rules.check(t, chat("سؤالك في الصميم، " + good))[0] == "filler_opener"
    assert rules.check(t, chat(good + " وأنا شخصيا جربته بنفسي"))[0] == "personal_life"
    assert rules.check(t, chat(good + " وسعره 350 ريال"))[0] == "numbers"
    assert rules.check(t, chat(good + " وروح روضة نورة"))[0] == "places"
    assert rules.check(t, chat(good + " انا من OpenAI"))[0] == "identity_claims"
    assert rules.opener_cap("helpful") == 30


def test_filler_and_numbers():
    assert filler_opener("هلا والله! تقدر تسوي كذا", "ابي اطبخ كبسة", "helpful") == "greeting"
    assert filler_opener("هلا والله! تقدر تسوي كذا", "السلام عليكم ابي اطبخ", "helpful") is None
    msgs = [{"role": "user", "content": "راتبي 8000"}, {"role": "assistant", "content": "اتصل على 0551234567"}]
    assert bad_numbers({"subset": "helpful", "topic": "budget_saving"}, msgs) == ["0551234567"]


def test_judge(places_file):
    judge = Judge(Config(), Places(places_file, 40, 3))
    t = {"subset": "identity", "topic": "identity"}
    schema = judge.schema(t)
    assert "honest_identity" in schema["properties"] and schema["required"][-1] == "reason"
    verdict = {k: want for k, (_, want) in judge.questions(t).items()}
    assert judge.failures(t, json.dumps({**verdict, "reason": "ok"}))[0] == []
    assert judge.failures(t, "not json") == (None, None)
