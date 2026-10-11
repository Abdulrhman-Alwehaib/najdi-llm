from pathlib import Path

from najdi import settings
from najdi.llm.output import parse_turns, strip_thinking
from najdi.science.checks import answer_in_reply, answers_match, cheap_filter, final_answer, numbers_in
from najdi.science.config import Config
from najdi.science.tasks import TaskFactory
from najdi.science.wiki import KeywordIndex, candidate, extract_passage

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_numbers_and_answers():
    assert numbers_in("الجواب ٢٠٤٠ ريال و 1,250 و 3/4") == [2040.0, 1250.0, 0.75]
    assert answers_match("2040 ريال", "٢٠٤٠")
    assert answers_match("25%", "0.25")
    assert not answers_match("12", "13")
    assert answer_in_reply("2040", "يعني بتدفع 2040 ريال")
    assert final_answer("<think>x</think>\nFINAL: 12 سنة", "stop") == "12 سنة"
    assert final_answer("FINAL: 3", "length") is None


def test_parse_turns():
    text = '{"turns": [{"user": "وش الطقس", "assistant": "الطقس حال الجو"}, {"user": "طيب", "assistant": "زين"}]}'
    msgs, obj = parse_turns(strip_thinking("<think>...</think>" + text), (2, 4))
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert parse_turns(text, 3) is None
    assert strip_thinking("<think>never ends") == ""


def test_cheap_filter():
    cfg = Config()
    reply = "وش " * 40
    msgs = [{"role": "user", "content": "وش الفرق"}, {"role": "assistant", "content": reply}]
    assert cheap_filter(msgs, cfg.filters) is None
    msgs[1]["content"] = reply + " حسب النص"
    assert cheap_filter(msgs, cfg.filters) == "source_mention"
    msgs[1]["content"] = "**مهم** " + reply
    assert cheap_filter(msgs, cfg.filters) == "markdown"


def test_passage_and_candidate():
    sentence = "الجاذبية قوة تجذب الأجسام نحو بعضها وتسبب سقوط الأشياء على الأرض وحركة الكواكب حول الشمس. "
    text = "مقدمة\n" + sentence * 12
    passage, n = extract_passage(text, 60, 120)
    assert 60 <= n <= 120 and passage.endswith(".")
    index = KeywordIndex({"physics": "جاذبية كواكب الشمس قوة_الجاذبية", "space": "كواكب"})
    article = {"id": "1", "url": "u", "title": "جاذبية", "text": text}
    cfg = settings.load(Config, CONFIGS / "science.yaml", ["wiki.passage_words=[60, 120]", "wiki.min_keyword_hits=2"])
    found = candidate(article, index, cfg.wiki)
    assert found and found["kw_topic"] == "physics"
    assert candidate({**article, "title": "نادي الهلال"}, index, cfg.wiki) is None


def test_tasks_are_deterministic_and_use_domain():
    cfg = settings.load(Config, CONFIGS / "medical.yaml")
    anchors = [[{"role": "user", "content": "هلا"}, {"role": "assistant", "content": "هلا بك"}]] * 4
    factory = TaskFactory(cfg, anchors)
    passage = {"id": "42", "label": "medical", "title": "حمى", "url": "u", "passage": "الحمى ارتفاع حرارة الجسم."}
    a, b = factory.science(passage), factory.science(passage)
    assert a == b and a["id"] == "med-42"
    user = factory.messages(a)[1]["content"]
    assert cfg.domain.extra_rules.strip() in user and "الحمى ارتفاع" in user
    math = factory.math(3)
    assert math["id"] == "math-00003" and factory.schema(math)["properties"]["final_answer"] == {"type": "string"}
