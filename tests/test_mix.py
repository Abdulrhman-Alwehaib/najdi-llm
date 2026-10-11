from najdi.mix.config import Config, Report
from najdi.mix.sources import Cleaner
from najdi.mix.splits import Splitter


def row(i, source, strata="", reply="تمام"):
    return {
        "id": f"{source}-{i}",
        "source": source,
        "_strata": strata,
        "messages": [{"role": "user", "content": "سؤال"}, {"role": "assistant", "content": reply}],
    }


def test_allocate_and_spread():
    assert Splitter.allocate(10, {"a": 1, "b": 1, "c": 2}) == {"a": 3, "b": 2, "c": 5}
    splitter = Splitter(Config())
    rows = [row(i, "voho_everyday", strata=f"t{i % 3}") for i in range(30)]
    picked = splitter.spread_pick(rows, 6)
    assert len(picked) == 6 and len({r["_strata"] for r in picked[:3]}) == 3


def test_splits_repeat_helpful_and_keep_test_clean():
    cfg = Config(
        science_n=40,
        voho_everyday_n=10,
        voho_enterprise_n=10,
        fusha_n=10,
        val_frac=0.1,
        test_n={"science_logic": 4, "voho": 4, "fusha": 2},
    )
    rows = (
        [row(i, "helpful", "outings") for i in range(20)]
        + [row(i, "identity", "identity") for i in range(10)]
        + [row(i, "science", "physics") for i in range(30)]
        + [row(i, "math_logic", "algebra") for i in range(20)]
        + [row(i, "voho_everyday", "a / b") for i in range(15)]
        + [row(i, "voho_enterprise", "c / d") for i in range(15)]
        + [row(i, "fusha") for i in range(15)]
    )
    splits, unique = Splitter(cfg)(rows)
    helpful_train = [r for r in splits["train"] if r["source"] == "helpful"]
    assert len(helpful_train) == 2 * (20 - 2) and sum(r["id"].endswith("-copy2") for r in helpful_train) == 18
    assert not any(r["source"] in ("helpful", "identity") for r in splits["test"])
    assert unique["helpful"] == 20 and unique["science"] + unique["math_logic"] == 40


def test_cleaner_fixes_roles_and_cuts_abshir():
    cfg = Config(non_najdi_words=["هيك"], voho_personal=["سيارتي"])
    report = Report()
    clean = Cleaner(cfg, report)
    rows = [row(i, "helpful", reply="أبشر، جرب تحطها في الثلاجة عشان ما تخرب") for i in range(10)]
    rows.append(row(99, "helpful", reply="هيك أحسن لك"))
    rows.append(row(1, "voho_everyday", reply="سيارتي خربانة"))
    rows.append(
        {"id": "x", "source": "voho_enterprise", "_strata": "", "messages": [{"role": "assistant", "content": "هلا"}]}
    )
    kept = clean(rows)
    assert {r["id"] for r in kept} == {f"helpful-{i}" for i in range(10)}
    assert report.abshir["helpful"] == (10, 5)
    assert sum(r["messages"][1]["content"].startswith("جرب") for r in kept) == 5
    assert report.removed["non_najdi"]["helpful"] == 1 and report.removed["voho_personal"]["voho_everyday"] == 1
