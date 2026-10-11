import pytest

from najdi.cleaning import FlirtRemover
from najdi.text.openers import PRAISE_NEARBY, cut_opener, filler_pattern, opener


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("سؤالك في الصميم يا عمي. وجود الميثان على المريخ يهز العلماء", "وجود الميثان على المريخ يهز العلماء"),
        ("سؤالك مهم جداً للتميز. المكان في الفيزياء مو مجرد فراغ", "المكان في الفيزياء مو مجرد فراغ"),
        ("سؤالك في محله لأن الضوء ينكسر لما يدخل الماء", "لأن الضوء ينكسر لما يدخل الماء"),
        ("هلا والله، سؤالك حلو، وترى الموضوع بسيط جدا", "ترى الموضوع بسيط جدا"),
        ("أهلاً يا عمي، سؤالك حلو. الانفتال يعني إن جزء من الأمعاء يلف", "الانفتال يعني إن جزء من الأمعاء يلف"),
    ],
)
def test_praise_openers_are_cut(text, expected):
    assert cut_opener(text) == (expected, True)


def test_other_openers_only_with_extra_phrases():
    text = "ولا يهمك، هذي مسألة سهلة نحسبها خطوة خطوة"
    assert cut_opener(text) == (text, False)
    assert cut_opener(text, filler_pattern(["ولا يهمك"])) == ("هذي مسألة سهلة نحسبها خطوة خطوة", True)
    assert filler_pattern([]) is None


def test_short_rest_is_left_alone():
    assert cut_opener("سؤالك حلو. صح") == ("سؤالك حلو. صح", False)


def test_praise_nearby_and_opener():
    assert PRAISE_NEARBY.match("الله يعطيك العافية، سؤالك حلو")
    assert opener("أبشر، جرب تحطها هناك وبعدين شوف") == "ابشر جرب تحطها هناك"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("يا حبيبتي، وش فيك اليوم؟", "وش فيك اليوم؟"),
        ("طيب يا حبيبتي، بس لا تتأخرين.", "طيب، بس لا تتأخرين."),
        ("تمام يا قلبي.", "تمام."),
        ("يا بعد قلبي يا حبيبتي، تعالي", "تعالي"),
        ("أبي حلويات يا حلوة؟", "أبي حلويات؟"),
        ("يا حلويات العيد", "يا حلويات العيد"),
        ("السطر الأول.\nيا حبيبتي، السطر الثاني", "السطر الأول.\nالسطر الثاني"),
        ("لا يا حبيبي، قومي ساعديني", "لا يا حبيبي، قومي ساعديني"),
    ],
)
def test_flirt_words_are_removed(text, expected):
    remove = FlirtRemover(["يا حبيبتي", "حبيبتي", "يا قلبي", "يا بعد قلبي", "يا حلو", "يا حلوة"])
    assert remove(text)[0] == expected
