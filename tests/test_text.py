from najdi.text.arabic import normalize, phrase_hits, phrases, stem_forms, term_hits, terms, words
from najdi.text.checks import foreign_script, longest_english_run
from najdi.text.dedup import MinHashIndex, is_near_dup, shingles
from najdi.text.dialect import check_per_chat, check_per_message, marker_counts
from najdi.text.safety import sensitive_hits


def test_normalize_and_words():
    assert normalize("إِلى المدرسةِ") == "الي المدرسه"
    assert words("هلا، وش أخبارك 123؟") == ["هلا", "وش", "اخبارك"]


def test_stem_forms_strip_prefixes():
    assert {"والكتاب", "الكتاب", "كتاب"} <= stem_forms("والكتاب")
    assert stem_forms("ولد") == {"ولد"}


def test_term_hits_with_prefixes_and_pairs():
    uni, bi = terms(["جاذبية", "سرعة الضوء"])
    assert term_hits("وبالجاذبية تتحرك الأشياء بسرعة الضوء", uni, bi) == {"جاذبيه", "سرعه الضوء"}


def test_phrase_hits_claims_skip_vocatives_and_articles():
    items = phrases(["أخوي", "أهلي"])
    assert phrase_hits("ابشر يا أخوي", items, claims=True) == set()
    assert phrase_hits("الأهلي فاز", items, claims=True) == set()
    assert phrase_hits("أخوي ساعدني", items, claims=True) == {"اخوي"}


def test_markers_and_najdi_checks():
    saudi, fusha = marker_counts("وش تبي الحين؟ ماذا تريد")
    assert (saudi, fusha) == (3, 2)
    chat = [
        {"role": "user", "content": "وش اسوي الحين؟"},
        {"role": "assistant", "content": "ابشر، جرب تحطها في مكان بارد عشان ما تخرب وشوف وش يصير بعدين"},
    ]
    assert check_per_chat(chat, 5.0, 1, 6, 0.2)
    fusha_chat = [
        {"role": "user", "content": "ماذا أفعل الآن؟"},
        {"role": "assistant", "content": "يجب عليك أن تذهب إلى الطبيب لكي تتأكد من حالتك وهذا هو الحل"},
    ]
    assert not check_per_chat(fusha_chat, 5.0, 1, 6, 0.2)
    assert not check_per_message(fusha_chat, 8.0, 1, 0.2)[0]


def test_sensitive_terms_ignore_biology():
    assert sensitive_hits("التكاثر الجنسي في النبات") == set()
    assert sensitive_hits("موضوع الجماع") == {"جماع"}


def test_script_checks():
    assert longest_english_run("this is a very good idea") == 6
    assert foreign_script("كلام عربي مع 汉字", 3)
    assert not foreign_script("استخدم H2O بس", 3)


def test_near_duplicates():
    a = shingles("وش أفضل مكان أطلع له مع العائلة")
    b = shingles("وش أفضل مكان أطلع له مع العايلة")
    assert is_near_dup(a, [b], 0.6)
    assert not is_near_dup(a, [shingles("كيف اسوي كبسة لاول مرة")], 0.6)
    index = MinHashIndex(0.7)
    assert index.add("هذا نص طويل عن الفيزياء والطاقة والحركة") is None
    assert index.add("هذا نص طويل عن الفيزياء والطاقة والحركة") == "exact"
    assert index.add("هذا نص طويل عن الفيزياء والطاقة والحركه!") == "exact"
    assert index.add("هذا نص طويل عن الفيزياء والطاقة والحركة السريعة") == "near"
