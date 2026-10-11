SYSTEM_FUSHA = {"role": "system", "content": "رد بالفصحى"}

FIXED_PROMPTS = [
    ("saudi chat", [{"role": "user", "content": "هلا والله، وش أخبارك؟ من زمان عنك"}]),
    ("saudi chat", [{"role": "user", "content": "ودي أطلع البر هالويكند بس ما أدري وين أروح، وش تقترح علي؟"}]),
    ("saudi chat", [{"role": "user", "content": "ولدي ما يبي يذاكر وطول يومه على الجوال، وش أسوي معه؟"}]),
    ("saudi chat", [{"role": "user", "content": "تعبت من زحمة الرياض، كل يوم ساعة رايح وساعة راجع"}]),
    ("science", [{"role": "user", "content": "ليش السما لونها أزرق؟"}]),
    ("science", [{"role": "user", "content": "كيف الثلاجة تبرد الأكل؟ وش اللي يصير داخلها؟"}]),
    ("science", [{"role": "user", "content": "ليش القمر يتغير شكله خلال الشهر؟"}]),
    ("math", [{"role": "user", "content": "اشتريت جوال سعره 2400 ريال وعليه خصم 15%، كم بدفع؟"}]),
    ("math", [{"role": "user", "content": "سيارة تمشي 120 كيلو بالساعة، كم تاخذ عشان تقطع 300 كيلو؟"}]),
    ("fusha", [SYSTEM_FUSHA, {"role": "user", "content": "ما أهمية القراءة في حياة الإنسان؟"}]),
    ("fusha", [SYSTEM_FUSHA, {"role": "user", "content": "اشرح باختصار كيف تتكون الأمطار."}]),
    ("outside the data", [{"role": "user", "content": "اكتب لي قصيدة قصيرة عن الرياض"}]),
]
