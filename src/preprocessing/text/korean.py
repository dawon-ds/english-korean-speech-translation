from jamo import hangul_to_jamo

PAD = "_"
EOS = "~"
PUNC = "!\'(),-.:;?"
SPACE = " "
_SILENCES = ["sp", "spn", "sil"]

JAMO_LEADS = "".join(chr(_) for _ in range(0x1100, 0x1113))
JAMO_VOWELS = "".join(chr(_) for _ in range(0x1161, 0x1176))
JAMO_TAILS = "".join(chr(_) for _ in range(0x11A8, 0x11C3))

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
KOR_SYMBOLS = list(PAD + EOS + VALID_CHARS) + _SILENCES

def tokenize(text, as_id=False):
    tokens = list(hangul_to_jamo(text))
    return tokens
