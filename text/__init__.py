import re
from .symbols import kor_symbols as symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")

def text_to_sequence(text, cleaner_names):
    sequence = []
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence = _symbols_to_sequence(text)
            break
        sequence = _symbols_to_sequence(m.group(2))
        text = m.group(3)
    return sequence

def sequence_to_text(sequence):
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]
    return result.replace("}{", " ")

def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols.split() if _should_keep_symbol(s)]

def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "~" and s != "_"
