import re
import inflect
_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9\,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9\,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9\.\,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
def _remove_commas(m): return m.group(1).replace(',', '')
def _expand_decimal_point(m): return m.group(1).replace('.', ' point ')
def _expand_ordinal(m): return _inflect.number_to_words(m.group(0))
def _expand_number(m): return _inflect.number_to_words(int(m.group(0)), andword='')
def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    return re.sub(_number_re, _expand_number, text)
