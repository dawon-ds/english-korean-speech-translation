from .korean import tokenize as ko_tokenize

def korean_cleaners(text):
    return ko_tokenize(text, as_id=False)
