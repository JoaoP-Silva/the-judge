import re

def extract_sentences(text : str) -> list[str]:
    """
    Split text to sentences using '.', '!' and '?' as delimiters.
    """
    sentence_delimiters = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_delimiters, text)
    return sentences