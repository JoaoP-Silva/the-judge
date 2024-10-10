import re

def extract_sentences(text : str) -> list[str]:
    """
    Split text to sentences using '.', '!' and '?' as delimiters.
    """
    sentence_delimiters = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_delimiters, text)
    return sentences


def get_right_sentence(answer : str, sentences : list[str]) -> int:
        """
        Return the sentence index that matches the given answer. 
        """
        for i, sentence in enumerate(sentences): 
            if answer in sentence:
                return i            
        return 0