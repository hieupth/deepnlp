import re 
from typing import Type, List

def word_tokenize_eng(text):
    word= re.findall(r"[\w'\"]+|[,.!?]", text)
    return word

def sentence_tokenize_eng(text):
    sentences= re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def word_tokenize(text: Type[str], language: Type[str]= 'eng') -> List[str]:
    if language == 'eng':
        return word_tokenize_eng(text)
    else:
        pass
def sentence_tokenize(text: Type[str], language: Type[str]= 'eng') -> List[str]:
    if language == 'eng':
        return sentence_tokenize_eng(text)