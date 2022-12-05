import re 
import unicodedata as ud


from typing import Type, List
def word_tokenize_vie(text:Type[str]):
    text = ud.normalize('NFC', text)
    specials = ["==>", "->", "\.\.\.", ">>",'\n']
    digit = "\d+([\.,_]\d+)+"
    email = "([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
    #web = "^(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+$"
    web = "\w+://[^\s]+"
    #datetime = [
    #    "\d{1,2}\/\d{1,2}(\/\d{1,4})(^\dw. )+",
    #    "\d{1,2}-\d{1,2}(-\d+)?",
    #]
    word = "\w+"
    non_word = "[^\w\s]"
    abbreviations = [
        "[A-ZĐ]+\.",
        "Mr\.", "Mrs\.", "Ms\.",
        "Dr\.", "ThS\."
    ]

    patterns = []
    patterns.extend(abbreviations)
    patterns.extend(specials)
    patterns.extend([web, email])
    patterns.extend([digit, non_word, word])

    patterns = "(" + "|".join(patterns) + ")"
    tokens = re.findall(patterns, text, re.UNICODE)
    return [token[0] for token in tokens]

def word_tokenize_eng(text):
    word= re.findall(r"[\w'\"]+|[,.!?;:]", text)
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



if __name__ == "__main__":
    result= word_tokenize_vie("Xin chào tôi tên là Nguyễn Tiến Đạt")
    print(result)