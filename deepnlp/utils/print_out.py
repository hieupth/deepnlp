from typing import Any, Optional, Type, Sequence
from deepnlp.utils.data_struct import TokenClassificationData, ParserData, MultiData
from deepnlp.utils.tokenizer import word_tokenize

def remove_prefix_ner(text, del_prefix= True):
    return text.replace('B-', '').replace('I-', '') if del_prefix else text

def print_tokenclassification(input: Type[TokenClassificationData], language:Type[str], del_prefix_ner:Optional[bool]= None):
    input= input.value()
    text= word_tokenize(input['Sequence'], language)
    inference= input['Inference']
    print(input['Sequence'])
    for i in range(len(text)):
        print(
            str(i + 1)
            + "\t"
            + text[i]
            + "\t"
            + ["O" if text[i] not in inference.keys() else remove_prefix_ner(inference[text[i]]['label'], del_prefix_ner)][0]
        )
def print_parserdata(input: Type[ParserData], language:Type[str]= None):
    input= input.value()
    text= word_tokenize(input['Sequence'], language)
    inference= input['Inference']
    print(input['Sequence'])
    for i in range(len(text)):
        print(
            str(i+1)
            + "\t"
            + inference['xpos'][i]
            + "\t"
            + inference['head'][i]
            + "\t"
            + inference['rela'][i]
        )


def print_multidata(input: Type[ParserData], language:Type[str], del_prefix_ner:Optional[bool]= None):
    input= input.value()
    text= word_tokenize(input['Sequence'], language)
    inference= input['Inference']
    print(input['Sequence'])
    for i in range(len(text)):
        print(
            str(i+1)
            + "\t"
            + inference['xpos'][i]
            + "\t"
            + remove_prefix_ner(inference['ner'][i], del_prefix_ner)
            + "\t"
            + inference['head'][i]
            + "\t"
            + inference['rela'][i]
        )

def print_out(input: Sequence[Any], language= 'eng', del_prefix_ner= True):
    for i in input:
        if isinstance(i, TokenClassificationData):
            print_tokenclassification(i, language, del_prefix_ner)
        elif isinstance(i, ParserData):
            print_parserdata(i, language)
        elif isinstance(i, MultiData):
            print_multidata(i, language, del_prefix_ner)
        print('\n')

