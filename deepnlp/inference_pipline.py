from typing import Any, Tuple, Type, List, Optional
import numpy as np
import tensorflow as tf 
from transformers import AutoTokenizer
from deepnlp.utils.data_struct import TokenClassificationData, ParserData, MultiData
from deepnlp.utils.tokenizer import word_tokenize, sentence_tokenize
from deepnlp.preprocess_load import (
    load_model, 
    load_vocabs,
    ensure_model_name,
    ensure_vocabs_name,
)

def word_ids(tokens: List[str], input_ids, tokenizer):
  ids= [] 
  for i, token in enumerate(tokens):
    ids.extend([i] + [None] * (len(tokenizer.tokenize(token)) - 1) )
  eos_ids= np.where(input_ids[0] == 2)[0][0]
  ids= ids[:eos_ids - 1]
  ids= [None] + ids + [None] * (len(input_ids[0]) - len(ids) - 1)
  return ids


class MultiTask:
    def __init__(self, model_name: Type[str]):
        self.__model_name= model_name
        ensure_model_name(model_name)

        # load model and vocab
        self.__model= load_model(model_name)
        self.__vocab= load_vocabs(model_name, task= 'multi') # pos, ner, dp 

        # tokenizer
        if model_name == 'deepnlp_eng':
            self.__tokenizer_name= 'distilroberta-base'
            self.__language= 'eng'
        self.__tokenizer= AutoTokenizer.from_pretrained(self.__tokenizer_name, add_prefix_space= True, use_fast= True)
    def __get_output(self, text: List[str], device:Optional[str]= None):
        e= self.__tokenizer.encode_plus(text, return_tensors= 'np',
                                        padding= 'longest', max_length= 512, truncation= True, is_split_into_words= True)
        e_word_ids= word_ids(text, e['input_ids'], self.__tokenizer)

        data= (tf.convert_to_tensor(e['input_ids']), tf.convert_to_tensor(e['attention_mask']))

        with tf.device(device):
            y_pred= self.__model.predict(data)

        return (y_pred, e_word_ids)

    def __process_token(self, text:List[str], task: Type[str],  device:Optional[str]= None):
        assert task in ['pos_tagger', 'ner_tagger', 'dp_parser', 'multi']
        y_pred, e_word_ids= self.__get_output(text, device)

        y_true= [-1] + [-1 if i is None else i for i in e_word_ids]

        if task == 'ner_tagger':
            y_pred= y_pred[1]
            result= tf.boolean_mask(y_pred[0], tf.not_equal(y_true[1:], -1))

            result_= [-1 if i == 1 else i for i in np.argmax(result, axis= -1)]

            return list(
                zip(
                    tf.boolean_mask(text, tf.not_equal(result_, -1)).numpy(),
                    tf.boolean_mask(np.max(result, axis= -1), tf.not_equal(result_, -1)).numpy(),
                    tf.boolean_mask(result_, tf.not_equal(result_, -1)).numpy()
                )
            )

        elif task == 'pos_tagger':
            y_pred= y_pred[0]
            result= tf.boolean_mask(y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()
            return list(
                zip(
                    text, 
                    np.max(result, axis= -1), 
                    np.argmax(result, axis= -1)
                )
            )
        elif task == 'dp_parser': 
            pos_y_pred= y_pred[0]
            pos_result= tf.boolean_mask(pos_y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()

            y_pred= y_pred[-1]

            head= tf.boolean_mask(np.argmax(y_pred[0], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()
            relation= tf.boolean_mask(np.argmax(y_pred[1], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()

            return list(
                zip(
                    text, 
                    np.argmax(pos_result, axis= -1),
                    head,
                    relation 
                )
            )

        elif task == 'multi':
            pos_y_pred= y_pred[0]
            ner_y_pred= y_pred[1]
            y_pred= y_pred[-1]

            pos_result= tf.boolean_mask(pos_y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()
            ner_result= tf.boolean_mask(ner_y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()


            head= tf.boolean_mask(np.argmax(y_pred[0], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()
            relation= tf.boolean_mask(np.argmax(y_pred[1], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()        

            return list(
                zip(
                    text, 
                    np.argmax(pos_result, axis= -1), 
                    np.argmax(ner_result, axis= -1), 
                    head,
                    relation, 
                )
            )



    def __pos_tagger(self, text:Type[str], device:Optional[str]= None) -> TokenClassificationData:
        text_ = word_tokenize(text, language= self.__language)
        result= self.__process_token(text_, 'pos_tagger', device)
        return TokenClassificationData(
            {'Sequence': text,
            'Inference':{
                      f'{i}': {'score': v, 'label': self.__vocab[0][m]} for i, v, m in result
              }
            }
        )

    def __ner_tagger(self, text:Type[str], device:Optional[str]= None) -> TokenClassificationData:
        text_ = word_tokenize(text, language= self.__language)
        result= self.__process_token(text_, 'ner_tagger', device)
        return TokenClassificationData(
            {'Sequence': text,
            'Inference':{
                      f'{i}': {'score': v, 'label': self.__vocab[1][m]} for i, v, m in result
              }
            }
        )

    def __dp_parser(self, text:Type[str], device:Optional[str]= None) -> ParserData:
        text_ = word_tokenize(text, language= self.__language)
        result= self.__process_token(text_, 'dp_parser', device)

        return ParserData(
            {
                'Sequence': text, 
                'Inference':{
                    'xpos': [self.__vocab[0][i] for (_, i, v, m) in result], 
                    'head': [v for (_, i, v, m) in result],
                    'rela':[self.__vocab[-1][m] for (_, i, v, m) in result],
                }
            }
        )
    
    def __multi(self, text:Type[str], device:Optional[str]= None) -> MultiData:
        text_= word_tokenize(text, language= self.__language)
        result= self.__process_token(text_, 'multi', device)

        return MultiData(
            {
                'Sequence': text, 
                'Inference':{
                    'xpos': [self.__vocab[0][i] for (_, i, k, v, m) in result], 
                    'ner': [self.__vocab[1][k] for (_, i, k, v, m) in result],
                    'head': [v for (_, i, k, v, m) in result],
                    'rela':[self.__vocab[-1][m] for (_, i, k, v, m) in result],
                }
            }
        )

    def inference(self, text:Type[str], device:Optional[str]= None) -> MultiData:
        return self.__multi(text, device)

    def __str__(self) -> str:
        return f'model_name: {self.__model_name}, vocab_name: {self.__model_name}, tokenizer_name: {self.__tokenizer_name}'

    def __repr__(self) -> str:
        return f'model_name: {self.__model_name}, vocab_name: {self.__model_name}, tokenizer_name: {self.__tokenizer_name}'


class PosTagger(MultiTask):
    def __init__(self, model_name:Type[str]):
        super().__init__(model_name)
    def inference(text: Type[str], device:Optional[str]= None) -> TokenClassificationData:
        return super().__pos_tagger(text, device)
    def __str__(self) -> str:
        return super().__str__()
    def __repr__(self) -> str:
        return super().__repr__()

class NerTagger(MultiTask):
    def __init__(self, model_name:Type[str]):
        super().__init__(model_name)
    def inference(text: Type[str], device:Optional[str]= None) -> TokenClassificationData:
        return super().__ner_tagger(text, device)
    def __str__(self) -> str:
        return super().__str__()
    def __repr__(self) -> str:
        return super().__repr__()

class DPParser(MultiTask):
    def __init__(self, model_name:Type[str]):
        super().__init__(model_name)
    def inference(text: Type[str], device:Optional[str]= None) -> ParserData:
        return super().__dp_parser(text, device)
    def __str__(self) -> str:
        return super().__str__()
    def __repr__(self) -> str:
        return super().__repr__()


class pipline:
    def __init__(self, model, task:Type[str]):
        assert task in ['pos_tagger', 'ner_tagger', 'dp_parser', 'multi']
        self.__model= model 
        self.__task= task  
        if self.__model._name== 'deepnlp_eng':
            self.__tokenizer_name= 'distilroberta-base'
            self.__language= 'eng'
            self.__vocab= load_vocabs('deepnlp_eng', task= 'multi')
        
        self.__tokenizer= AutoTokenizer.from_pretrained(self.__tokenizer_name, add_prefix_space= True, use_fast= True)

    def __preprocess(self, text: List[str], device:Optional[str]= None):
        e= self.__tokenizer.encode_plus(text, return_tensors= 'np',
                                        padding= 'longest', max_length= 512, truncation= True, is_split_into_words= True)
        e_word_ids= word_ids(text, e['input_ids'], self.__tokenizer)

        data= (tf.convert_to_tensor(e['input_ids']), tf.convert_to_tensor(e['attention_mask']))

        with tf.device(device):
            y_pred= self.__model.predict(data)

        return (y_pred, e_word_ids)
    
    def __process_token(self, text:List[str], device:Optional[str]= None):
        y_pred, e_word_ids= self.__preprocess(text, device)

        y_true= [-1] + [-1 if i is None else i for i in e_word_ids]

        if self.__task == 'ner_tagger':
            y_pred= y_pred[1]
            result= tf.boolean_mask(y_pred[0], tf.not_equal(y_true[1:], -1))

            result_= [-1 if i == 1 else i for i in np.argmax(result, axis= -1)]

            return list(
                zip(
                    tf.boolean_mask(text, tf.not_equal(result_, -1)).numpy(),
                    tf.boolean_mask(np.max(result, axis= -1), tf.not_equal(result_, -1)).numpy(),
                    tf.boolean_mask(result_, tf.not_equal(result_, -1)).numpy()
                )
            )

        elif self.__task == 'pos_tagger':
            y_pred= y_pred[0]
            result= tf.boolean_mask(y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()
            return list(
                zip(
                    text, 
                    np.max(result, axis= -1), 
                    np.argmax(result, axis= -1)
                )
            )
        elif self.__task == 'dp_parser': 
            pos_y_pred= y_pred[0]
            pos_result= tf.boolean_mask(pos_y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()

            y_pred= y_pred[-1]

            head= tf.boolean_mask(np.argmax(y_pred[0], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()
            relation= tf.boolean_mask(np.argmax(y_pred[1], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()

            return list(
                zip(
                    text, 
                    np.argmax(pos_result, axis= -1),
                    head,
                    relation 
                )
            )

        elif self.__task == 'multi':
            pos_y_pred= y_pred[0]
            ner_y_pred= y_pred[1]
            y_pred= y_pred[-1]

            pos_result= tf.boolean_mask(pos_y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()
            ner_result= tf.boolean_mask(ner_y_pred[0], tf.not_equal(y_true[1:], -1)).numpy()


            head= tf.boolean_mask(np.argmax(y_pred[0], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()
            relation= tf.boolean_mask(np.argmax(y_pred[1], axis= -1)[0], tf.not_equal(y_true, -1)).numpy()        

            return list(
                zip(
                    text, 
                    np.argmax(pos_result, axis= -1), 
                    np.argmax(ner_result, axis= -1), 
                    head,
                    relation, 
                )
            )

    
    def __call__(self, text: Type[str], device: Optional[str]= None): 
        text_= word_tokenize(text, language= self.__language)
        result= self.__process_token(text_, device)

        if self.__task== 'pos_tagger':
            return TokenClassificationData(
            {'Sequence': text,
            'Inference':{
                      f'{i}': {'score': v, 'label': self.__vocab[0][m]} for i, v, m in result
              }
            })
        elif self.__task == 'ner_tagger':
            return TokenClassificationData(
            {'Sequence': text,
            'Inference':{
                      f'{i}': {'score': v, 'label': self.__vocab[1][m]} for i, v, m in result
              }
            })
        elif self.__task == 'dp_parser':
            return ParserData(
            {
                'Sequence': text, 
                'Inference':{
                    'xpos': [self.__vocab[0][i] for (_, i, v, m) in result], 
                    'head': [v for (_, i, v, m) in result],
                    'rela':[self.__vocab[-1][m] for (_, i, v, m) in result],
                }
            })
        elif self.__task== 'multi':
            return MultiData(
            {
                'Sequence': text, 
                'Inference':{
                    'xpos': [self.__vocab[0][i] for (_, i, k, v, m) in result], 
                    'ner': [self.__vocab[1][k] for (_, i, k, v, m) in result],
                    'head': [v for (_, i, k, v, m) in result],
                    'rela':[self.__vocab[-1][m] for (_, i, k, v, m) in result],
                }
            })


