import re 
import gdown
import os 
import tensorflow as tf 
from deepnlp.constant import model_url, vocabs_url, vocabs_save, model_save

def word_tokenize(text):
    text= re.findall(r"[\w'\"]+|[,.!?]", text)
    return text 

def sentence_tokenize(text):
    sentences= re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    return sentences

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return False 
    return True
        
def print_out(text):
    pass

def ensure_model_name(model_name):
    assert model_name in model_url, "Not found model_name"
    return model_url[model_name]

def ensure_vocabs_name(vocab_name):
    assert vocab_name in vocabs_url, "Not found vocab_name"
    return vocabs_url[vocab_name]
        
    

def download(model_name= 'deepnlp_eng', vocab_name= 'deepnlp_eng'):
    url_model= ensure_model_name(model_name)
    url_vocabs= ensure_vocabs_name(vocab_name)

    path_model= os.path.join(model_save, model_name)
    path_vocabs= os.path.join(vocabs_save, vocab_name)

    
    if not ensure_dir(path_model): 
        print('=' * 10 + 'Install Pretrained' + '=' * 10)
        gdown.download_folder(url_model, output= path_model)

    if not ensure_dir(path_vocabs):
        print('=' * 10 + 'Install Vocabs' + '=' * 10)
        gdown.download_folder(url_vocabs, output= path_vocabs)

def load_model():
    pass

