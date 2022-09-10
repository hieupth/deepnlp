import re 
import gdown
import os
import shutil
import pickle as pkl 
import tensorflow as tf 
from deepnlp.constant import (
    model_url, 
    vocabs_url, 
    vocabs_save, 
    model_save 
)

def handler(func, path, exc_info):
    print("Inside handler")
    print(exc_info)

def set_gpus(gpu_ids_list):
    # if gpu_ids_list == [] ==> using cpu 
    gpus= tf.config.list_physical_devices('GPU')
    if gpus: 
        try:
            gpus_used= [gpus[i] for i in gpu_ids_list]
            tf.config.set_visible_devices(gpus_used, 'GPU')

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return False 
    return True
        
def print_out(text):
    # for dependency parsing
    pass

def ensure_model_name(model_name):
    assert model_name in model_url, "Not found model_name"
    return model_url[model_name]

def ensure_vocabs_name(vocab_name):
    assert vocab_name in vocabs_url, "Not found vocab_name"
    return vocabs_url[vocab_name]
        
    
def download_model(model_name, required= True):
    url_model= ensure_model_name(model_name)
    path_model= os.path.join(model_save, model_name)
    
    if not ensure_dir(path_model) and required: 
        print('=' * 10 + ' Install Pretrained ' + '=' * 10)
        gdown.download_folder(url_model, output= path_model)

def download_vocabs(vocab_name):
    url_vocabs= ensure_vocabs_name(vocab_name)
    path_vocabs= os.path.join(vocabs_save, vocab_name)
    
    if not ensure_dir(path_vocabs): 
        print('=' * 10 + ' Install Vocabs ' + '=' * 10)
        gdown.download_folder(url_vocabs, output= path_vocabs)

def download(model_name= 'deepnlp_eng', vocab_name= 'deepnlp_eng'):
    download_model(model_name)
    download_vocabs(vocab_name)

def clear_cache_model(model_name):
    ensure_model_name(model_name)
    path_model= os.path.join(model_save, model_name)
    shutil.rmtree(path_model, onerror= handler)

def clear_cache_vocabs(vocab_name):
    ensure_vocabs_name(vocab_name)
    path_vocabs= os.path.join(vocabs_save, vocab_name)
    shutil.rmtree(path_vocabs, onerror= handler)

def clear_cache(name):
    clear_cache_model(name)
    clear_cache_vocabs(name)


def load_model(model_name):
    ensure_model_name(model_name)
    path_model= os.path.join(model_save, model_name)
    return tf.keras.models.load_model(path_model, compile= False)

    

def load_vocabs(vocab_name, *,  task= 'pos'):
    ensure_vocabs_name(vocab_name)
    path_vocabs= os.path.join(vocabs_save, vocab_name)
    assert task in ['pos', 'ner', 'dp', 'multi']
    if task== 'multi':
        # pos load
        with open(os.path.join(path_vocabs, 'pos.pkl'), 'rb') as handel: 
            pos= pkl.load(handel)
        # ner load
        with open(os.path.join(path_vocabs, 'ner.pkl'), 'rb') as handel: 
            ner= pkl.load(handel)
        # dp load
        with open(os.path.join(path_vocabs, 'dp.pkl'), 'rb') as handel: 
            dp= pkl.load(handel)

        return pos, ner, dp 
    else: 
        with open(os.path.join(path_vocabs, f'{task}.pkl'), 'rb') as handel: 
            vocab= pkl.load(handel)
        return vocab

