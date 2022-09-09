from deepnlp.preprocess_load import (
    download,
    load_model, 
    word_tokenize, 
    sentence_tokenize, 
    load_vocabs, 
    set_gpus, 
    download_model, 
    download_vocabs,
    clear_cache_model, 
    clear_cache_vocabs,
    clear_cache, 

)

__version__= "1.0.1"
__all__ = [
    'download',
    'load_model',
    'load_vocabs',
    'word_tokenize',
    'sentence_tokenize',
    'set_gpus', 
    'download_model', 
    'download_vocabs', 
    'clear_cache_model', 
    'clear_cache_vocabs', 
    'clear_cache'
]