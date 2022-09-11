from deepnlp.utils.tokenizer import word_tokenize, sentence_tokenize
from deepnlp.preprocess_load import (
    download,
    load_model, 
    load_vocabs, 
    set_gpus, 
    download_model, 
    download_vocabs,
    clear_cache_model, 
    clear_cache_vocabs,
    clear_cache, 
)

from deepnlp.utils.print_out import print_out
from deepnlp.inference_pipline import (
    MultiTask,
    PosTagger,
    NerTagger,
    DPParser,
    pipline
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
    'clear_cache',
    'MultiTask',
    'PosTagger',
    'NerTagger',
    'DPParser',
    'pipline',
    'print_out'
]