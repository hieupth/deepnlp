# DeepNLP
This is a new natural language processing library based on modern deep learning methods. The library focus on basic NLP tasks such as: POS (part of speech), NER (named entity recognition) and DP (dependency parsing). The main language is English but we are working hard to support Vietnamese and others in the near future.

## Installation 🔥
- This repository is tested on python 3.7+ and Tensorflow 2.8+
- Deepnlp can be installed using pip as follows: 
```
pip install deepnlp-cerelab 
```
- Deepnlp can also be installed from source with the following commands: 
```
git clone https://github.com/hieupth/deepnlp.git
cd deepnlp/
pip install -e .
```
## Tutorials 🥮

- [1. Sentence Segmentation](#sentence_tokenize)
- [2. Word Tokenizer](#word_tokenize)
- [3. Install and load pretrained model and vocabs](#pretrained)
- [4. POS Tagging](#xpos)
- [5. Named Entity Recognition](#ner)
- [6. Dependency Parsing](#parser)
- [7. Multil Task](#multi)
- [8. Clear Cache](#cache)
- [9. List of pretrained models](#list_pretrained)

### 1. Sentence Segmentation
<a name= 'sentence_tokenize'></a>
Usage

```python
>>> import deepnlp 
>>> text = """\
Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e. he paid a lot for it. Did he mind? Adam Jones Jr. thinks he didn't. In any case, this isn't true... Well, with a probability of .9 it isn't.
"""
>>> deepnlp.sentence_tokenize(text)
['Mr. Smith bought cheapsite.com for 1.5 million dollars, i.e. he paid a lot for it.',
 'Did he mind?',
 "Adam Jones Jr. thinks he didn't.",
 "In any case, this isn't true...",
 "Well, with a probability of .9 it isn't.",
 '']
```
### 2. Word Tokenize
Usage
<a name= 'word_tokenize'> </a>

```python
>>> import deepnlp 
>>> text = "I have an apple."
>>> deepnlp.word_tokenize(text)
['I', 'have', 'an', 'apple', '.']
```
### 3. Install and load pretrained model and vocabs 
- Install pretrained model and vocabs
<a name= 'pretrained'></a>

```python 
>>> import deepnlp
>>> deepnlp.download('deepnlp_eng')
```
- Or you can also install pretrained model and vocabs independently of each other 

```python
>>> import deepnlp 
>>> deepnlp.download_model('deepnlp_eng')
>>> deepnlp.download_vocabs('deepnlp_eng')
```
- Load models and vocabs 

```python
>>> import deepnlp 
>>> model = deepnlp.load_model('deepnlp_eng')
>>> vocabs= deepnlp.load_vocabs('deepnlp_eng', task= 'multi') # pos, ner, dp
```
### 4. POS Tagging
<a name= 'xpos'></a>

- With `PosTagger` class 
```python
>>> import deepnlp
>>> model= deepnlp.PosTagger('deepnlp_eng')
>>> model 
model_name: deepnlp_eng, vocab_name: deepnlp_eng, tokenizer_name: distilroberta-base
>>> output= model.inference('I have an apple.', device= 'cpu') # default device = 'cpu'
>>> output
<deepnlp.utils.data_struct.TokenClassificationData at 0x7fbc3ddbab90>
>>> output.value()
{'Sequence': 'I have an apple.',
 'Inference': {'I': {'score': 0.9175689, 'label': 'PRP'},
  'have': {'score': 0.9232193, 'label': 'VBP'},
  'an': {'score': 0.9158458, 'label': 'DT'},
  'apple': {'score': 0.86957675, 'label': 'NN'},
  '.': {'score': 0.8892631, 'label': '.'}}}
>>> deepnlp.print_out([output])
I have an apple.
1	I	PRP
2	have	VBP
3	an	DT
4	apple	NN
5	.       .
```
- With `pipeline` class

```python
>>> import deepnlp 
>>> model= deepnlp.load_model('deepnlp_eng')
>>> pipeline= deepnlp.pipeline(model, task= 'pos_tagger')
>>> output= pipeline("I have an apple.", device= 'cpu') # default device = 'cpu'
>>> deepnlp.print_out([output])
I have an apple.
1	I	PRP
2	have	VBP
3	an	DT
4	apple	NN
5	.       .
```
### 5. Named Entity Recognition
<a name= 'ner'></a>
With `NerTagger` class 

```python 
>>> import deepnlp
>>> model = deepnlp.NerTagger('deepnlp_eng')
>>> output= model.inference('Please confirm your song choice: Same Old War, playing on the kitchen speaker', device= 'cpu') # default device = 'cpu'
output
<deepnlp.utils.data_struct.TokenClassificationData at 0x7f69d9504750>
>>> output.value()
{'Sequence': 'Please confirm your song choice: Same Old War, playing on the kitchen speaker',
 'Inference': {'Same': {'score': 0.922773, 'label': 'B-MISC'},
  'Old': {'score': 0.9353856, 'label': 'I-MISC'},
  'War': {'score': 0.92017937, 'label': 'I-MISC'}}}
>>> deepnlp.print_out([output], del_prefix_ner= False) # if you set del_prefix_ner= True, B-MISC or I-MISC will become MISC 
Please confirm your song choice: Same Old War, playing on the kitchen speaker
1	Please	    O
2	confirm	    O
3	your	    O
4	song        O
5	choice 	    O
6	Same	    B-MISC
7	Old	    I-MISC
8	War	    I-MISC
9	,	    O
10	playing	    O
11	on	    O
12	the	    O
13	kitchen	    O
14	speaker	    O
```
With `pipeline` class 

```python
>>> import deepnlp
>>> model= deepnlp.load_model('deepnlp_eng')
>>> pipeline= deepnlp.pipeline(model, task= 'ner_tagger')
>>> output= pipeline("Please confirm your song choice: Same Old War, playing on the kitchen speaker") 
>>> deepnlp.print_out([output], del_prefix_ner= True, device= 'cpu') # default device = 'cpu'
Please confirm your song choice: Same Old War, playing on the kitchen speaker
1	Please	    O
2	confirm	    O
3	your	    O
4	song        O
5	choice 	    O
6	Same	    MISC
7	Old	    MISC
8	War	    MISC
9	,	    O
10	playing	    O
11	on	    O
12	the	    O
13	kitchen	    O
14	speaker	    O
```
### 6. Dependency Parsing 
<a name= 'parser'></a>
With `DPParser` class

```python 
>>> import deepnlp
>>> model = deepnlp.DPParser('deepnlp_eng')
>>> output= model.inference("I have an apple.", device= 'cpu') # default device = 'cpu'
>>> output 
<deepnlp.utils.data_struct.ParserData at 0x7f69da3125d0>
>>> output.value()
{'Sequence': 'I have an apple.',
 'Inference': {'xpos': ['PRP', 'VBP', 'DT', 'NN', '.'],
  'head': [3, 0, 5, 3, 3],
  'rela': ['nsubj', 'root', 'det', 'obj', 'punct']}}
>>> deepnlp.print_out([output])
I have an apple.
1	I	    PRP	  3	  nsubj
2	have	VBP	  0	  root
3	an	    DT	  5	  det
4	apple	NN	  3	  obj
5	.	    .	  3	  punct
```
With `pipeline` class 

```python 
>>> import deepnlp
>>> model= deepnlp.load_model('deepnlp_eng')
>>> pipeline= deepnlp.pipeline(model, task= 'dp_parser')
>>> output= pipeline("I have an apple.", device= 'cpu') # default device = 'cpu'
>>> deepnlp.print_out([output])
I have an apple.
1	I	    PRP	  3	  nsubj
2	have	    VBP	  0	  root
3	an	    DT	  5	  det
4	apple	    NN	  3	  obj
5	.	    .	  3	  punct
```

### 7. Multi Task 
<a name= 'multi'></a>
With `MultiTask`

```python
>>> import deepnlp
>>> model = deepnlp.MultiTask('deepnlp_eng')
>>> output= model.inference("Please confirm your song choice: Same Old War, playing on the kitchen speaker", device= 'cpu') # default device = 'cpu'
>>> output 
<deepnlp.utils.data_struct.MultiData at 0x7f69da8f7650>
>>> deepnlp.print_out([output])
Please confirm your song choice: Same Old War, playing on the kitchen speaker
1	Please	  UH	O	3	discourse
2	confirm	  VB	O	0	root
3	your	  PRP$	O 	6	nmod:poss
4	song	  NN	O	6	compound
5	choice	  NN	O 	3	obj
6	Same	  JJ	MISC	9	amod
7	Old	  NNP	MISC	9	compound
8	War	  NNP	MISC	3	obj
9	,	  ,	O	3	punct
10	playing	  VBG	O	3	advcl
11	on	  IN	O	15	case
12	the	  DT	O	15	det
13	kitchen	  NN	O	15	compound
14	speaker   NN	O	11	obl
```
With `pipeline` 

```python
>>> import deepnlp 
>>> model= deepnlp.load_model('deepnlp_eng')
>>> pipeline= deepnlp.pipeline(model, task= 'multi')
>>> output= pipeline("I have an apple.", device= 'cpu') # default device = 'cpu'
>>> deepnlp.print_out([output])
Please confirm your song choice: Same Old War, playing on the kitchen speaker
1	Please	  UH	O	3	discourse
2	confirm	  VB	O	0	root
3	your	  PRP$	O 	6	nmod:poss
4	song	  NN	O	6	compound
5	choice	  NN	O 	3	obj
6	Same	  JJ	MISC	9	amod
7	Old	  NNP	MISC	9	compound
8	War	  NNP	MISC	3	obj
9	,	  ,	O	3	punct
10	playing	  VBG	O	3	advcl
11	on	  IN	O	15	case
12	the	  DT	O	15	det
13	kitchen	  NN	O	15	compound
14	speaker   NN	O	11	obl
```
### 8. Clear Cache
<a name= 'cache'></a>

- Remove pretrained model and vocabs `deepnlp_eng`
```python
>>> deepnlp.clear_cache('deepnlp_eng')
```
- Or
```python
>>> deepnlp.clear_model('deepnlp_eng')
>>> deepnlp.clear_vocabs('deepnlp_eng') 
```

### 9. List of pretrained models
<a name= 'list_pretrained'></a>
- `deppnlp_eng` - support for English:  <a href= 'https://drive.google.com/drive/folders/1ub0T9T70lcrAq5C3fH3fy8QqCgkYSZlm?usp=sharing'>download pretrained model</a> - <a href= 'https://drive.google.com/drive/folders/1SS7ra-xnaAQ2Y5KeR5ulQAqu-OtPKbhJ?usp=sharing'>download vocabs </a>
- `deepnlp_vie` - support for Vietnamese: Will be updated in the future
## License
[Apache 2.0 License](https://github.com/hieupth/deepnlp). <br>
Copyright &copy; 2022 [Hieu Pham](https://github.com/hieupth). All rights reserved.
