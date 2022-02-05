# Natural Language Text Pre-preprocessing

- [Natural Language Text Pre-preprocessing](#natural-language-text-pre-preprocessing)
  - [Introduction](#introduction)
  - [Tokenization](#tokenization)
  - [Stopwords](#stopwords)
    - [including punctuations](#including-punctuations)
  - [Stemming](#stemming)
  - [POS Tagger](#pos-tagger)


```python
"""
cd .\00text-preprocess\
jupyter nbconvert --to markdown pre-process.ipynb --output README.md
"""
import nltk
nltk.download()
```

    showing info https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml





    True



## Introduction

Data preprocessing is an essential step in building a Machine Learning model and depending on how well the data has been preprocessed; the results are seen.

In NLP, text preprocessing is the first step in the process of building a model.
The various text preprocessing steps are:

- Tokenization
- Lower casing
- Stop words removal
- Stemming
- Lemmatization

## Tokenization

Tokenization: Splitting the sentence into words.
Strings can be tokenized into tokens via `nltk.word_tokenize`.



```python
from nltk.tokenize import sent_tokenize,word_tokenize
# prerequisite:nltk.download('punkt')
```


```python
sample_text = "Does this thing really work? Lets see."
```


```python
sent_tokenize(sample_text)
```




    ['Does this thing really work?', 'Lets see.']




```python
words = word_tokenize(sample_text)
words

```




    ['Does', 'this', 'thing', 'really', 'work', '?', 'Lets', 'see', '.']



## Stopwords

Stop words removal: Stop words are very commonly used words (**a, an, the, etc.**) in the documents. These words do not really signify any importance as they do not help in distinguishing two documents. We can use `nltk.corpus.stopwords.words(‘english’)` to fetch a list of `stopwords` in the English dictionary. Then, we remove the tokens that are `stopwords`.



```python
from nltk.corpus import stopwords
```


```python
stop = stopwords.words('english')

print(stop[:20])
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his']



```python
clean_words = [w for w in words if w not in stop ]
print(words)
print(clean_words)

```

    ['Does', 'this', 'thing', 'really', 'work', '?', 'Lets', 'see', '.']
    ['Does', 'thing', 'really', 'work', '?', 'Lets', 'see', '.']


> !! **Watch out for Uppercase**: for example `this` in the above got removed as it is a stopword. But if we would have used `This`, it will not be removed


```python
sample_text = "Does This thing really work? Lets see."
words = word_tokenize(sample_text)
words
```




    ['Does', 'This', 'thing', 'really', 'work', '?', 'Lets', 'see', '.']




```python
clean_words = [w for w in words if w not in stop]
print(words)
print(clean_words)

```

    ['Does', 'This', 'thing', 'really', 'work', '?', 'Lets', 'see', '.']
    ['Does', 'This', 'thing', 'really', 'work', '?', 'Lets', 'see', '.']


> Solution:


```python
sample_text = "Does This thing really work? Lets see."
sample_text = sample_text.lower()
words = word_tokenize(sample_text)
words
```




    ['does', 'this', 'thing', 'really', 'work', '?', 'lets', 'see', '.']




```python
clean_words = [w for w in words if w not in stop]
print(words)
print(clean_words)

```

    ['does', 'this', 'thing', 'really', 'work', '?', 'lets', 'see', '.']
    ['thing', 'really', 'work', '?', 'lets', 'see', '.']


### including punctuations


```python
import string
punctuations = list(string.punctuation)
stop = stop + punctuations
```


```python
clean_words = [w for w in words if w not in stop]
print(words)
print(clean_words)
```

    ['does', 'this', 'thing', 'really', 'work', '?', 'lets', 'see', '.']
    ['thing', 'really', 'work', 'lets', 'see']


## Stemming

Stemming: It is a process of transforming a word to its root form.
We stem the tokens using `nltk.stem.porter.PorterStemmer` to get the stemmed tokens.


```python
from nltk.stem import PorterStemmer
ps = PorterStemmer()

words = ["play","playing","player","played"]

stemmed_words = [ ps.stem(w) for w in words]
stemmed_words
```




    ['play', 'play', 'player', 'play']




```python
words = ["machine","happying"]

stemmed_words = [ps.stem(w) for w in words]
stemmed_words
```




    ['machin', 'happi']



Explanation: The word `'machine'` has its suffix `'e'` chopped off. The stem does not make sense as it is not a word in English. This is a disadvantage of stemming.


## POS Tagger


