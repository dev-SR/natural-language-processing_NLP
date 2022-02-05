# Natural Language Text Pre-preprocessing

- [Natural Language Text Pre-preprocessing](#natural-language-text-pre-preprocessing)
  - [Introduction](#introduction)
  - [Tokenization](#tokenization)
  - [Stopwords](#stopwords)
    - [including punctuations](#including-punctuations)
  - [Stemming](#stemming)
  - [POS(part of speech) tagger](#pospart-of-speech-tagger)
  - [Lemmatization](#lemmatization)


```python
"""
cd .\00text-preprocess\
jupyter nbconvert --to markdown pre-process.ipynb --output README.md
"""
import nltk
# nltk.download()
```

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


## POS(part of speech) tagger

We can use `nltk.pos_tag` to retrieve the `part of speech` of each token in a list.

pos_tag **abbreviations**:

- NNP:		proper noun, singular (sarah)
- NNS: noun, common, plural
- NNPS:		proper noun, plural (indians or americans)
- PDT:		predeterminer (all, both, half)
- POS:		possessive ending (parent\ ‘s)
- DT: determiner
- .....

[https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk](https://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk)


```python
from nltk import pos_tag
pos_tag(['any'])
```




    [('any', 'DT')]



> Watch Out: post_tag takes `list` not  `string`

load text:


```python
from nltk.corpus import state_union
# Prerequisite:
# nltk.download('state_union')
```

Most corpora consist of a set of files, each containing a document (or other pieces of text). A list of identifiers for these files is accessed via the `fileids()` method of the corpus reader:


```python
documents = nltk.corpus.state_union.fileids()
print(documents[:20])
```

    ['1945-Truman.txt', '1946-Truman.txt', '1947-Truman.txt', '1948-Truman.txt', '1949-Truman.txt', '1950-Truman.txt', '1951-Truman.txt', '1953-Eisenhower.txt', '1954-Eisenhower.txt', '1955-Eisenhower.txt', '1956-Eisenhower.txt', '1957-Eisenhower.txt', '1958-Eisenhower.txt', '1959-Eisenhower.txt', '1960-Eisenhower.txt', '1961-Kennedy.txt', '1962-Kennedy.txt', '1963-Johnson.txt', '1963-Kennedy.txt', '1964-Johnson.txt']



```python
speech  = state_union.raw('2006-GWBush.txt')
speech[:500]
```




    "PRESIDENT GEORGE W. BUSH'S ADDRESS BEFORE A JOINT SESSION OF THE CONGRESS ON THE STATE OF THE UNION\n \nJanuary 31, 2006\n\nTHE PRESIDENT: Thank you all. Mr. Speaker, Vice President Cheney, members of Congress, members of the Supreme Court and diplomatic corps, distinguished guests, and fellow citizens: Today our nation lost a beloved, graceful, courageous woman who called America to its founding ideals and carried on a noble dream. Tonight we are comforted by the hope of a glad reunion with the hus"




```python
speech_in_words = word_tokenize(speech)
pos = pos_tag(speech_in_words)
pos[:10]
```




    [('PRESIDENT', 'NNP'),
     ('GEORGE', 'NNP'),
     ('W.', 'NNP'),
     ('BUSH', 'NNP'),
     ("'S", 'POS'),
     ('ADDRESS', 'NNP'),
     ('BEFORE', 'IN'),
     ('A', 'NNP'),
     ('JOINT', 'NNP'),
     ('SESSION', 'NNP')]



## Lemmatization

**Lemmatization**: Unlike stemming, lemmatization reduces the words to a word existing in the language.

For lemmatization to resolve a word to its `lemma`, **`part of speech` of the word is required**. This helps in transforming the word into a proper root form. However, for doing so, it requires extra computational linguistics power such as a **part of speech tagger**.

[what-is-the-difference-between-stemming-and-lemmatization/](https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/)

Lemmatization is preferred over Stemming because lemmatization does a morphological analysis of the words.



```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
```


```python
lemmatizer.lemmatize("bats")
```




    'bat'




```python
sentence = "The striped bats are hanging on their feet for best"
# Tokenize: Split the sentence into words
word_list = nltk.word_tokenize(sentence)
print(word_list)
# Lemmatize list of words and join
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)
```

    ['The', 'striped', 'bats', 'are', 'hanging', 'on', 'their', 'feet', 'for', 'best']
    The striped bat are hanging on their foot for best


Notice it didn’t do a good job. Because, `‘are’` is not converted to` ‘be’` and `‘hanging’` is not converted to `‘hang’` as expected. This can be corrected if we provide the correct **‘part-of-speech’ tag (`POS` tag)** as the second argument to `lemmatize()`. Sometimes, the same word can have a multiple lemmas based on the meaning / context.


```python
lemmatizer.lemmatize("painting", pos='n')
```




    'painting'




```python
lemmatizer.lemmatize("painting", pos='v')
```




    'paint'




```python
lemmatizer.lemmatize("hanging", pos='v')

```




    'hang'




```python
lemmatizer.lemmatize("are", pos='v')
```




    'be'




```python
lemmatizer.lemmatize("is", pos='v')
```




    'be'




```python
w = "hanging"
postag = pos_tag([w])
postag

```




    [('hanging', 'VBG')]



Simple Function to convert `pos_tag` abbreviations to simple form that the `lemmatize()` function takes. For example `NN`,`NNS` etc to (`n`), `VBG` etc to `v`


```python
from nltk.corpus import wordnet
def get_simple_pos(tag):

	if tag.startswith("J"):
		return wordnet.ADJ
	elif tag.startswith("V"):
		return wordnet.VERB
	elif tag.startswith("N"):
		return wordnet.NOUN
	elif tag.startswith("R"):
		return wordnet.ADV
	else:
		return wordnet.NOUN

```


```python
w="hanging"
postag = pos_tag([w])
print(postag)
print(postag[0][1]+" --> ",end="" )
pos = get_simple_pos(postag[0][1])
print(pos)

```

    [('hanging', 'VBG')]
    VBG --> v



```python
sentence = "The striped bats are hanging on their feet for best"
word_list = nltk.word_tokenize(sentence)
o = []
for w in word_list:
	postag = pos_tag([w])
	pos = get_simple_pos(postag[0][1])
	clean_word = lemmatizer.lemmatize(w, pos=pos)
	o.append(clean_word)

print(o)


```

    ['The', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'best']


[https://www.machinelearningplus.com/nlp/lemmatization-examples-python/](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/)
