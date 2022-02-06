#  Basic Text Classification using Naive Baye

- [Basic Text Classification using Naive Baye](#basic-text-classification-using-naive-baye)
  - [Basic Textual Data Cleaning I - NLP Pipeline](#basic-textual-data-cleaning-i---nlp-pipeline)
  - [Textual Data Cleaning II - Working with Files](#textual-data-cleaning-ii---working-with-files)
  - [Movie Review Prediction - Using Multinomial Naive Bayes](#movie-review-prediction---using-multinomial-naive-bayes)
    - [1. Cleaning](#1-cleaning)
    - [2. Vectorization](#2-vectorization)
    - [3. Multinomial Naive Bayes](#3-multinomial-naive-bayes)
    - [Experimenting](#experimenting)

```python
"""
cd .\01basic-classification-nb\
jupyter nbconvert --to markdown tc_nb.ipynb --output README.md
"""
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
```

## Basic Textual Data Cleaning I - NLP Pipeline


```python
sample_text = """I loved this movie <br /><br /> since I was 7 and I saw it on the opening day. It was so touching and beautiful. I strongly recommend seeing for all. It's a movie to watch with your family by far.<br /><br />My MPAA rating: PG-13 for thematic elements, prolonged scenes of disastor, nudity/sexuality and some language."""

tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()
```


```python
def getCleanReview(review):

    review = review.lower()
    review = review.replace("<br /><br />", " ")

    # Tokenize
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]

    cleaned_review = ' '.join(stemmed_tokens)

    return cleaned_review

```


```python
getCleanReview(sample_text)

```




    'love movi sinc 7 saw open day touch beauti strongli recommend see movi watch famili far mpaa rate pg 13 themat element prolong scene disastor nuditi sexual languag'



## Textual Data Cleaning II - Working with Files


```python
def getStemmedDocument(inputFile):

    outputFile = inputFile.replace(".txt", "_stemmed.txt")
    out = open(outputFile, 'w', encoding="utf8")

    with open(inputFile, encoding="utf8") as f:
        reviews = f.readlines()

    for review in reviews:
        cleaned_review = getCleanReview(review)
        print((cleaned_review), file=out)

    out.close()

```


```python
getStemmedDocument("imdb_toy_x.txt")

```

> Running in CLI

```
cd .\01basic-classification-nb\
conda activate base
python clean_reviews.py imdb_toy_x.txt
```


## Movie Review Prediction - Using Multinomial Naive Bayes

### 1. Cleaning


```python

x = ["This was an awesome movie",
	"Great movie! I liked it a lot",
	"Happy Ending! awesome acting by the hero",
	"loved it! truly great",
	"bad not upto the mark",
	"could have been better",
	"Surely a Disappointing movie"
]

y = [1,1,1,1,0,0,0] # 1 - Positive, 0 - Negative Class
```


```python
x_test = ["I was happy & happy and I loved the acting in the movie",
		"The movie I saw bad"]
```


```python
X_train_clean = [getCleanReview(i) for i in x]
X_test_clean = [getCleanReview(i) for i in x_test]
```


```python
X_train_clean, X_test_clean

```




    (['awesom movi',
      'great movi like lot',
      'happi end awesom act hero',
      'love truli great',
      'bad upto mark',
      'could better',
      'sure disappoint movi'],
     ['happi happi love act movi', 'movi saw bad'])



### 2. Vectorization


```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
```


```python
x_vec = cv.fit_transform(X_train_clean).toarray()
print(x_vec)
print(x_vec.shape)

```

    [[0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0]
     [0 0 0 0 0 0 0 1 0 0 1 1 0 0 1 0 0 0]
     [1 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0]
     [0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1]
     [0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 0]]
    (7, 18)



```python
print(cv.get_feature_names())
```

    ['act', 'awesom', 'bad', 'better', 'could', 'disappoint', 'end', 'great', 'happi', 'hero', 'like', 'lot', 'love', 'mark', 'movi', 'sure', 'truli', 'upto']


> Watch Out on Vectorization of Test Data using `fit_transform`


```python
x_test_vec = cv.fit_transform(X_test_clean).toarray()
print(x_test_vec.shape)
print(cv.get_feature_names())
```

    (2, 6)
    ['act', 'bad', 'happi', 'love', 'movi', 'saw']



```python
x_vec = cv.fit_transform(X_train_clean).toarray()
x_test_vec = cv.transform(X_test_clean).toarray()
print(x_test_vec.shape)

```

    (2, 18)


[Why-we-use-`fit_transform()-`on-**training**-data-but-`transform()`-on-the-**test**-data?](https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe)


[stackoverflow/what-is-the-difference-between-transform-and-fit-transform-in-sklearn](https://stackoverflow.com/questions/23838056/what-is-the-difference-between-transform-and-fit-transform-in-sklearn)


Using the `transform` method we can use the same `mean` and `variance` as it is calculated from our **training** data to transform our test data. Thus, the parameters learned by our model using the training data will help us to transform our test data.

Generic difference between the methods:

- `fit(raw_documents[, y])`: Learn a vocabulary dictionary of all tokens in the raw documents.
- `fit_transform(raw_documents[, y]):` Learn the vocabulary dictionary and return term-document matrix. This is equivalent to fit followed by the transform, but more efficiently implemented.
- `transform(raw_documents)`: Transform documents to document-term matrix. Extract token counts out of raw text documents using the vocabulary fitted with fit or the one provided to the constructor(from our **training** data).
- Both fit_transform and transform returns the same, Document-term matrix.


### 3. Multinomial Naive Bayes


```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
```


```python
mnb = MultinomialNB()
```


```python
# Train the model using the training sets
mnb.fit(x_vec, y)
```




    MultinomialNB()




```python
mnb.predict(x_test_vec)
```




    array([1, 0])



`[1,0]` = `[Positive Review , Negative Review]`


```python
mnb.score(x_vec, y)

```




    1.0



### Experimenting


```python
x_test = ["I was happy & happy and I loved the acting in the movie",
		"The movie I saw not bad"]
```


```python
cv = CountVectorizer()
# fit_transform() is must other wise `transform()` will not work
x_vec = cv.fit_transform(X_train_clean).toarray()
print(x_vec.shape)

X_test_clean = [getCleanReview(i) for i in x_test]
print(X_test_clean)

x_test_vec = cv.transform(X_test_clean).toarray()
print(x_test_vec.shape)
# print(cv.get_feature_names())

```

    (7, 18)
    ['happi happi love act movi', 'movi saw bad']
    (2, 18)


`Stopwords` removed `not` . We can see the after `stopwords` the negative reviews also changed to positive. A bit scary right?






```python
mnb.fit(x_vec, y)
```




    MultinomialNB()




```python
mnb.predict(x_test_vec)
```




    array([1, 0])



`[1,0]` = `[Positive Review , Negative Review]` but last one should be also `Positive` Review.

[https://dev.to/sunilaleti/don-t-blindly-remove-stopwords-in-sentiment-analysis-3nok](https://dev.to/sunilaleti/don-t-blindly-remove-stopwords-in-sentiment-analysis-3nok)
