#  Basic Text Classification using Naive Baye

- [Basic Text Classification using Naive Baye](#basic-text-classification-using-naive-baye)
  - [Basic Textual Data Cleaning I - NLP Pipeline](#basic-textual-data-cleaning-i---nlp-pipeline)
  - [Textual Data Cleaning II - Working with Files](#textual-data-cleaning-ii---working-with-files)
  - [Movie Review Prediction - Using Multinomial Naive Bayes](#movie-review-prediction---using-multinomial-naive-bayes)


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


