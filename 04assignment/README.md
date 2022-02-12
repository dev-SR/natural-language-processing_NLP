# Assignment-1

<div align="center">
<img src="img/asgn.jpg" alt="asgn.jpg" width="1000px">
</div>

- [Assignment-1](#assignment-1)
  - [Load Data](#load-data)
    - [Processing file - test codes](#processing-file---test-codes)
    - [Processing file - Final](#processing-file---final)
    - [Label Encoding](#label-encoding)
  - [Data Pre-Process](#data-pre-process)
  - [Split the data into training and test sets](#split-the-data-into-training-and-test-sets)
  - [Vectorization: Convert text features to numeric](#vectorization-convert-text-features-to-numeric)
  - [Train the model](#train-the-model)
  - [Hyperparameter tuning: Searching for the ideal model](#hyperparameter-tuning-searching-for-the-ideal-model)
    - [Plotting each models performance](#plotting-each-models-performance)
  - [Evaluating the model and make predictions using the Best Model](#evaluating-the-model-and-make-predictions-using-the-best-model)
  - [Results](#results)


```python
"""
cd .\04assignment\
jupyter nbconvert --to markdown asng.ipynb --output README.md
"""
```

## Load Data

### Processing file - test codes


```python
with open('38943-0.txt', 'r') as target_file:
	for num, line in enumerate(target_file.readlines()):
		if "Title" in line:
			title = line.split(":")[1].strip()
			print(title)
		if "Author" in line:
			author_name = line.split(":")[1].strip()
			print(author_name)
author_list = [author_name]*5
author_list
```

    Science and Medieval Thought
    Sir Thomas Clifford Allbutt





    ['Sir Thomas Clifford Allbutt',
     'Sir Thomas Clifford Allbutt',
     'Sir Thomas Clifford Allbutt',
     'Sir Thomas Clifford Allbutt',
     'Sir Thomas Clifford Allbutt']




```python
out = open("64171-0_chapter.txt", "w")
with open('64171-0.txt', 'r') as f:
	data = f.read()
	if "CHAPTER I." in data:
		data = data.replace("CHAPTER I.", "CHAPTERI.")
		print(data, file=out)
out.close()

```


```python
list_of_blocks = []
with open('64171-0_chapter.txt', 'r') as f:
	data = f.read()
	# print(len(data.split()))
	total_word = len(data.split())
	data = data.split()
	# indices = []
	# for i, w in enumerate(data):
	# 	# start reading from CHAPTER I.
	# 	if "CHAPTERI" in w:
	# 		indices.append(i)
	# last = indices[-1]
	# # print(last)
	# data = data[last+1:]

	n = 200
	for i in range(n):
		s = round(total_word*i/n)
		e = round(total_word*(1+i)/n)
		# print(s,e)
		chunks = data[s:e]
		chunks = " ".join(chunks)
		# print(chunks)
		# print(100*"=")
		list_of_blocks.append(chunks)

# print(list_of_blocks)

```

### Processing file - Final


```python
import os
def getTextFileList():
	all_files = os.listdir()
	txt_files = []
	for file in all_files:
		if file.endswith(".txt"):
			txt_files.append(file)
	return txt_files
```


```python
from termcolor import cprint
def processFiles(text_files,document_no=200):
	df_obj_of_documents = {
		"documents":[],
		"author":[]
	}
	list_of_documents = []
	list_of_authors = []
	for file in text_files:
		cprint(f"Extracting from: {file}", 'green')
		# get Title and Author name
		author_name=""
		with open(file, 'r') as target_file:
			for num, line in enumerate(target_file.readlines()):
				pass
				if "Title:" in line:
					title = line.split(":")[1].strip()
				if "Author:" in line:
					print(line)
					author_name = line.split(":")[1].strip()

		# get documents from each file
		with open(file, 'r') as f:
			data = f.read()
			total_word = len(data.split())
			data = data.split()
			# indices = []
			# for i, w in enumerate(data):
			# 	# start reading from CHAPTER I.
			# 	if "CHAPTERI" in w:
			# 		indices.append(i)
			# last = indices[-1]
			# # print(last)
			# data = data[last+1:]

			# n = 200
			for i in range(document_no):
				start_from = round(total_word*i/document_no)
				end_at = round(total_word*(1+i)/document_no)
				document = data[start_from:end_at]
				document = " ".join(document)
				# print(chunks)
				# print(100*"=")
				list_of_documents.append(document)

		print("Book Title: ",end="")
		cprint(title, 'yellow')
		print("Author: ",end="")
		cprint(author_name, 'yellow')
		# generated_doc_size = len(list_of_documents)
		print("Documents generated: ",end="")
		cprint(document_no, 'yellow')
		repeating_author = [author_name]*document_no
		# print(repeating_author)
		list_of_authors = list_of_authors + repeating_author
		# save to dataframe
	df_obj_of_documents["author"] = list_of_authors
	df_obj_of_documents["documents"] = list_of_documents
	# print(df_obj_of_documents)
	print(len(list_of_authors))
	print(len(list_of_documents))
	return df_obj_of_documents

```

```pseudocode
fil1
	doc 1 - a
	doc 2 - a
	doc 3 - a
fil2
	doc 1 - b
	doc 2 - b
	doc 3 - b

doc 	= [doc 1, doc 2, doc 3,doc 1 ,doc 2,doc 3]
author 	= [a,a,a,b ,b,b]
```


```python
df_obj_of_documents = processFiles(getTextFileList(),document_no=200)
```

    [32mExtracting from: 38943-0.txt[0m
    Author: Sir Thomas Clifford Allbutt

    Book Title: [33mScience and Medieval Thought[0m
    Author: [33mSir Thomas Clifford Allbutt[0m
    Documents generated: [33m200[0m
    [32mExtracting from: 64171-0.txt[0m
    Author: Robert C. (Chamblet) Adams

    Book Title: [33mOn Board the "Rocket"[0m
    Author: [33mRobert C. (Chamblet) Adams[0m
    Documents generated: [33m200[0m
    [32mExtracting from: 65708-0.txt[0m
    Author: Averroes

    Book Title: [33mThe Philosophy and Theology of Averroes[0m
    Author: [33mAverroes[0m
    Documents generated: [33m200[0m
    [32mExtracting from: pg24055.txt[0m
    Author: Confucius

    Book Title: [33mThe Sayings Of Confucius[0m
    Author: [33mConfucius[0m
    Documents generated: [33m200[0m
    [32mExtracting from: pg2412.txt[0m
    Author: Aristotle

    Book Title: [33mThe Categories[0m
    Author: [33mAristotle[0m
    Documents generated: [33m200[0m
    [32mExtracting from: pg34283.txt[0m
    Author: Alfred William Benn

    Book Title: [33mHistory of Modern Philosophy[0m
    Author: [33mAlfred William Benn[0m
    Documents generated: [33m200[0m
    [32mExtracting from: pg66566.txt[0m
    Author: T.J. de Boer

    Book Title: [33mThe History of Philosophy in Islam[0m
    Author: [33mT.J. de Boer[0m
    Documents generated: [33m200[0m
    1400
    1400



```python
import pandas as pd
# df = pd.DataFrame({
# 	'data': list_of_blocks,
# 	"author": author_list
# })
df = pd.DataFrame(df_obj_of_documents)
df.head()
# df.to_csv("data.csv", index=False)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>documents</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>﻿The Project Gutenberg eBook, Science and Medi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SCIENCE AND MEDIEVAL THOUGHT. * * * * * London...</td>
      <td>Sir Thomas Clifford Allbutt</td>
    </tr>
    <tr>
      <th>2</th>
      <td>et facit nos concludere quæstionem, sed non ce...</td>
      <td>Sir Thomas Clifford Allbutt</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a phantom, and again the spirit of a new world...</td>
      <td>Sir Thomas Clifford Allbutt</td>
    </tr>
    <tr>
      <th>4</th>
      <td>first applied to the art and romance of the Mi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['words_count'] = df['documents'].apply(lambda x: len(x.split()))
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>documents</th>
      <th>author</th>
      <th>words_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>﻿The Project Gutenberg eBook, Science and Medi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SCIENCE AND MEDIEVAL THOUGHT. * * * * * London...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>143</td>
    </tr>
    <tr>
      <th>2</th>
      <td>et facit nos concludere quæstionem, sed non ce...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a phantom, and again the spirit of a new world...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
    </tr>
    <tr>
      <th>4</th>
      <td>first applied to the art and romance of the Mi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>143</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['words_count'].min()
```




    87




```python
# df[df["author"]=="Aristotle"]
```


```python
df = df.drop(df[df['words_count'] < 100].index)
# drop documents with less than 100 words
```


```python
df['words_count'].min()
```




    142



### Label Encoding


```python
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

```


```python
df['label'] = labelencoder.fit_transform( df['author'])
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>documents</th>
      <th>author</th>
      <th>words_count</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>﻿The Project Gutenberg eBook, Science and Medi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SCIENCE AND MEDIEVAL THOUGHT. * * * * * London...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>143</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>et facit nos concludere quæstionem, sed non ce...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a phantom, and again the spirit of a new world...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>first applied to the art and romance of the Mi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>143</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shuffle data
df = df.sample(frac=1).reset_index(drop=True)
df.head()

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>documents</th>
      <th>author</th>
      <th>words_count</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>people tremble.[32] [Footnote 32: _Tremble_ an...</td>
      <td>Confucius</td>
      <td>173</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pushed back into efficient reason or divine wi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>in the interpretation itself. Amongst these ma...</td>
      <td>Averroes</td>
      <td>278</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>highest degree of perfection and an eternal pl...</td>
      <td>T.J. de Boer</td>
      <td>320</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>also the masts and yards, and wearing away the...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>395</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



## Data Pre-Process


```python
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

```


```python
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REMOVE_NUM = re.compile('[\d+]')
EMAIL_RE = re.compile('\b[\w\-.]+?@\w+?\.\w{2,4}\b')
PHONE_RE=re.compile('\b(\+\d{1,2}\s?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b')
NUMBER_RE=re.compile('\d+(\.\d+)?')
URLS_RE = re.compile('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)')
PUNCTUATION_RE = re.compile('[^\w\s]')
EXTRA_SPACE_RE = re.compile('\s+')
STOPWORDS = set(stopwords.words('english'))

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

def cleanText(msg):
	msg = REPLACE_BY_SPACE_RE.sub(' ',msg)
	msg = BAD_SYMBOLS_RE.sub('',msg)
	msg = EMAIL_RE.sub('',msg)
	msg = URLS_RE.sub('',msg)
	msg = PHONE_RE.sub('',msg)
	msg = NUMBER_RE.sub('',msg)
	msg = PUNCTUATION_RE.sub('',msg)
	msg = EXTRA_SPACE_RE.sub(' ',msg)

	# Lower case
	msg = msg.lower()

	# Tokenize
	words = word_tokenize(msg)

	# Remove Stop Words
	words = [w for w in words if not w in STOPWORDS]

	# Stemming
	stemmed_words = [ps.stem(w) for w in words]

	# Lemmatization
	# lemmatized_words = [lemmatizer.lemmatize(w, get_simple_pos(pos_tag([w])[0][1])) for w in words]
	# for w in words:
	# 	postag = pos_tag([w])
	# 	pos = get_simple_pos(postag[0][1])
	# 	clean_word = lemmatizer.lemmatize(w, pos=pos)
	# 	lemmatized_words.append(clean_word)

	# Join the words back into one string separated by space,
	stemmed_sen = ' '.join(stemmed_words)

	return stemmed_sen

```


```python
df['clean_msg'] = df.documents.apply(cleanText)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>documents</th>
      <th>author</th>
      <th>words_count</th>
      <th>label</th>
      <th>clean_msg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>people tremble.[32] [Footnote 32: _Tremble_ an...</td>
      <td>Confucius</td>
      <td>173</td>
      <td>2</td>
      <td>peopl trembl ootnot _remble_ _chestnut_ sound ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pushed back into efficient reason or divine wi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>4</td>
      <td>push back effici reason divin almost vanish la...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>in the interpretation itself. Amongst these ma...</td>
      <td>Averroes</td>
      <td>278</td>
      <td>1</td>
      <td>interpret mongst may mention bu amid l hazzali...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>highest degree of perfection and an eternal pl...</td>
      <td>T.J. de Boer</td>
      <td>320</td>
      <td>5</td>
      <td>highest degre perfect etern plenitud realiti s...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>also the masts and yards, and wearing away the...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>395</td>
      <td>3</td>
      <td>also mast yard wear away deck holyston well le...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['documents'].iloc[0]

```




    "people tremble.[32] [Footnote 32: _Tremble_ and _chestnut_ have the same sound in Chinese.] On hearing this, the Master said, I do not speak of what is ended, chide what is settled, or find fault with what is past.[33] [Footnote 33: In old times men had been sacrificed at the earth-altars, and Tsai Wo's answer might seem to approve the practice.] 22. The Master said, How shallow was Kuan Chung! But, said one, was not Kuan Chung thrifty? The Kuan, said the Master, owned San Kuei, and no one of his household held two posts: was that thrift? At least Kuan Chung knew good form. The Master said, Kings screen their gates with trees; the Kuan, too, had trees to screen his gate. When two kings are carousing, they have a stand for the turned-down cups; the Kuan had a turned-down cup-stand, too! If the Kuan knew good form, who does not know good form?[34] [Footnote 34: Kuan Chung (+ 645 B.C.), a famous man in his day, was chief minister to the Duke"




```python
df['clean_msg'].iloc[0]

```




    'peopl trembl ootnot _remble_ _chestnut_ sound hines n hear aster said speak end chide settl find fault past ootnot n old time men sacrif earthaltar sai os answer might seem approv practic aster said ow shallow uan hung ut said one uan hung thrifti uan said aster own uei one household held two post thrift least uan hung knew good form aster said ing screen gate tree uan tree screen gate hen two king carous stand turneddown cup uan turneddown cupstand f uan knew good form know good form ootnot uan hung famou man day chief minist uke'



## Split the data into training and test sets



```python
# Split into X/y
from sklearn.model_selection import train_test_split, cross_val_score

x = df["documents"]
y = df["label"]
print(x.shape)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```

    (1200,)
    (1200,)
    (960,)
    (240,)
    (960,)
    (240,)


## Vectorization: Convert text features to numeric



```python
from sklearn.feature_extraction.text import TfidfVectorizer
# sublinear_df=True, use a logarithmic form for frequency

# cv2 = TfidfVectorizer(ngram_range=(1, 2))

cv2 = TfidfVectorizer(sublinear_tf=True, min_df=10, norm='l2',
                      ngram_range=(1, 2), stop_words='english')

# min_df is the minimum numbers of documents a word must be present in to be kept
# norm is set to l2, to ensure all our feature vectors have a euclidian norm of 1

X_traincv = cv2.fit_transform(x_train)
x_testcv = cv2.transform(x_test)
print(X_traincv.toarray())

```

    [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]]


## Train the model


```python
from sklearn.naive_bayes import MultinomialNB
mnb= MultinomialNB()
```


```python
mnb.fit(X_traincv, y_train)
```




    MultinomialNB()




```python
mnb.score(x_testcv, y_test)*100
```




    89.58333333333334




```python
print(f"Test Text:")
print(x_test.iloc[1])
```

    Test Text:
    said, that music could reach such heights. 14. Jan Yu said, Is the Master for the lord of Wei?[66] [Footnote 66: The grandson of Duke Ling, the husband of Nan-tzu. His father had been driven from the country for plotting to kill Nan-tzu. When Duke Ling died, he was succeeded by his grandson, who opposed by force his father's attempts to seize the throne.] I shall ask him, said Tzu-kung. He went in, and said, What kind of men were Po-yi[67] and Shu-ch'i? Worthy men of yore, said the Master. Did they rue the past? They sought love and found it; what had they to rue? Tzu-kung went out, and said, The Master is not for him. 15. The Master said, Eating coarse rice and drinking water, with bent arm for pillow, we may be merry; but ill-gotten wealth and honours are to me a wandering cloud. 16. The Master said, Given a few more years, making fifty for learning the Yi,[68] I might be freed from gross faults. [Footnote 67: See Book



```python
actual_label = y_test.iloc[1]
actual_label
```




    2




```python
labelencoder.inverse_transform([actual_label])
```




    array(['Confucius'], dtype=object)




```python
y_pred = mnb.predict(cv2.transform([x_test.iloc[1]]))
print(f"Predicted Y : {y_pred[0]}, author: {labelencoder.inverse_transform([y_pred[0]])}")
```

    Predicted Y : 2, author: ['Confucius']



```python
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=50)
# knn.fit(X_traincv, y_train)
# knn.score(x_testcv, y_test)*100
```

## Hyperparameter tuning: Searching for the ideal model


```python
from termcolor import cprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# ignore ConvergenceWarnings
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

models = [
    { "model_instance": KNeighborsClassifier(),
      "model_name": "KNeighbors",
      "params": {
          "n_neighbors": [25,30,35,45],
          "weights": ['uniform','distance'],
          "leaf_size": [25,30,35]
        }
    },
    { "model_instance": DecisionTreeClassifier(),
      "model_name": "DecisionTree",
      "params": {
          "criterion": ['gini','entropy'],
          "splitter": ['best','random'],
          "max_depth": [None,90,95,100],
          "max_features": [None, "auto","sqrt","log2"],
          "random_state": [42]
      }
    },
    { "model_instance": MultinomialNB(),
      "model_name":"MultinomialNB",
      "params": {
          "fit_prior": [True, False]
      }
    },
    { "model_instance": LinearSVC(),
      "model_name": "SVC",
      "params": {
          "loss": ['hinge','squared_hinge'],
          "multi_class": ['ovr', 'crammer_singer'],
          "fit_intercept": [True, False],
          "random_state": [42],
          "max_iter": [900, 1000, 1100]
      }
    },
    { "model_instance": svm.SVC(),
      "model_name": "SVM",
      "params": {
          'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']
      }
    },
    { "model_instance": RandomForestClassifier(),
      "model_name": "RandomForest",
      "params": {
      "criterion": ['gini','entropy'],
      "bootstrap": [True, False],
      "max_depth": [85,90,95,100],
      "max_features": ['sqrt','log2'],
      "n_estimators": [60, 80, 90],
      "random_state": [42]
      }
    },
    { "model_instance": SGDClassifier(),
      "model_name": "SGDClassifier",
      "params": {
          "loss": ['hinge','log','perceptron'],
          "penalty": ['l2', 'l1'],
          "alpha": [0.0001, 0.0003, 0.0010],
          "early_stopping": [True],
          "max_iter": [1000, 1500],
          "random_state": [42]
      }
    }
]

scores = []
highest_acc = 0
best_model = None

for model in models:

  # Create a based model
  model_instance = model["model_instance"]
  model_name = model["model_name"]
  print("Running Model:",end="")
  cprint(model_name, "green")
  # Instantiate the grid search model
  classifier = GridSearchCV(estimator=model_instance, param_grid=model["params"],
                            cv = 10, n_jobs = 1)

  # Fit the model
  classifier.fit(X_traincv, y_train);

  # Make a prediction on the test split to find model accuracy
  predicted = classifier.predict(x_testcv)
  acc = accuracy_score(predicted, y_test)
  # If model have the highest accuracy, it's out best model
  if acc > highest_acc:
    highest_acc = acc
    best_model = classifier

  scores.append({
    "model":model_name,
    "training_best_score": classifier.best_score_,
    "test_best_score": acc,
    "best_params": classifier.best_params_
  })

  print("Best Training Score:",end="")
  cprint(f"{round(classifier.best_score_*100,2)}%  ", "cyan", end="")
  print("Best Test Score:",end="")
  cprint(f"{round(acc*100,2)}%  ", "cyan")

  print("Best Params:",end="")
  cprint(classifier.best_params_,"yellow")

print("\nBest Model:",end="")
cprint(best_model.best_estimator_,"red")

```

    Running Model:[32mKNeighbors[0m
    Best Training Score:[36m88.85%  [0mBest Test Score:[36m88.75%  [0m
    Best Params:[33m{'leaf_size': 25, 'n_neighbors': 25, 'weights': 'uniform'}[0m
    Running Model:[32mDecisionTree[0m
    Best Training Score:[36m75.62%  [0mBest Test Score:[36m75.83%  [0m
    Best Params:[33m{'criterion': 'gini', 'max_depth': None, 'max_features': None, 'random_state': 42, 'splitter': 'best'}[0m
    Running Model:[32mMultinomialNB[0m
    Best Training Score:[36m90.21%  [0mBest Test Score:[36m90.42%  [0m
    Best Params:[33m{'fit_prior': False}[0m
    Running Model:[32mSVC[0m
    Best Training Score:[36m92.08%  [0mBest Test Score:[36m92.08%  [0m
    Best Params:[33m{'fit_intercept': True, 'loss': 'hinge', 'max_iter': 900, 'multi_class': 'crammer_singer', 'random_state': 42}[0m
    Running Model:[32mSVM[0m
    Best Training Score:[36m92.71%  [0mBest Test Score:[36m91.67%  [0m
    Best Params:[33m{'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}[0m
    Running Model:[32mRandomForest[0m
    Best Training Score:[36m91.25%  [0mBest Test Score:[36m90.0%  [0m
    Best Params:[33m{'bootstrap': False, 'criterion': 'gini', 'max_depth': 100, 'max_features': 'log2', 'n_estimators': 80, 'random_state': 42}[0m
    Running Model:[32mSGDClassifier[0m
    Best Training Score:[36m91.77%  [0mBest Test Score:[36m91.25%  [0m
    Best Params:[33m{'alpha': 0.001, 'early_stopping': True, 'loss': 'hinge', 'max_iter': 1000, 'penalty': 'l2', 'random_state': 42}[0m

    Best Model:[31mLinearSVC(loss='hinge', max_iter=900, multi_class='crammer_singer',
              random_state=42)[0m



```python
d = pd.DataFrame(scores)
d

```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>training_best_score</th>
      <th>test_best_score</th>
      <th>best_params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KNeighbors</td>
      <td>0.888542</td>
      <td>0.887500</td>
      <td>{'leaf_size': 25, 'n_neighbors': 25, 'weights'...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTree</td>
      <td>0.756250</td>
      <td>0.758333</td>
      <td>{'criterion': 'gini', 'max_depth': None, 'max_...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MultinomialNB</td>
      <td>0.902083</td>
      <td>0.904167</td>
      <td>{'fit_prior': False}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVC</td>
      <td>0.920833</td>
      <td>0.920833</td>
      <td>{'fit_intercept': True, 'loss': 'hinge', 'max_...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SVM</td>
      <td>0.927083</td>
      <td>0.916667</td>
      <td>{'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RandomForest</td>
      <td>0.912500</td>
      <td>0.900000</td>
      <td>{'bootstrap': False, 'criterion': 'gini', 'max...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SGDClassifier</td>
      <td>0.917708</td>
      <td>0.912500</td>
      <td>{'alpha': 0.001, 'early_stopping': True, 'loss...</td>
    </tr>
  </tbody>
</table>
</div>



### Plotting each models performance


```python
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 5))
sns.barplot(x='model', y='test_best_score', data=d, errwidth=0)

```




    <AxesSubplot:xlabel='model', ylabel='test_best_score'>





![png](README_files/README_46_1.png)



## Evaluating the model and make predictions using the Best Model


```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='micro')
    recall = recall_score(y_true, y_preds, average='micro')
    f1 = f1_score(y_true, y_preds, average='micro')
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")

    return metric_dict

```

Now we make predictions using the test data to see how the model performs


```python
predicted = best_model.predict(x_testcv)
evaluate_preds(y_test, predicted)

```

    Acc: 92.08%
    Precision: 0.92
    Recall: 0.92
    F1 score: 0.92





    {'accuracy': 0.92, 'precision': 0.92, 'recall': 0.92, 'f1': 0.92}



Classification report


```python
from sklearn import metrics
print(metrics.classification_report(y_test, predicted,
                                    target_names=df['author'].unique()))
```

                                 precision    recall  f1-score   support

                      Confucius       1.00      0.82      0.90        49
    Sir Thomas Clifford Allbutt       0.95      0.97      0.96        40
                       Averroes       0.91      0.93      0.92        42
                   T.J. de Boer       1.00      0.98      0.99        43
     Robert C. (Chamblet) Adams       0.76      0.89      0.82        36
            Alfred William Benn       0.91      0.97      0.94        30

                       accuracy                           0.92       240
                      macro avg       0.92      0.93      0.92       240
                   weighted avg       0.93      0.92      0.92       240



Confusion Matrix


```python
df["author"].unique()

```




    array(['Confucius', 'Sir Thomas Clifford Allbutt', 'Averroes',
           'T.J. de Boer', 'Robert C. (Chamblet) Adams',
           'Alfred William Benn'], dtype=object)




```python
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, predicted)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=df["author"].unique(), yticklabels=df["author"].unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

```



![png](README_files/README_55_0.png)



## Results

- `n-fold = 3`

<div align="center">
<img src="img/cv_3.jpg" alt="cv_3.jpg" width="900px">
</div>

- default `n-fold = 5`

<div align="center">
<img src="img/cv_5.jpg" alt="cv_5.jpg" width="900px">
</div>

- default `n-fold = 5`, without dropping words len < 100

<div align="center">
<img src="img/cv_5_no_w_drop.jpg" alt="cv_5_no_w_drop.jpg" width="900px">
</div>

- `n-fold = 10`

<div align="center">
<img src="img/cv_10.jpg" alt="cv_10.jpg" width="900px">
</div>

<div align="center">
<img src="img/cv_10_1.jpg" alt="cv_10_1.jpg" width="900px">
</div>

<div align="center">
<img src="img/cv_10_2.jpg" alt="cv_10_2.jpg" width="900px">
</div>
