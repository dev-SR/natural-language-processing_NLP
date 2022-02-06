# Spam Text Classification

- [Spam Text Classification](#spam-text-classification)
  - [Load,Explore and Clean Data](#loadexplore-and-clean-data)
  - [Preprocess Messages](#preprocess-messages)
  - [Vectorization](#vectorization)
  - [🤖 Building and evaluating a model](#-building-and-evaluating-a-model)
    - [Pipeline](#pipeline)


```python
"""
cd .\02spam_msg_classification\
jupyter nbconvert --to markdown spam.ipynb --output README.md

"""
import pandas as pd
import numpy as np
import nltk

```

## Load,Explore and Clean Data

[dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)


```python
df = pd.read_csv('spam.csv')
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.rename(columns={'v1': 'label', 'v2': 'messages'}, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Replace ham with 0 and spam with 1
df["label"] = df["label"].replace(['ham', 'spam'], [0, 1])
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>messages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Count the number of words in each Text
df['Count'] = df['messages'].apply(len)
df.head()
# or,
# create "Count" column
# df['Count'] = 0
# for i in np.arange(0, len(df.messages)):
#     df.loc[i, 'Count'] = int(len(df.loc[i, 'messages']))

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>messages</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Total ham(0) and spam(1) messages
df['label'].value_counts()

```




    0    4825
    1     747
    Name: label, dtype: int64



## Preprocess Messages


```python
df['messages'][0],df['messages'][1],df['messages'][2]
```




    ('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...',
     'Ok lar... Joking wif u oni...',
     "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's")




```python
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

```


```python
ps = PorterStemmer()
```


```python
n,=df['messages'].shape
n
```




    5572




```python
corpus = []
for i in np.arange(0, n):
	msg = df['messages'][i]
	if i==0:
		print("Original",msg)
	# Remove Emails
	msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', msg)
	# Remove url's
	msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', msg)
	# Remove Money Symbols
	msg = re.sub('£|\$', 'moneysymb', msg)
	# Remove Phone Numbers
	msg = re.sub('\b(\+\d{1,2}\s?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b', 'phonenumbr', msg)
	# Remove Numbers
	msg = re.sub('\d+(\.\d+)?', 'numbr', msg)
	# Remove Punctuation
	msg = re.sub('[^\w\s]', '', msg)
	# Remove Extra Spaces
	msg = re.sub('\s+', ' ', msg)


	if i==0:
		print("After Regular Expression: ", msg)

	# Lower case
	msg = msg.lower()
	if i == 0:
		print("After Lower case: ", msg)

	# Tokenize
	words = word_tokenize(msg)
	if i == 0:
		print("After Tokenize M: ",words)

	# Remove Stop Words
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]
	if i == 0:
		print("After Remove Stop Words: ",words)

	# Stemming
	stemmed_words = [ps.stem(w) for w in words]
	if i == 0:
		print("After Stemming: ",stemmed_words)

	# Join the words back into one string separated by space,
	sen = ' '.join(stemmed_words)
	if i == 0:
		print("Final: \n",sen)

	corpus.append(sen)

```

    Original Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
    After Regular Expression:  Go until jurong point crazy Available only in bugis n great world la e buffet Cine there got amore wat
    After Lower case:  go until jurong point crazy available only in bugis n great world la e buffet cine there got amore wat
    After Tokenize M:  ['go', 'until', 'jurong', 'point', 'crazy', 'available', 'only', 'in', 'bugis', 'n', 'great', 'world', 'la', 'e', 'buffet', 'cine', 'there', 'got', 'amore', 'wat']
    After Remove Stop Words:  ['go', 'jurong', 'point', 'crazy', 'available', 'bugis', 'n', 'great', 'world', 'la', 'e', 'buffet', 'cine', 'got', 'amore', 'wat']
    After Stemming:  ['go', 'jurong', 'point', 'crazi', 'avail', 'bugi', 'n', 'great', 'world', 'la', 'e', 'buffet', 'cine', 'got', 'amor', 'wat']
    Final:
     go jurong point crazi avail bugi n great world la e buffet cine got amor wat



```python
corpus[:5]
```




    ['go jurong point crazi avail bugi n great world la e buffet cine got amor wat',
     'ok lar joke wif u oni',
     'free entri numbr wkli comp win fa cup final tkt numbrst may numbr text fa numbr receiv entri questionstd txt ratetc appli numbrovernumbr',
     'u dun say earli hor u c alreadi say',
     'nah dont think goe usf live around though']




```python
def text_process(msg):
	# Remove Emails
	msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', msg)
	# Remove url's
	msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', msg)
	# Remove Money Symbols
	msg = re.sub('£|\$', 'moneysymb', msg)
	# Remove Phone Numbers
	msg = re.sub('\b(\+\d{1,2}\s?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b', 'phonenumbr', msg)
	# Remove Numbers
	msg = re.sub('\d+(\.\d+)?', 'numbr', msg)
	# Remove Punctuation
	msg = re.sub('[^\w\s]', '', msg)
	# Remove Extra Spaces
	msg = re.sub('\s+', ' ', msg)

	# Lower case
	msg = msg.lower()

	# Tokenize
	words = word_tokenize(msg)

	# Remove Stop Words
	stop_words = set(stopwords.words('english'))
	words = [w for w in words if not w in stop_words]

	# Stemming
	stemmed_words = [ps.stem(w) for w in words]

	# Join the words back into one string separated by space,
	stemmed_sen = ' '.join(stemmed_words)

	return stemmed_sen


```


```python
df['clean_msg'] = df.messages.apply(text_process)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>messages</th>
      <th>Count</th>
      <th>clean_msg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>111</td>
      <td>go jurong point crazi avail bugi n great world...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>29</td>
      <td>ok lar joke wif u oni</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>155</td>
      <td>free entri numbr wkli comp win fa cup final tk...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>49</td>
      <td>u dun say earli hor u c alreadi say</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>61</td>
      <td>nah dont think goe usf live around though</td>
    </tr>
  </tbody>
</table>
</div>



## Vectorization


```python
X = df.clean_msg
y = df.label
print(X.shape)
print(y.shape)

```

    (5572,)
    (5572,)



```python
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

```

    (4179,)
    (1393,)
    (4179,)
    (1393,)



```python
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

```


```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(X_train_dtm)
tfidf_transformer.transform(X_train_dtm)

```




    <4179x5974 sparse matrix of type '<class 'numpy.float64'>'
    	with 35127 stored elements in Compressed Sparse Row format>



## 🤖 Building and evaluating a model



```python
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

```


```python
# train the model using X_train_dtm (timing it with an IPython "magic command")
%time nb.fit(X_train_dtm, y_train)

```

    Wall time: 5 ms





    MultinomialNB()




```python
msg = vect.inverse_transform(X_test_dtm[1])
" ".join(msg[0])
```




    'anyway even good mani'




```python
print("Actual: ", y_test.iloc[1])

```

    Actual:  0



```python
print("Predicted: ", nb.predict(X_test_dtm[1])[0])
```

    Predicted:  0



```python
y_pred_class = nb.predict(X_test_dtm)
```


```python
# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)

```


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

```


```python
# calculate accuracy of class predictions
accuracy_score(y_test, y_pred_class)

```




    0.9842067480258435




```python
print(classification_report(y_test, nb.predict(X_test_dtm)))

```

                  precision    recall  f1-score   support

               0       0.99      0.99      0.99      1213
               1       0.95      0.93      0.94       180

        accuracy                           0.98      1393
       macro avg       0.97      0.96      0.96      1393
    weighted avg       0.98      0.98      0.98      1393




```python
confusion_matrix(y_test, y_pred_class)

```




    array([[1204,    9],
           [  13,  167]], dtype=int64)




```python
# print message text for false positives (ham incorrectly classifier)
# X_test[(y_pred_class==1) & (y_test==0)]
X_test[y_pred_class > y_test]

```




    4598                                laid airtel line rest
    386                                     custom place call
    1289    heygreat dealfarm tour numbram numbrpm moneysy...
    3245    funni fact nobodi teach volcano numbr erupt ts...
    1235    opinion numbr numbr jada numbr kusruthi numbr ...
    2146                                    collect ur laptop
    5094    hi shanilrakhesh httpaddr exchang uncut diamon...
    494                                      free nowcan call
    3140                                    custom place call
    Name: clean_msg, dtype: object




```python
# print message text for false negatives (spam incorrectly classifier)
X_test[y_pred_class < y_test]

```




    4674    hi babe chloe r u smash saturday night great w...
    3528    xma new year eve ticket sale club day numbram ...
    4247     accordingli repeat text word ok mobil phone send
    3417    life never much fun great came made truli spec...
    2773    come take littl time child afraid dark becom t...
    5       freemsg hey darl numbr week word back id like ...
    2078                         numbr freeringtonerepli real
    1457    clair havin borin time alon u wan na cum numbr...
    190             uniqu enough find numbrth august httpaddr
    2429    guess ithi first time creat web page httpaddr ...
    4067    tbspersolvo chase us sinc sept fornumbr defini...
    3358               sorri miss call let talk time im numbr
    2821    romcapspam everyon around respond well presenc...
    Name: clean_msg, dtype: object



### Pipeline


```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()),
                 ('tfid', TfidfTransformer()),
                 ('model', MultinomialNB())])
pipe.fit(X_train, y_train)

```




    Pipeline(steps=[('bow', CountVectorizer()), ('tfid', TfidfTransformer()),
                    ('model', MultinomialNB())])




```python
y_pred = pipe.predict(X_test)
```


```python
accuracy_score(y_test, y_pred)

```




    0.9698492462311558


