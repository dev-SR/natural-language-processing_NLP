# Assignment-1

<div align="center">
<img src="img/asgn.jpg" alt="asgn.jpg" width="1000px">
</div>

- [Assignment-1](#assignment-1)
  - [Load Data](#load-data)
    - [Processing file - test codes](#processing-file---test-codes)
    - [Processing file - Final](#processing-file---final)
    - [Label Encoding](#label-encoding)
      - [using `pd.factorize()`](#using-pdfactorize)
      - [using `sklearn.preprocessing.LabelEncoder()`](#using-sklearnpreprocessinglabelencoder)
  - [Data Pre-Process](#data-pre-process)
  - [Split the data into training and test sets](#split-the-data-into-training-and-test-sets)
  - [Convert text features to numeric](#convert-text-features-to-numeric)
  - [Train the model](#train-the-model)

```python
"""
cd .\04assignment\
jupyter nbconvert --to markdown asng.ipynb --output README.md
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
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
df = df.drop(df[df['words_count'] < 100].index)
# drop documents with less than 100 words
```


```python
df['words_count'].min()
```




    142



### Label Encoding

#### using `pd.factorize()`


```python
y , label = df['author'].factorize()
label
```




    Index(['Sir Thomas Clifford Allbutt', 'Robert C. (Chamblet) Adams', 'Averroes',
           'Confucius', 'Alfred William Benn', 'T.J. de Boer'],
          dtype='object')




```python
np.unique(y)
```




    array([0, 1, 2, 3, 4, 5], dtype=int64)




```python
if_predit = 1
label[if_predit]
```




    'Robert C. (Chamblet) Adams'




```python
df['label'] = y
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
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SCIENCE AND MEDIEVAL THOUGHT. * * * * * London...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>143</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>et facit nos concludere quæstionem, sed non ce...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a phantom, and again the spirit of a new world...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>first applied to the art and romance of the Mi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>143</td>
      <td>0</td>
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
      <td>is beneath it, and through logical inference w...</td>
      <td>T.J. de Boer</td>
      <td>320</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>“If we had pleased, we had certainly given eve...</td>
      <td>Averroes</td>
      <td>277</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>astronomy, and alchemy. Averroes it was who fi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>matter of principle there can of course be but...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>394</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>also the masts and yards, and wearing away the...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>395</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### using `sklearn.preprocessing.LabelEncoder()`


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
      <td>its place. According to Cousin, in all countri...</td>
      <td>Alfred William Benn</td>
      <td>220</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>families of Meng, Shu, and Chi were descended,...</td>
      <td>Confucius</td>
      <td>173</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>were the conjectures about her, and some of ou...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>395</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>He is the author of the famous saying--the sol...</td>
      <td>Alfred William Benn</td>
      <td>220</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>not truthful, I will know nothing. 17. The Mas...</td>
      <td>Confucius</td>
      <td>173</td>
      <td>2</td>
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
import re
ps = PorterStemmer()

```


```python
def text_process(msg):
	# Remove Emails
	msg = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', msg)
	# Remove url's
	msg = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', msg)
	# Remove Money Symbols
	msg = re.sub('£|\$', 'moneysymb', msg)
	# Remove Phone Numbers
	msg = re.sub(
		'\b(\+\d{1,2}\s?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4})\b', 'phonenumbr', msg)
	# Remove Numbers
	msg = re.sub('\d+(\.\d+)?', '', msg)
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
df['clean_msg'] = df.documents.apply(text_process)
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
      <td>is beneath it, and through logical inference w...</td>
      <td>T.J. de Boer</td>
      <td>320</td>
      <td>5</td>
      <td>beneath logic infer final ration consider dire...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>“If we had pleased, we had certainly given eve...</td>
      <td>Averroes</td>
      <td>277</td>
      <td>2</td>
      <td>pleas certainli given everi soul direct word h...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>astronomy, and alchemy. Averroes it was who fi...</td>
      <td>Sir Thomas Clifford Allbutt</td>
      <td>142</td>
      <td>0</td>
      <td>astronomi alchemi averro first assert independ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>matter of principle there can of course be but...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>394</td>
      <td>1</td>
      <td>matter principl cours one answerchrist teach e...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>also the masts and yards, and wearing away the...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>395</td>
      <td>1</td>
      <td>also mast yard wear away deck holyston well le...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['documents'].iloc[0]

```




    'is beneath it, and through logical inference with what is above it, and finally with itself by rational consideration or direct intuition. Of these kinds of knowledge the surest and the most deserving of preference is knowledge of one’s self. When human knowledge attempts to go farther than this, it proves itself to be limited in many ways. Therefore one must not philosophize straight away about questions like the origin or the eternity of the world, but make his first essays with what is simpler. And only through renunciation of the world, and righteous conduct, does the soul lift itself gradually up to the pure knowledge of the Highest.” 5. After secular instruction in Grammar, Poetry and History, and after religious education and doctrine, philosophic study should begin with the mathematical branches. Here everything is set forth in Neo-Pythagorean and Indian fashion. Not only numbers but even the letters of the alphabet are employed in childish trifling. It was particularly convenient for the Brethren that the number of letters in the Arabic alphabet is 28, or 4 multiplied by 7. Instead of proceeding according to practical and real points of view, they give the rein to fancy in all the sciences, in accordance with grammatical analogies and relations of numbers. Their Arithmetic does not investigate Number as such, but rather its significance. No search is made for any more suitable mode of expressing number in the case of phenomena; but things are themselves explained in accordance with the system of numbers. The Theory of number is Divine wisdom, and is above Things, for things are only formed after the pattern of numbers. The absolute principle of all existence and thought is the number One. The science of number, therefore, is found at the beginning, middle, and end of all philosophy. Geometry, with its figures addressing the eye, serves merely to make it more easily understood by beginners, but Arithmetic alone'




```python
df['clean_msg'].iloc[0]

```




    'beneath logic infer final ration consider direct intuit kind knowledg surest deserv prefer knowledg one self human knowledg attempt go farther prove limit mani way therefor one must philosoph straight away question like origin etern world make first essay simpler renunci world righteou conduct soul lift gradual pure knowledg highest secular instruct grammar poetri histori religi educ doctrin philosoph studi begin mathemat branch everyth set forth neopythagorean indian fashion number even letter alphabet employ childish trifl particularli conveni brethren number letter arab alphabet multipli instead proceed accord practic real point view give rein fanci scienc accord grammat analog relat number arithmet investig number rather signific search made suitabl mode express number case phenomena thing explain accord system number theori number divin wisdom thing thing form pattern number absolut principl exist thought number one scienc number therefor found begin middl end philosophi geometri figur address eye serv mere make easili understood beginn arithmet alon'



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


## Convert text features to numeric



```python
from sklearn.feature_extraction.text import TfidfVectorizer
# sublinear_df=True, use a logarithmic form for frequency

# cv2 = TfidfVectorizer(ngram_range=(1, 2))  # 82.5%


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




    92.5




```python
print(f"Test Text:")
print(x_test.iloc[1])
```

    Test Text:
    he has given it the attention which the subject demands as a part of the history of the country. It would be a difficult matter to get at the first American sailor, or to even guess when he existed, but that our continent was once well populated, and that its prehistoric inhabitants sailed the lakes and seas as well as trod the land, is a matter of certainty. Later, when America became known to Europeans, the new comers found Indians well provided with excellent canoes, built of bark or fashioned from logs, but they were "near shore" sailors. The author quotes one instance where a deep sea voyage was undertaken by them in the early days of the English settlers. Certain Carolina Indians, he says, wearied of the white man's sinful ways in trade, thought themselves able to deal direct with the consumers across the "Big Sea Water." So they built several large canoes and loading these with furs and tobacco paddled straight out to sea bound for England. But their ignorance of navigation speedily got the best of their valor. They were never heard of more. The early white navigators of our waters can hardly be considered American sailors. The new found continent was to them of value only for what could be brought away from them in treasure or in merchantable produce, and it was only when an actual and permanent colonization began that a race of native-born sailors was developed on the Atlantic coasts. OLD CONCORD: HER HIGHWAYS AND BYWAYS. Ill. By Margaret Sidney. Boston: D. Lothrop Co. Price $3.00. Of all the books of the year there is not one which carries within it such an aroma of peculiar delight as this series of sketches and descriptions of the highways and byways of that most picturesque of towns, Old Concord. Concord is like no other place in New England. There may be other places as beautiful in their way; there are others, perhaps, of more importance in the Commonwealth, and we know there are hundreds of places where there is more active life to the square foot, but with all these admissions Concord still remains a place of special charm, the result and consequence of more causes than we care to analyze. Its picturesqueness and a certain quaintness of the village has always been noticed by visitors, no matter from



```python
actual_label = y_test.iloc[1]
actual_label
```




    3




```python
labelencoder.inverse_transform([actual_label])
```




    array(['Robert C. (Chamblet) Adams'], dtype=object)




```python
y_pred = mnb.predict(cv2.transform([x_test.iloc[1]]))
print(f"Predicted Y : {y_pred[0]}, author: {labelencoder.inverse_transform([y_pred[0]])}")
```

    Predicted Y : 3, author: ['Robert C. (Chamblet) Adams']



```python
# For pd.factorize()
# print(f"Actual Y: {y_test.iloc[0]}, author: {label[y_test.iloc[0]]}")
# y_pred = mnb.predict(cv2.transform([x_test.iloc[0]]))
# print(f"Predicted Y : {y_pred[0]}, author: {label[y_pred[0]]}")

```
