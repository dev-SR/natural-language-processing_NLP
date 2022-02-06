# Assignment-1

- [Assignment-1](#assignment-1)
  - [Load Data](#load-data)
  - [Data Pre-Process](#data-pre-process)


```python
"""
cd .\03assignment\
jupyter nbconvert --to markdown asng.ipynb --output README.md
"""
```

## Load Data


```python
author_name=""
with open("64171-0.txt", 'r') as target_file:
	for num, line in enumerate(target_file.readlines()):
		if str("Author") in line:
			author_name = line.split(":")[1].strip()
author_name
```




    'Robert C. (Chamblet) Adams'




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
	indices = []
	for i, w in enumerate(data):
		# start reading from CHAPTER I.
		if "CHAPTERI" in w:
			indices.append(i)
	last = indices[-1]
	# print(last)
	data = data[last+1:]

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


```python
import pandas as pd
```


```python
author_list = [author_name]*len(list_of_blocks)
# author_list
```


```python
df = pd.DataFrame({
	'data': list_of_blocks,
	"author": author_list
})
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
      <th>data</th>
      <th>author</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>In Lloyds Register is recorded:--"_Rocket_, Bk...</td>
      <td>Robert C. (Chamblet) Adams</td>
    </tr>
    <tr>
      <th>1</th>
      <td>out. Wishing to choose for myself who should s...</td>
      <td>Robert C. (Chamblet) Adams</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quiet, but the mate remarked, he thought we ha...</td>
      <td>Robert C. (Chamblet) Adams</td>
    </tr>
    <tr>
      <th>3</th>
      <td>shore again and you'll never catch me on board...</td>
      <td>Robert C. (Chamblet) Adams</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lower rigging of the ship, forms the great vol...</td>
      <td>Robert C. (Chamblet) Adams</td>
    </tr>
  </tbody>
</table>
</div>



## Data Pre-Process


```python
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
df['clean_msg'] = df.data.apply(text_process)
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
      <th>data</th>
      <th>author</th>
      <th>clean_msg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>In Lloyds Register is recorded:--"_Rocket_, Bk...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>lloyd regist recorded_rocket_ bk medford wo ic...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>out. Wishing to choose for myself who should s...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>wish choos sail mani month ship master told se...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>quiet, but the mate remarked, he thought we ha...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>quiet mate remark thought pretti hard crew wat...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>shore again and you'll never catch me on board...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>shore youll never catch board ship morn light ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>lower rigging of the ship, forms the great vol...</td>
      <td>Robert C. (Chamblet) Adams</td>
      <td>lower rig ship form great volum sound constant...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['data'][0]

```




    'In Lloyds Register is recorded:--"_Rocket_, Bk. 384, 135, 25, 16.5, 1851, Medford, W.O., icf.," which being interpreted means, Bark _Rocket_, 384 tons, 135 feet long, 25 feet beam, 16-1/2 feet depth of hold, built in 1851, at Medford, of white oak, with iron and copper fastenings. To which may be added, that she was a well known trader to the East Indies, being called in those ports "the green bark," on account of being painted a dark green, or what the painters style tea color. She was a good looking vessel, neatly finished about the decks, and the masts and yards were all scraped bright. The chief peculiarity was that she was narrow in proportion to her length, being compared by an old sailor to "a plank set on edge." This caused her to be reputed, and not undeservedly, a crank vessel, and many a gloomy croaker has uttered the foreboding that like her sister ship, the "Dauntless," she would go to sea sometime--never to return. Yet for many years she had gone and come, and though occasionally threatening to capsize, she had never really performed this undesirable manÅ“uvre. The builder and the subsequent owner were two of the most practical merchants of Boston. She must therefore have been well put together and properly cared for, as there was truth in the remark made, that "what Nat G----, and Dick B---- didn\'t know about a ship wasn\'t worth knowing." * * * * * The _Rocket_ was lying at Central Wharf in Boston, loading a cargo for the East Indies. Barrels of beef, pork, tar and pitch were stowed in the bottom; then followed in miscellaneous order, lumber, sewing machines, kerosene oil, flour, biscuits, preserves, ice pitchers, carriages, oars and many other articles. As the sailing day drew near, the important matter of choosing officers and crew had to be considered. The first person who applied was an aspirant to the mate\'s berth. "How long have you been to sea?" was asked. "Thirty years." "Why! how old are you?" "Twenty-nine." "How do you make that out?" "Oh, I was born and bred at sea." He was thought to be too old a sailor for a young captain to manage, and was not engaged. Soon a young man applied, with more modest demeanor, and he was secured. The rest of the crew were soon picked'




```python
df['clean_msg'][0]

```




    'lloyd regist recorded_rocket_ bk medford wo icf interpret mean bark _rocket_ ton feet long feet beam feet depth hold built medford white oak iron copper fasten may ad well known trader east indi call port green bark account paint dark green painter style tea color good look vessel neatli finish deck mast yard scrape bright chief peculiar narrow proport length compar old sailor plank set edg caus reput undeservedli crank vessel mani gloomi croaker utter forebod like sister ship dauntless would go sea sometimenev return yet mani year gone come though occasion threaten capsiz never realli perform undesir manåuvr builder subsequ owner two practic merchant boston must therefor well put togeth properli care truth remark made nat g dick b didnt know ship wasnt worth know _rocket_ lie central wharf boston load cargo east indi barrel beef pork tar pitch stow bottom follow miscellan order lumber sew machin kerosen oil flour biscuit preserv ice pitcher carriag oar mani articl sail day drew near import matter choos offic crew consid first person appli aspir mate berth long sea ask thirti year old twentynin make oh born bred sea thought old sailor young captain manag engag soon young man appli modest demeanor secur rest crew soon pick'


