# NLP workflow

We are going to use an example to go through the general process of NLP. In a lot of documentations, segmentation, cleaning and normalization are combined with name pre-processing. The general idea of pre-processing is reducing the vocabulary's size (which in turn may improve the performance and reduce the resource consuption of downstream tasks) while keeping the most important `flags`. 

![text_processing_flow.png](attachment:12e59a02-2c30-4b98-9fc8-51f70227f5c4.png)

## Segmentation

`input`: _helloworld_

`output`: _hello world_

Segmentation task is not very common in the English / Frech world. In most of the cases, we use segmentation to handle Chinese / Japanese / Korean characters. 

The solutoion of segmentation is pretty standard and usually gives us an accuracy around 95%~97%. From the high level, we can categorize the solution into two classes:
- Rule based (foward max matching, backward max matching)
- Statistic based, e.g. language model, Hidden Markov Model, Conditoinal Random Field

For statistic based segmentation, the core of infering is Viterbi Algorithm (a DP based algorithm). Besides NLP, Viterbi algorithm is also widely used in decoding for communication system. 

Here we will use Chinese as an example to show segmentation works. The library we use is [jieba](https://github.com/jsrpy/Chinese-NLP-Jieba), there are other libraries can serve the same purpose like SnowNLP, LTP, etc.


```python
import jieba
text= "自然语言处理很有趣" # it means natural language processing is interesting
segmentations = jieba.cut(text)
```


```python
list(segmentations)
```

    Building prefix dict from the default dictionary ...
    Loading model from cache /tmp/jieba.cache
    Loading model cost 0.393 seconds.
    Prefix dict has been built successfully.





    ['自然语言', '处理', '很', '有趣']



As we can see in the above example, `jieba` cut the sentence into different word. 

## Cleaning
The purpose of the task is to remvoe useless information (noise) from the text. Depends on which task / model you are dealing with, you will need to choose different cleaning stratgy, i.e. the cleaning is task/model specific. For example, removing stop words is a common techniques in text cleaning. but when using models like `BERT`, it is usually a good idea to keep the stop words.

In general, the common task you can try is removing noise in the text, e.g.
- _stop words_ (e.g. i, me, the)
- `HTML` tags (e.g. `<br>`, `<p>`)
- emoji
- upper case to lower case

### Stop words
Here is an example to show some stop words provided by `NLTK`. In production, you can use Spark NLP to remove stop words.


```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to /home/jj/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
stopwords.words('english')[:10]
```




    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]



Here is a list of all languages that NLTK provides stop words support
- hungarian
- swedish
- kazakh
- norwegian
- finnish
- arabic
- indonesian
- portuguese
- turkish
- azerbaijani
- slovene
- spanish
- danish
- nepali
- romanian
- greek
- dutch
- tajik
- german
- english
- russian
- french
- italian

### Use Spark NLP to clean your text
In this section we will illustrate how to use spark NLP to clean text


```python
from pyspark.ml.feature import Imputer, OneHotEncoder, RobustScaler, StandardScaler, StopWordsRemover, StringIndexer, VectorAssembler
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from datetime import datetime
from pyspark.ml import Transformer
from pyspark import keyword_only
from pyspark.ml.param.shared import HasOutputCol, Param, Params, HasInputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
import pandas as pd
spark = SparkSession.builder \
    .appName("Spark NLP clean")\
    .master("local[12]")\
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.2.0").getOrCreate()
pd.set_option('display.max_colwidth', None)
pd.set_option('max_colwidth',500)

df = spark.read.format('csv') \
  .option("inferSchema", True) \
  .option("header", True) \
  .option("sep", ',') \
  .load('./bbc-text.csv')
```


```python
# Display first 10 rows of the content
df.limit(10).toPandas()
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
      <th>category</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tech</td>
      <td>tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room  the way people watch tv will be radically different in five years  time.  that is according to an expert panel which gathered at the annual consumer electronics show in las vegas to discuss how these new technologies will impact one of our favourite pastimes. with the us leading the trend  programmes and other content will be delivered to viewers v...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>business</td>
      <td>worldcom boss  left books alone  former worldcom boss bernie ebbers  who is accused of overseeing an $11bn (£5.8bn) fraud  never made accounting decisions  a witness has told jurors.  david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded guilty to fraud and is...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sport</td>
      <td>tigers wary of farrell  gamble  leicester say they will not be rushed into making a bid for andy farrell should the great britain rugby league captain decide to switch codes.   we and anybody else involved in the process are still some way away from going to the next stage   tigers boss john wells told bbc radio leicester.  at the moment  there are still a lot of unknowns about andy farrell  not least his medical situation.  whoever does take him on is going to take a big  big gamble.  farre...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>sport</td>
      <td>yeading face newcastle in fa cup premiership side newcastle united face a trip to ryman premier league leaders yeading in the fa cup third round.  the game - arguably the highlight of the draw - is a potential money-spinner for non-league yeading  who beat slough in the second round. conference side exeter city  who knocked out doncaster on saturday  will travel to old trafford to meet holders manchester united in january. arsenal were drawn at home to stoke and chelsea will play host to scu...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>entertainment</td>
      <td>ocean s twelve raids box office ocean s twelve  the crime caper sequel starring george clooney  brad pitt and julia roberts  has gone straight to number one in the us box office chart.  it took $40.8m (£21m) in weekend ticket sales  according to studio estimates. the sequel follows the master criminals as they try to pull off three major heists across europe. it knocked last week s number one  national treasure  into third place. wesley snipes  blade: trinity was in second  taking $16.1m (£8...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>politics</td>
      <td>howard hits back at mongrel jibe michael howard has said a claim by peter hain that the tory leader is acting like an  attack mongrel  shows labour is  rattled  by the opposition.  in an upbeat speech to his party s spring conference in brighton  he said labour s campaigning tactics proved the tories were hitting home. mr hain made the claim about tory tactics in the anti-terror bill debate.  something tells me that someone  somewhere out there is just a little bit rattled   mr howard said. ...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>politics</td>
      <td>blair prepares to name poll date tony blair is likely to name 5 may as election day when parliament returns from its easter break  the bbc s political editor has learned.  andrew marr says mr blair will ask the queen on 4 or 5 april to dissolve parliament at the end of that week. mr blair has so far resisted calls for him to name the day but all parties have stepped up campaigning recently. downing street would not be drawn on the claim  saying election timing was a matter for the prime mini...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>sport</td>
      <td>henman hopes ended in dubai third seed tim henman slumped to a straight sets defeat in his rain-interrupted dubai open quarter-final against ivan ljubicic.  the croatian eighth seed booked his place in the last four with a 7-5 6-4 victory over the british number one. henman had looked on course to level the match after going 2-0 up in the second set  but his progress was halted as the rain intervened again. ljubicic hit back after the break to seal a fourth straight win over henman. earlier ...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sport</td>
      <td>wilkinson fit to face edinburgh england captain jonny wilkinson will make his long-awaited return from injury against edinburgh on saturday.  wilkinson  who has not played since injuring his bicep on 17 october  took part in full-contact training with newcastle falcons on wednesday. and the 25-year-old fly-half will start saturday s heineken cup match at murrayfield on the bench. but newcastle director of rugby rob andrew said:  he s fine and we hope to get him into the game at some stage.  ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>entertainment</td>
      <td>last star wars  not for children  the sixth and final star wars movie may not be suitable for young children  film-maker george lucas has said.  he told us tv show 60 minutes that revenge of the sith would be the darkest and most violent of the series.  i don t think i would take a five or six-year-old to this   he told the cbs programme  to be aired on sunday. lucas predicted the film would get a us rating advising parents some scenes may be unsuitable for under-13s. it opens in the uk and ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector()\
    .setInputCols(['document'])\
    .setOutputCol('sentences')

tokenizer = Tokenizer() \
  .setInputCols(["sentences"]) \
  .setOutputCol("token")

# clean tokens 
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

finisher = Finisher() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)


pipeline = Pipeline(
    stages=[document_assembler,
            sentenceDetector,
            tokenizer,
            normalizer,
            stopwords_cleaner, 
            finisher]).fit(df)
result = pipeline.transform(df)

cleaned = result.select(['text', 'token_features'])
```


```python
result.limit(3).toPandas()
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
      <th>category</th>
      <th>text</th>
      <th>document</th>
      <th>sentences</th>
      <th>token</th>
      <th>normalized</th>
      <th>cleanTokens</th>
      <th>token_features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tech</td>
      <td>tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room  the way people watch tv will be radically different in five years  time.  that is according to an expert panel which gathered at the annual consumer electronics show in las vegas to discuss how these new technologies will impact one of our favourite pastimes. with the us leading the trend  programmes and other content will be delivered to viewers v...</td>
      <td>[(document, 0, 4332, tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room  the way people watch tv will be radically different in five years  time.  that is according to an expert panel which gathered at the annual consumer electronics show in las vegas to discuss how these new technologies will impact one of our favourite pastimes. with the us leading the trend  programmes and other content will be d...</td>
      <td>[(document, 0, 217, tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room  the way people watch tv will be radically different in five years  time., {'sentence': '0'}, []), (document, 220, 404, that is according to an expert panel which gathered at the annual consumer electronics show in las vegas to discuss how these new technologies will impact one of our favourite pastimes., {'sentence': '1'}, []), ...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, future, {'sentence': '0'}, []), (token, 10, 11, in, {'sentence': '0'}, []), (token, 13, 15, the, {'sentence': '0'}, []), (token, 17, 21, hands, {'sentence': '0'}, []), (token, 23, 24, of, {'sentence': '0'}, []), (token, 26, 32, viewers, {'sentence': '0'}, []), (token, 34, 37, with, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatre, {'sentence': '0'}, []), (token, 52, 58, systems, {'sentence':...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, future, {'sentence': '0'}, []), (token, 10, 11, in, {'sentence': '0'}, []), (token, 13, 15, the, {'sentence': '0'}, []), (token, 17, 21, hands, {'sentence': '0'}, []), (token, 23, 24, of, {'sentence': '0'}, []), (token, 26, 32, viewers, {'sentence': '0'}, []), (token, 34, 37, with, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatre, {'sentence': '0'}, []), (token, 52, 58, systems, {'sentence':...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, future, {'sentence': '0'}, []), (token, 17, 21, hands, {'sentence': '0'}, []), (token, 26, 32, viewers, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatre, {'sentence': '0'}, []), (token, 52, 58, systems, {'sentence': '0'}, []), (token, 61, 66, plasma, {'sentence': '0'}, []), (token, 68, 81, highdefinition, {'sentence': '0'}, []), (token, 84, 86, tvs, {'sentence': '0'}, []), (token, 93, 99, di...</td>
      <td>[tv, future, hands, viewers, home, theatre, systems, plasma, highdefinition, tvs, digital, video, recorders, moving, living, room, way, people, watch, tv, radically, different, five, years, time, according, expert, panel, gathered, annual, consumer, electronics, show, las, vegas, discuss, new, technologies, impact, one, favourite, pastimes, us, leading, trend, programmes, content, delivered, viewers, via, home, networks, cable, satellite, telecoms, companies, broadband, service, providers, f...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>business</td>
      <td>worldcom boss  left books alone  former worldcom boss bernie ebbers  who is accused of overseeing an $11bn (£5.8bn) fraud  never made accounting decisions  a witness has told jurors.  david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded guilty to fraud and is...</td>
      <td>[(document, 0, 1841, worldcom boss  left books alone  former worldcom boss bernie ebbers  who is accused of overseeing an $11bn (£5.8bn) fraud  never made accounting decisions  a witness has told jurors.  david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded g...</td>
      <td>[(document, 0, 181, worldcom boss  left books alone  former worldcom boss bernie ebbers  who is accused of overseeing an $11bn (£5.8bn) fraud  never made accounting decisions  a witness has told jurors., {'sentence': '0'}, []), (document, 184, 331, david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems., {'sentence': '1'}, []), (document, 333, 443, the phone company collapsed in 2002 and prosecutors ...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, left, {'sentence': '0'}, []), (token, 20, 24, books, {'sentence': '0'}, []), (token, 26, 30, alone, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, bernie, {'sentence': '0'}, []), (token, 61, 66, ebbers, {'sentence': '0'}, []), (token, 69, 71, who, {...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, left, {'sentence': '0'}, []), (token, 20, 24, books, {'sentence': '0'}, []), (token, 26, 30, alone, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, bernie, {'sentence': '0'}, []), (token, 61, 66, ebbers, {'sentence': '0'}, []), (token, 69, 71, who, {...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, left, {'sentence': '0'}, []), (token, 20, 24, books, {'sentence': '0'}, []), (token, 26, 30, alone, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, bernie, {'sentence': '0'}, []), (token, 61, 66, ebbers, {'sentence': '0'}, []), (token, 76, 82, accuse...</td>
      <td>[worldcom, boss, left, books, alone, former, worldcom, boss, bernie, ebbers, accused, overseeing, bn, bn, fraud, never, made, accounting, decisions, witness, told, jurors, david, myers, made, comments, questioning, defence, lawyers, arguing, mr, ebbers, responsible, worldcom, problems, phone, company, collapsed, prosecutors, claim, losses, hidden, protect, firm, shares, mr, myers, already, pleaded, guilty, fraud, assisting, prosecutors, monday, defence, lawyer, reid, weingarten, tried, dista...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sport</td>
      <td>tigers wary of farrell  gamble  leicester say they will not be rushed into making a bid for andy farrell should the great britain rugby league captain decide to switch codes.   we and anybody else involved in the process are still some way away from going to the next stage   tigers boss john wells told bbc radio leicester.  at the moment  there are still a lot of unknowns about andy farrell  not least his medical situation.  whoever does take him on is going to take a big  big gamble.  farre...</td>
      <td>[(document, 0, 1341, tigers wary of farrell  gamble  leicester say they will not be rushed into making a bid for andy farrell should the great britain rugby league captain decide to switch codes.   we and anybody else involved in the process are still some way away from going to the next stage   tigers boss john wells told bbc radio leicester.  at the moment  there are still a lot of unknowns about andy farrell  not least his medical situation.  whoever does take him on is going to take a bi...</td>
      <td>[(document, 0, 173, tigers wary of farrell  gamble  leicester say they will not be rushed into making a bid for andy farrell should the great britain rugby league captain decide to switch codes., {'sentence': '0'}, []), (document, 177, 323, we and anybody else involved in the process are still some way away from going to the next stage   tigers boss john wells told bbc radio leicester., {'sentence': '1'}, []), (document, 326, 426, at the moment  there are still a lot of unknowns about andy f...</td>
      <td>[(token, 0, 5, tigers, {'sentence': '0'}, []), (token, 7, 10, wary, {'sentence': '0'}, []), (token, 12, 13, of, {'sentence': '0'}, []), (token, 15, 21, farrell, {'sentence': '0'}, []), (token, 24, 29, gamble, {'sentence': '0'}, []), (token, 32, 40, leicester, {'sentence': '0'}, []), (token, 42, 44, say, {'sentence': '0'}, []), (token, 46, 49, they, {'sentence': '0'}, []), (token, 51, 54, will, {'sentence': '0'}, []), (token, 56, 58, not, {'sentence': '0'}, []), (token, 60, 61, be, {'sentence...</td>
      <td>[(token, 0, 5, tigers, {'sentence': '0'}, []), (token, 7, 10, wary, {'sentence': '0'}, []), (token, 12, 13, of, {'sentence': '0'}, []), (token, 15, 21, farrell, {'sentence': '0'}, []), (token, 24, 29, gamble, {'sentence': '0'}, []), (token, 32, 40, leicester, {'sentence': '0'}, []), (token, 42, 44, say, {'sentence': '0'}, []), (token, 46, 49, they, {'sentence': '0'}, []), (token, 51, 54, will, {'sentence': '0'}, []), (token, 56, 58, not, {'sentence': '0'}, []), (token, 60, 61, be, {'sentence...</td>
      <td>[(token, 0, 5, tigers, {'sentence': '0'}, []), (token, 7, 10, wary, {'sentence': '0'}, []), (token, 15, 21, farrell, {'sentence': '0'}, []), (token, 24, 29, gamble, {'sentence': '0'}, []), (token, 32, 40, leicester, {'sentence': '0'}, []), (token, 42, 44, say, {'sentence': '0'}, []), (token, 63, 68, rushed, {'sentence': '0'}, []), (token, 75, 80, making, {'sentence': '0'}, []), (token, 84, 86, bid, {'sentence': '0'}, []), (token, 92, 95, andy, {'sentence': '0'}, []), (token, 97, 103, farrell...</td>
      <td>[tigers, wary, farrell, gamble, leicester, say, rushed, making, bid, andy, farrell, great, britain, rugby, league, captain, decide, switch, codes, anybody, else, involved, process, still, way, away, going, next, stage, tigers, boss, john, wells, told, bbc, radio, leicester, moment, still, lot, unknowns, andy, farrell, least, medical, situation, whoever, take, going, take, big, big, gamble, farrell, persistent, knee, problems, operation, knee, five, weeks, ago, expected, another, three, month...</td>
    </tr>
  </tbody>
</table>
</div>



## Normalization
Normalization can reduce the size of dictionary, which can help improve the performance of a lot of models. _Notes:_ for models like `BERT`, `GPT` or `XLNET`, normalization is not mandatory. Also for tasks like NER, POS Tagging, we cannot use normalization.
From the high level, there are two tasks in normalization:
- stemming
- lemmalization

### Stemming
`input`: went / go / going

`output`: go

A lot of stemming process is based on `Poter Stemmer` (a rule based word transformer). Since it is rule based, it also means the output may not be an `English` word. For example, the stemmer will convert `fly / flies` to `fli`. 

### Lemmatization
Since the output of stemming may not be an English word, the goal of lemmatization is to convert the word to a true English word.

### Use Spark NLP to do text mornalization


```python
!wget -q https://raw.githubusercontent.com/mahavivo/vocabulary/master/lemmas/AntBNC_lemmas_ver_001.txt
```


```python
document_assembler = DocumentAssembler() \
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
  .setInputCols(["document"]) \
  .setOutputCol("token")

normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")

stopwords_cleaner = StopWordsCleaner()\
      .setInputCols("normalized")\
      .setOutputCol("cleanTokens")\
      .setCaseSensitive(False)

stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")


lemmatizer = Lemmatizer() \
    .setInputCols(["stem"]) \
    .setOutputCol("lemma") \
    .setDictionary("./AntBNC_lemmas_ver_001.txt", value_delimiter ="\t", key_delimiter = "->")

finisher = Finisher() \
    .setInputCols(["lemma"]) \
    .setOutputCols(["token_features"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline(
    stages=[document_assembler, 
            tokenizer,
            normalizer,
            stopwords_cleaner,
            stemmer,
            lemmatizer,
            finisher]).fit(df)
result = pipeline.transform(df)

result.limit(3).toPandas()
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
      <th>category</th>
      <th>text</th>
      <th>document</th>
      <th>token</th>
      <th>normalized</th>
      <th>cleanTokens</th>
      <th>stem</th>
      <th>lemma</th>
      <th>token_features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>tech</td>
      <td>tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room  the way people watch tv will be radically different in five years  time.  that is according to an expert panel which gathered at the annual consumer electronics show in las vegas to discuss how these new technologies will impact one of our favourite pastimes. with the us leading the trend  programmes and other content will be delivered to viewers v...</td>
      <td>[(document, 0, 4332, tv future in the hands of viewers with home theatre systems  plasma high-definition tvs  and digital video recorders moving into the living room  the way people watch tv will be radically different in five years  time.  that is according to an expert panel which gathered at the annual consumer electronics show in las vegas to discuss how these new technologies will impact one of our favourite pastimes. with the us leading the trend  programmes and other content will be d...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, future, {'sentence': '0'}, []), (token, 10, 11, in, {'sentence': '0'}, []), (token, 13, 15, the, {'sentence': '0'}, []), (token, 17, 21, hands, {'sentence': '0'}, []), (token, 23, 24, of, {'sentence': '0'}, []), (token, 26, 32, viewers, {'sentence': '0'}, []), (token, 34, 37, with, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatre, {'sentence': '0'}, []), (token, 52, 58, systems, {'sentence':...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, future, {'sentence': '0'}, []), (token, 10, 11, in, {'sentence': '0'}, []), (token, 13, 15, the, {'sentence': '0'}, []), (token, 17, 21, hands, {'sentence': '0'}, []), (token, 23, 24, of, {'sentence': '0'}, []), (token, 26, 32, viewers, {'sentence': '0'}, []), (token, 34, 37, with, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatre, {'sentence': '0'}, []), (token, 52, 58, systems, {'sentence':...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, future, {'sentence': '0'}, []), (token, 17, 21, hands, {'sentence': '0'}, []), (token, 26, 32, viewers, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatre, {'sentence': '0'}, []), (token, 52, 58, systems, {'sentence': '0'}, []), (token, 61, 66, plasma, {'sentence': '0'}, []), (token, 68, 81, highdefinition, {'sentence': '0'}, []), (token, 84, 86, tvs, {'sentence': '0'}, []), (token, 93, 99, di...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, futur, {'sentence': '0'}, []), (token, 17, 21, hand, {'sentence': '0'}, []), (token, 26, 32, viewer, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatr, {'sentence': '0'}, []), (token, 52, 58, system, {'sentence': '0'}, []), (token, 61, 66, plasma, {'sentence': '0'}, []), (token, 68, 81, highdefinit, {'sentence': '0'}, []), (token, 84, 86, tv, {'sentence': '0'}, []), (token, 93, 99, digit, {'se...</td>
      <td>[(token, 0, 1, tv, {'sentence': '0'}, []), (token, 3, 8, futur, {'sentence': '0'}, []), (token, 17, 21, hand, {'sentence': '0'}, []), (token, 26, 32, viewer, {'sentence': '0'}, []), (token, 39, 42, home, {'sentence': '0'}, []), (token, 44, 50, theatr, {'sentence': '0'}, []), (token, 52, 58, system, {'sentence': '0'}, []), (token, 61, 66, plasma, {'sentence': '0'}, []), (token, 68, 81, highdefinit, {'sentence': '0'}, []), (token, 84, 86, tv, {'sentence': '0'}, []), (token, 93, 99, digit, {'se...</td>
      <td>[tv, futur, hand, viewer, home, theatr, system, plasma, highdefinit, tv, digit, video, record, move, live, room, wai, peopl, watch, tv, radic, differ, five, year, time, accord, expert, panel, gather, annual, consum, electron, show, la, vega, discuss, new, technologi, impact, on, favourit, pastim, u, lead, trend, programm, content, deliv, viewer, via, home, network, cabl, satellit, telecom, compani, broadband, servic, provid, front, room, portabl, devic, on, talkedabout, technologi, ce, digit...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>business</td>
      <td>worldcom boss  left books alone  former worldcom boss bernie ebbers  who is accused of overseeing an $11bn (£5.8bn) fraud  never made accounting decisions  a witness has told jurors.  david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded guilty to fraud and is...</td>
      <td>[(document, 0, 1841, worldcom boss  left books alone  former worldcom boss bernie ebbers  who is accused of overseeing an $11bn (£5.8bn) fraud  never made accounting decisions  a witness has told jurors.  david myers made the comments under questioning by defence lawyers who have been arguing that mr ebbers was not responsible for worldcom s problems. the phone company collapsed in 2002 and prosecutors claim that losses were hidden to protect the firm s shares. mr myers has already pleaded g...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, left, {'sentence': '0'}, []), (token, 20, 24, books, {'sentence': '0'}, []), (token, 26, 30, alone, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, bernie, {'sentence': '0'}, []), (token, 61, 66, ebbers, {'sentence': '0'}, []), (token, 69, 71, who, {...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, left, {'sentence': '0'}, []), (token, 20, 24, books, {'sentence': '0'}, []), (token, 26, 30, alone, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, bernie, {'sentence': '0'}, []), (token, 61, 66, ebbers, {'sentence': '0'}, []), (token, 69, 71, who, {...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, left, {'sentence': '0'}, []), (token, 20, 24, books, {'sentence': '0'}, []), (token, 26, 30, alone, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, bernie, {'sentence': '0'}, []), (token, 61, 66, ebbers, {'sentence': '0'}, []), (token, 76, 82, accuse...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, left, {'sentence': '0'}, []), (token, 20, 24, book, {'sentence': '0'}, []), (token, 26, 30, alon, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, berni, {'sentence': '0'}, []), (token, 61, 66, ebber, {'sentence': '0'}, []), (token, 76, 82, accus, {'s...</td>
      <td>[(token, 0, 7, worldcom, {'sentence': '0'}, []), (token, 9, 12, boss, {'sentence': '0'}, []), (token, 15, 18, leave, {'sentence': '0'}, []), (token, 20, 24, book, {'sentence': '0'}, []), (token, 26, 30, alon, {'sentence': '0'}, []), (token, 33, 38, former, {'sentence': '0'}, []), (token, 40, 47, worldcom, {'sentence': '0'}, []), (token, 49, 52, boss, {'sentence': '0'}, []), (token, 54, 59, berni, {'sentence': '0'}, []), (token, 61, 66, ebber, {'sentence': '0'}, []), (token, 76, 82, accus, {'...</td>
      <td>[worldcom, boss, leave, book, alon, former, worldcom, boss, berni, ebber, accus, overse, bn, bn, fraud, never, make, account, decis, wit, tell, juror, david, myer, make, comment, question, defenc, lawyer, argu, mr, ebber, respons, worldcom, problem, phone, compani, collaps, prosecutor, claim, loss, hide, protect, firm, share, mr, myer, alreadi, plead, guilti, fraud, assist, prosecutor, mondai, defenc, lawyer, reid, weingarten, tri, distanc, client, alleg, cross, examin, ask, mr, myer, ever, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sport</td>
      <td>tigers wary of farrell  gamble  leicester say they will not be rushed into making a bid for andy farrell should the great britain rugby league captain decide to switch codes.   we and anybody else involved in the process are still some way away from going to the next stage   tigers boss john wells told bbc radio leicester.  at the moment  there are still a lot of unknowns about andy farrell  not least his medical situation.  whoever does take him on is going to take a big  big gamble.  farre...</td>
      <td>[(document, 0, 1341, tigers wary of farrell  gamble  leicester say they will not be rushed into making a bid for andy farrell should the great britain rugby league captain decide to switch codes.   we and anybody else involved in the process are still some way away from going to the next stage   tigers boss john wells told bbc radio leicester.  at the moment  there are still a lot of unknowns about andy farrell  not least his medical situation.  whoever does take him on is going to take a bi...</td>
      <td>[(token, 0, 5, tigers, {'sentence': '0'}, []), (token, 7, 10, wary, {'sentence': '0'}, []), (token, 12, 13, of, {'sentence': '0'}, []), (token, 15, 21, farrell, {'sentence': '0'}, []), (token, 24, 29, gamble, {'sentence': '0'}, []), (token, 32, 40, leicester, {'sentence': '0'}, []), (token, 42, 44, say, {'sentence': '0'}, []), (token, 46, 49, they, {'sentence': '0'}, []), (token, 51, 54, will, {'sentence': '0'}, []), (token, 56, 58, not, {'sentence': '0'}, []), (token, 60, 61, be, {'sentence...</td>
      <td>[(token, 0, 5, tigers, {'sentence': '0'}, []), (token, 7, 10, wary, {'sentence': '0'}, []), (token, 12, 13, of, {'sentence': '0'}, []), (token, 15, 21, farrell, {'sentence': '0'}, []), (token, 24, 29, gamble, {'sentence': '0'}, []), (token, 32, 40, leicester, {'sentence': '0'}, []), (token, 42, 44, say, {'sentence': '0'}, []), (token, 46, 49, they, {'sentence': '0'}, []), (token, 51, 54, will, {'sentence': '0'}, []), (token, 56, 58, not, {'sentence': '0'}, []), (token, 60, 61, be, {'sentence...</td>
      <td>[(token, 0, 5, tigers, {'sentence': '0'}, []), (token, 7, 10, wary, {'sentence': '0'}, []), (token, 15, 21, farrell, {'sentence': '0'}, []), (token, 24, 29, gamble, {'sentence': '0'}, []), (token, 32, 40, leicester, {'sentence': '0'}, []), (token, 42, 44, say, {'sentence': '0'}, []), (token, 63, 68, rushed, {'sentence': '0'}, []), (token, 75, 80, making, {'sentence': '0'}, []), (token, 84, 86, bid, {'sentence': '0'}, []), (token, 92, 95, andy, {'sentence': '0'}, []), (token, 97, 103, farrell...</td>
      <td>[(token, 0, 5, tiger, {'sentence': '0'}, []), (token, 7, 10, wari, {'sentence': '0'}, []), (token, 15, 21, farrel, {'sentence': '0'}, []), (token, 24, 29, gambl, {'sentence': '0'}, []), (token, 32, 40, leicest, {'sentence': '0'}, []), (token, 42, 44, sai, {'sentence': '0'}, []), (token, 63, 68, rush, {'sentence': '0'}, []), (token, 75, 80, make, {'sentence': '0'}, []), (token, 84, 86, bid, {'sentence': '0'}, []), (token, 92, 95, andi, {'sentence': '0'}, []), (token, 97, 103, farrel, {'senten...</td>
      <td>[(token, 0, 5, tiger, {'sentence': '0'}, []), (token, 7, 10, wari, {'sentence': '0'}, []), (token, 15, 21, farrel, {'sentence': '0'}, []), (token, 24, 29, gambl, {'sentence': '0'}, []), (token, 32, 40, leicest, {'sentence': '0'}, []), (token, 42, 44, sai, {'sentence': '0'}, []), (token, 63, 68, rush, {'sentence': '0'}, []), (token, 75, 80, make, {'sentence': '0'}, []), (token, 84, 86, bid, {'sentence': '0'}, []), (token, 92, 95, andi, {'sentence': '0'}, []), (token, 97, 103, farrel, {'senten...</td>
      <td>[tiger, wari, farrel, gambl, leicest, sai, rush, make, bid, andi, farrel, great, britain, rugbi, leagu, captain, decid, switch, code, anybodi, el, involv, process, still, wai, awai, go, next, stage, tiger, boss, john, well, tell, bbc, radio, leicest, moment, still, lot, unknown, andi, farrel, least, medic, situat, whoever, take, go, take, big, big, gambl, farrel, persist, knee, problem, oper, knee, five, week, ago, expect, anoth, three, month, leicest, saracen, believ, head, list, rugbi, uni...</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pyspark.sql import functions as F
result_df = result.select(F.explode(F.arrays_zip('cleanTokens.result', 'stem.result',  'lemma.result')).alias("cols")) \
.select(F.expr("cols['0']").alias("token"),
        F.expr("cols['1']").alias("stem"),
        F.expr("cols['2']").alias("lemma")).where(F.col("stem")!=F.col("lemma")).where(F.col("stem")!=F.col("token")).limit(20).toPandas()

result_df
# result_df.head(50)
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
      <th>token</th>
      <th>stem</th>
      <th>lemma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>used</td>
      <td>us</td>
      <td>we</td>
    </tr>
    <tr>
      <th>1</th>
      <td>increasing</td>
      <td>increas</td>
      <td>increa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>increased</td>
      <td>increas</td>
      <td>increa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>else</td>
      <td>els</td>
      <td>el</td>
    </tr>
    <tr>
      <th>4</th>
      <td>use</td>
      <td>us</td>
      <td>we</td>
    </tr>
    <tr>
      <th>5</th>
      <td>used</td>
      <td>us</td>
      <td>we</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sense</td>
      <td>sens</td>
      <td>sen</td>
    </tr>
    <tr>
      <th>7</th>
      <td>increase</td>
      <td>increas</td>
      <td>increa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>course</td>
      <td>cours</td>
      <td>cour</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cursing</td>
      <td>curs</td>
      <td>cur</td>
    </tr>
    <tr>
      <th>10</th>
      <td>university</td>
      <td>univers</td>
      <td>univer</td>
    </tr>
    <tr>
      <th>11</th>
      <td>increased</td>
      <td>increas</td>
      <td>increa</td>
    </tr>
    <tr>
      <th>12</th>
      <td>eases</td>
      <td>eas</td>
      <td>ea</td>
    </tr>
    <tr>
      <th>13</th>
      <td>boring</td>
      <td>bore</td>
      <td>bear</td>
    </tr>
    <tr>
      <th>14</th>
      <td>opposed</td>
      <td>oppos</td>
      <td>oppo</td>
    </tr>
    <tr>
      <th>15</th>
      <td>use</td>
      <td>us</td>
      <td>we</td>
    </tr>
    <tr>
      <th>16</th>
      <td>course</td>
      <td>cours</td>
      <td>cour</td>
    </tr>
    <tr>
      <th>17</th>
      <td>use</td>
      <td>us</td>
      <td>we</td>
    </tr>
    <tr>
      <th>18</th>
      <td>use</td>
      <td>us</td>
      <td>we</td>
    </tr>
    <tr>
      <th>19</th>
      <td>use</td>
      <td>us</td>
      <td>we</td>
    </tr>
  </tbody>
</table>
</div>



In general, pre-processing is highly depends on the data itself, downstream model and task. You need to have a good understand of your data, task and downstream model before choosing the `correct` pre-process pipeline. 


```python

```
