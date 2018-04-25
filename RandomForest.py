from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

import datetime, time

def dateToInt(x):
 _date = x.split('/')

 year, time = _date[2].split()
 hour, minute = time.split(':')

 if int(hour) > 23:
  hour = 23
 if int(hour) < 0:
  hour = 0

 t = datetime.datetime(int(year), int(_date[0]), int(_date[1]), int(hour), int(minute), 0, 0).timestamp()
 return t

def getLabelMap(data):
 # Map shape to an integer
 labels = data['shape']
 labels = labels.unique()
 nums = [i for i in range(len(labels))]
 return dict(zip(labels, nums))


from sklearn.feature_extraction.text import CountVectorizer

def vectorize(text):
 # create the transform
 vectorizer = CountVectorizer()

 # tokenize and build vocab
 vectorizer.fit(text)

 # encode document
 vector = vectorizer.transform(text)

 return vector.toarray()



from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf():
    # list of text documents
    text = ["The quick brown fox jumped over the lazy dog.",
            "The dog.",
            "The fox"]
    # create the transform
    vectorizer = TfidfVectorizer()

    # tokenize and build vocab
    vector = vectorizer.fit_transform(text)

    # # summarize
    # print(vectorizer.vocabulary_)
    # print(vectorizer.idf_)

    # # encode document
    # vector = vectorizer.transform(text)

    # summarize encoded vector
    # print(vector.shape)
    print(vector.toarray())
    print('\n')
    adf = pd.DataFrame(vector.todense())


names=['datetime', 'city', 'state', 'country',
       'shape', 'duration(seconds)', 'duration(hours/min)', 'comments',
       'date posted', 'longitude', 'latitude']
data = pd.read_csv('ufo-scrubbed-geocoded-time-standardized.csv',
                   names=names,
                   dtype=object,
                   error_bad_lines=False, warn_bad_lines=False)

data = data[data['country'] == 'us']

# Columns currently not using
# names.remove('shape')
names.remove('date posted')
names.remove('duration(hours/min)')
names.remove('country')
names.remove('state')
names.remove('city')

# Make date into ints
data['datetime'] = data.datetime.apply(lambda x: dateToInt(x))

# Filter rows that have strings in the duration column
data = data[data['duration(seconds)'].apply(lambda x : str.isdigit(x))]

# Add keys to be able to merge
data['key'] = [i for i in range(data.shape[0])]
names.append('key')

# Get rid of unicode characters
text = data["comments"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in str(x)]))

# Create dataframe out of the comment vector
commentDf = pd.DataFrame(vectorize(text))
commentDf['key'] = [i for i in range(commentDf.shape[0])]

# Merge comment dataframe and original dataframe
# data.merge(commentDf, on='key', how='left')

# data.shape[0]

#

# Since we already vectorized the comments we can get rid of them now
if 'comments' in names:
    names.remove('comments')

dt = data[names].merge(commentDf, on='key', how='left')
dt.head()


# Get x and y
print('Getting x and y...')
label_dict = getLabelMap(dt)

Y = np.array([label_dict[i] for i in dt['shape']])

x_cols = list(dt)

if 'shape' in x_cols:
    x_cols.remove('shape')

X = dt[x_cols]
print('Finished Getting x and y...')


##### Split data to train and test
print('Splitting...')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42)
print('Finished splitting')

#### Fit data
print('Fitting data...')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
print('Finished Fitting data...')

##### Predict
print('Predicting data..')
pred = rf.predict(X_test)

s = y_test
count = 0

for i in range(len(pred)):
    if pred[i] == s[i]:
        count += 1
print(count)
print('Finished Predicting data..')

print(count/len(pred))

#####