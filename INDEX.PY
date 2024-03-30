# ICHHA KUMARI
# MUSIC RECOMANDER SYSTEM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

# IMPORT DATASET
from google.colab import files


uploaded = files.upload()

import pandas as pd
import io

#store csv file in df variable..
tracks = pd.read_csv(io.BytesIO(uploaded['data (1).csv']))
print(tracks)

tracks.head()

#data cleaning
tracks.shape

tracks.info()

tracks.isnull().sum()

tracks.dropna(inplace = True)
tracks.isnull().sum().plot.bar()
plt.show()

tracks = tracks.drop(['id'], axis = 1)

tracks.head()

plt.figure(figsize = (10, 5))
sb.countplot(tracks['release_date'])
plt.axis('on')
plt.show()

#columns with float values in tracks
floats = []
for col in tracks.columns:
   if tracks[col].dtype == 'float':
        floats.append(col)

len(floats) 
