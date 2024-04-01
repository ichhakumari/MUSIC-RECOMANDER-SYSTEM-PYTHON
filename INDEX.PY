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


!pip install seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

plt.subplots(figsize = (15, 5))
for i, col in enumerate(floats):
    plt.subplot(3, 5, i + 1)
    sb.distplot(tracks[col])
plt.tight_layout()
plt.show()


#Due to the large dataset size, the computation cost/time will be high. The recommended system will be implemented using the most popular 10,000 songs.
tracks = tracks.sort_values(by=['popularity'], ascending=False).head(10000)

def get_similarities(song_name, data):

    # Getting vector for the input song.
    text_array1 = song_vectorizer.transform(data[data['name']==song_name]['danceability']).toarray()
    num_array1 = data[data['name']==song_name].select_dtypes(include=np.number).to_numpy()

    # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data.iterrows():
        name = row['name']
          # Getting vector for current song.
        text_array2 = song_vectorizer.transform(data[data['name']==name]['danceability']).toarray()
        num_array2 = data[data['name']==name].select_dtypes(include=np.number).to_numpy()

        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)

    return sim
       
