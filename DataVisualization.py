#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


trendingVideos = pd.read_csv ('C:/Users/pc/Desktop/Youtube/TrendingVideosData.csv')


# In[3]:


trendingVideos.head()


# In[6]:


publish_time = pd.to_datetime(trendingVideos['publishedAt'])

print(publish_time.dt.hour)


# In[5]:


publish_time = pd.to_datetime(trendingVideos['publishedAt'])

hour = (publish_time.dt.hour)

numberOfTrendingVideosByHour = hour.groupby(hour).count()
numberOfTrendingVideosByHour = numberOfTrendingVideosByHour.to_dict()

plt.figure(figsize=(15,4))
plt.title('Numri i videove në trend në raport me orën e publikimit')
sns.barplot(x = list(numberOfTrendingVideosByHour.keys()), y = list(numberOfTrendingVideosByHour.values()))


# In[6]:


publish_time = pd.to_datetime(trendingVideos['publishedAt'])

day = (publish_time.dt.day_name())

numberOfTrendingVideosByDay = hour.groupby(day).count()
numberOfTrendingVideosByDay = numberOfTrendingVideosByDay.to_dict()

plt.figure(figsize=(15,4))
plt.title('Numri i videove në trend në raport me ditën e publikimit')
sns.barplot(x = list(numberOfTrendingVideosByDay.keys()), y = list(numberOfTrendingVideosByDay.values()))


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer


# In[8]:


count_vectorizer = CountVectorizer()

trendingVideos['channelTitle'] = trendingVideos['channelTitle'].str.replace(r'[^\x00-\x7F]+', '')

#create a matrix 
count_matrix = count_vectorizer.fit_transform(trendingVideos['channelTitle']).todense()

#create an object of titles words
map_index_to_word = {i: x for i, x in enumerate(count_vectorizer.get_feature_names())}


# In[9]:


import numpy as np

df_words = pd.DataFrame()
df_words = pd.concat([df_words, pd.DataFrame(count_matrix)], axis = 0)

#change columns with titles
df_words.columns = map_index_to_word.values()

#count total for each column
df_words.loc['total'] = df_words.select_dtypes(np.number).sum()

#sort columns by frequency
df_sorted = df_words.sort_values(by = 'total', axis = 1, ascending = False)


# In[10]:


import nltk
from nltk.corpus import stopwords

df_sorted = df_sorted.loc[:, ~df_sorted.columns.isin(stopwords.words('english'))]


plt.figure(figsize=(20,4))
plt.title('Fjalët më të përdorura në titujt e videove në trend')
sns.barplot(x = list(df_sorted.columns)[:20], y = list(df_sorted.loc['total'])[:20])


# In[4]:


def contains_capitalized_word(s):
    for w in s.split():
        if w.isupper():
            return True
    return False


trendingVideos["contains_capitalized"] = trendingVideos["title"].apply(contains_capitalized_word)

value_counts = trendingVideos["contains_capitalized"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['Jo', 'Po'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Titulli përmban shkronjë të madhe')


# In[11]:


trendingVideos["title_length"] =trendingVideos["title"].apply(lambda x: len(x))

PLOT_COLORS = ["#268bd2"]

fig, ax = plt.subplots()
_ = sns.distplot(trendingVideos["title_length"], kde=False, rug=False, 
                 color=PLOT_COLORS[0], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Gjatësia e titullit", ylabel="Numri i videove", xticks=range(0, 110, 10))
_ = ax.set_title('Numri i videove në raport me gjatësinë e titullit')


# In[ ]:




