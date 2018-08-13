
# coding: utf-8

# In[395]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D:\\dm\\hw1\\ted-talks\\ted_main.csv')
data.info()


# In[396]:

data.head()


# In[ ]:




# In[397]:

import datetime
data['film_date'] = pd.to_datetime(data['film_date'], unit='s')
data['published_date'] = pd.to_datetime(data['published_date'], unit='s')
data[['film_date','published_date']]. head()


# In[398]:

data['ratings'].head()


# In[399]:

rating_names = set([])
import ast
# Method to read get a list of rating "names".
def split_ratings(ratings):
    val = ast.literal_eval(ratings)
    for rating in val:
        rating_names.add(rating['name'])


# In[400]:

rating_types = set([])
for record in data['ratings']:
    lists = ast.literal_eval(record)
    for entry in lists:
        rating_types.add(entry['name'])

rating_types


# In[401]:

# To count value of each rating type
def count_ratings(ratings, rating_type):
    entry = ast.literal_eval(ratings)
    for rating in entry:
        if rating['name'] == rating_type:
            return rating['count']


# In[402]:

for each_type in rating_types:
    data[each_type] = data['ratings'].apply(lambda rating : count_ratings(rating, each_type))

rating_list = list(rating_types)
data['ratings'] = data[rating_list].sum(axis=1)
rating_list.insert(0,'ratings')
data[rating_list].head()


# In[ ]:




# In[403]:

topPercent = data['views'].quantile(0.995) # taking top 0.5%
popular = data.loc[data['views'] > topPercent]
popular = popular.sort_values(by = ['views'], ascending = False)
popular[['name','views']]


# In[404]:

data['views'].describe()


# In[405]:

data['views'].plot(kind='density')


# In[406]:

import pylab 
import scipy.stats as stats
  
stats.probplot(data['views'], dist="norm", plot=pylab)
pylab.show()


# In[407]:

data.plot.scatter(x = 'views', y = 'duration')
print("correlation = "+ str(data['views'].corr(data['duration'])))


# In[408]:

df = data.drop(list(rating_types), axis=1)
import seaborn as sns
correlation = df.corr()
sns.heatmap(correlation, annot= True, linewidths=.5)


# In[409]:

data.plot.scatter(x = 'views', y = 'ratings')
print("correlation = "+ str(data['views'].corr(data['ratings'])))


# In[410]:

data.plot.scatter(x = 'comments', y = 'ratings')
print("correlation = "+ str(data['comments'].corr(data['ratings'])))


# In[411]:

data.plot.scatter(x = 'comments', y = 'views')
print("correlation = "+ str(data['comments'].corr(data['views'])))


# In[412]:

data.describe()


# In[413]:

occupation_d = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]
occupation_d.columns = ['occupation', 'appearances']
occupation_d = occupation_d.sort_values('appearances', ascending=False)
occupation_d.head(10)


# In[414]:

def splitRows(row,row_accumulator,target_column):
    split_row = row[target_column].replace('+',',').replace(" and ",",").split(",")
    for s in split_row:
        new_row = row.to_dict()
        new_row[target_column] = s.strip()
        row_accumulator.append(new_row)




# In[415]:

#to get single occupation
occupation_data = data.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]
occupation_data.columns = [['speaker_occupation', 'appearances']]

split_occupation = []
occupation_data.apply(splitRows,axis=1,args = (split_occupation,'speaker_occupation'))
occupation_single = pd.DataFrame(split_occupation)
occupation = occupation_single.groupby('speaker_occupation').sum()
occupation = occupation.sort_values('appearances', ascending=False)
occupation


# In[416]:

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(16, 10))
sns.boxplot(x='speaker_occupation', y='views', data=df[df['speaker_occupation'].isin(occupation_d.head(10)['occupation'])], palette="muted", ax =ax)
ax.set_ylim([0, 1.1e7])
plt.show()


# In[417]:


popular[list(rating_types)].plot(kind='bar', stacked=True,colormap = 'Spectral', figsize=(12,8))
plt.show()


# In[418]:

#popular[list(rating_types)].plot(kind='box',rot=90, color = 'g')
fig2, ax2 = plt.subplots(nrows=1, ncols=1,figsize=(20, 10))
sns.boxplot(data=popular[list(rating_types)], orient="h", palette="Set2", ax = ax2)
plt.show()


# In[419]:

data[['title', 'main_speaker','Funny']].sort_values('Funny', ascending=False)[:10]


# In[420]:

data['year'] = pd.DatetimeIndex(data['film_date']).year
year = pd.DataFrame(data['year'].value_counts().reset_index())
year.columns = ['year', 'talks']
plt.figure(figsize=(20,5))
sns.pointplot(x='year', y='talks', data=year)
plt.show()


# In[421]:

data['month'] = pd.DatetimeIndex(data['film_date']).month
month = pd.DataFrame(data['month'].value_counts().reset_index())
month.columns = ['month', 'talks']
import calendar
month['month'] = month['month'].apply(lambda x: calendar.month_abbr[x])
plt.figure(figsize=(20,5))
sns.barplot(x='month', y='talks', data=month)
plt.show()


# In[422]:

data['diff'] = (data['published_date'] - data['film_date'])/np.timedelta64(1, 'M')
data['diff'].describe()
data.loc[data['diff'].idxmax()]


# In[423]:

speaker = data.groupby('main_speaker').count().reset_index()[['main_speaker', 'comments']]
speaker.columns = ['main_speaker', 'appearances']
speaker = speaker.sort_values('appearances', ascending=False)
speaker.head(10)


# In[424]:

year_view = data[['year','views']]
year_view.groupby('year').sum()
year_view.plot.scatter(x = 'year',y='views')
print("correlation = "+ str(year_view['year'].corr(data['views'])))
plt.show()


# In[425]:

events = df[['title', 'event']].groupby('event').count().reset_index()
events.columns = ['event', 'talks']
events = events.sort_values('talks', ascending=False)
events.head(10)
pp = sns.barplot(x='event', y='talks', data=events.head(15))
for item in pp.get_xticklabels():
    item.set_rotation(45)
plt.show()


# In[426]:

df['languages'].describe()
data.plot.scatter(x = 'languages', y = 'views')
print("correlation = "+ str(data['views'].corr(data['languages'])))
plt.show()


# In[427]:

data['duration'].describe()
data.loc[[data['duration'].idxmin(), data['duration'].idxmax()]][['name','duration']]


# In[428]:

th = data.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
th.name = 'theme'
theme = df.drop('tags', axis=1).join(th)
theme['theme'].value_counts()


# In[442]:


df['related_talks'] = df['related_talks'].apply(lambda x: ast.literal_eval(x))
s = df.apply(lambda x: pd.Series(x['related_talks']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'related'
related_df = rdf.drop('related_talks', axis=1).join(s)
related_df['related'] = related_df['related'].apply(lambda x: x['title'])
d = dict(related_df['title'].drop_duplicates())
d = {v: k for k, v in d.items()}


# In[433]:

related_df['title'] = related_df['title'].apply(lambda x: d[x])
related_df['related'] = related_df['related'].apply(lambda x: d[x])
related_df = related_df[['title', 'related']]
related_df.head()


# In[434]:

rCount = related_df.groupby('title').count().reset_index()[['title', 'related']]
rCount.columns = ['title', 'count']
rCount = rCount.sort_values('count', ascending=False)
rCount.head(10)
related_df


# In[342]:

edges = list(zip(related_df['title'], related_df['related']))
import networkx as nx
G = nx.Graph()
G.add_edges_from(edges)
plt.figure(figsize=(25, 25))
nx.draw_networkx(G, with_labels=False, node_color= 'y')
plt.show()


# In[440]:

H=nx.DiGraph(G)
mDict = dict(H.in_degree())
degree = pd.DataFrame(list(H.in_degree().items()))
degree.columns = ['node','deg']
degree = degree.sort_values(by = ['node'], ascending = True)
degree.reset_index(drop = True)


# In[443]:

fin = df
dd = fin.join(degree)
dd


# In[445]:

dd.plot.scatter(x = 'deg',y='views')
print("correlation = "+ str(dd['deg'].corr(dd['views'])))
plt.show()

