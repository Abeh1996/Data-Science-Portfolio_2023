#!/usr/bin/env python
# coding: utf-8

# ##### <center>British Airway Virtual Internship</center>
# ## <center>British Airway Data Science Virtual Internship</center>
# 
# <img src = './Downloads/Virtualnternship/QuantiumVisualizations/ba.jfif' width = '100%' height ='20%'>

# 
# # Task 1 :Web scraping and analysis
# 
# This Jupyter notebook includes some code to get you started with web scraping. We will use a package called `BeautifulSoup` to collect the data from the web. Once you've collected your data and saved it into a local `.csv` file you should start with your analysis.
# 
# ### Scraping data from Skytrax
# 
# If you visit [https://www.airlinequality.com] you can see that there is a lot of data there. For this task, we are only interested in reviews related to British Airways and the Airline itself.
# 
# If you navigate to this link: [https://www.airlinequality.com/airline-reviews/british-airways] you will see this data. Now, we can use `Python` and `BeautifulSoup` to collect all the links to the reviews and then to collect the text data on each of the individual review links.

# ## 1. Web Scraping
# 
# Data will be extracted using:
# - Python requests library to download web pages.
# - BeautifulSoup Python to parse the downloaded to an html format.

# ## Import required libraries

# In[1]:


# Importing the request libraray that will be used to download the webpages
import requests
# Importing the BeautifulSoup lirary that would be used to parse the webpages to html format
from bs4 import BeautifulSoup


# In[ ]:


get_ipython().run_cell_magic('time', '', "date = []\nreview = []\nrcmd = []\npages = 301\nimport pandas as pd\nfor page in range(1,pages):\n    url = 'https://www.airlinequality.com/airline-reviews/british-airways/page/'+str(page)+'/'\n    webpage = requests.get(url).text\n    soup = BeautifulSoup(webpage,'html.parser')\n    date_tags = soup.find_all('time')\n    review_tags=soup.find_all('div',{'class':'text_content'})\n    rcmd_tags = rcmd_tags = soup.find_all('td',{'class':['review-value rating-yes','review-value rating-no']})\n    for tag in date_tags:\n        date.append(tag.text)\n    for tag in review_tags:\n        review.append(tag.text)\n    for tag in rcmd_tags:\n        rcmd.append(tag.text)\nreview_df = pd.DataFrame({'Date':date,'Review':review,'Recommended':rcmd})\n    \n    ")


# In[2]:


import pandas as pd
#review_df.to_csv('BritishAirWay_reviews.csv',index = False)
reviews_df = pd.read_csv('BritishAirWay_reviews.csv')
reviews_df


# In[6]:


import matplotlib.pyplot as plt
plt.figure(figsize = (12,6))
import datetime as dt
def yr_month(date):
    dat = pd.to_datetime(date).strftime('%Y %b')
    return dat
reviews_df['Date'] = reviews_df['Date'].apply(yr_month)
reviews_df['Year'] = pd.to_datetime(reviews_df['Date']).dt.year
monthlyBA_passgers_df = reviews_df.groupby('Date')['Review'].size().reset_index().head(50)
plt.plot(monthlyBA_passgers_df['Date'],monthlyBA_passgers_df['Review'],color = 'g')
plt.xticks(rotation = 70)
plt.title('Monthly number of passengers from 2015-2019')
plt.xlabel('Year-Month')
plt.ylabel('No. of Passegers')
plt.show()


# ## 2. Data Wrangling

# In[7]:


#Displaying on of the reviews to identify if there are unwanted characters to be removed.
reviews_df.Review = reviews_df.Review.str.split('|',expand=True)


# In[8]:


#df = reviews_df.dropna()
#df.to_csv('BritishAirWay_reviews.csv',index = False)
df = pd.read_csv('BritishAirWay_reviews.csv')
df


# Looking a the data, it contains some unwanted characted that have to removed.

# In[9]:


# Removing un wanted characters from the using re module
import re

def clean(text):
    text = re.sub('[^A-Za-z]+',' ',str(text))
    return text
df['cleaned_Review'] = df['Review'].apply(clean)


# In[10]:


df


# Now, the data is cleaned as there are no more special characters or numbers.

# ### Natural Language Processing Tool Kit 

# In[11]:


#Import nltk
import nltk
#Importing a tokeninizer
from nltk.tokenize import word_tokenize
#import POS tagging algorithm
from nltk import pos_tag
#import stop words
from nltk.corpus import stopwords
from nltk.corpus import wordnet


# Word is used to represent Part of speech in the WordNet lexical database. WordNet is a large lexical database of English words that is used in natural language processing and computational linguistics.

# In[12]:


text = "John is tall"
tags = pos_tag(word_tokenize(text))
for word,tag in tags:
    print(word)


# In[13]:


pos_tag(word_tokenize(text))


# ### Tokenization and POS tagging

# In[14]:


POS_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

def tokenize_POS_StopWord(text):
    pos_tags = pos_tag(word_tokenize(text))
    newlist = []
    for word,tag in pos_tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word,POS_dict.get(tag[0])]))
    return newlist

df['Pos_tags'] = df['cleaned_Review'].apply(tokenize_POS_StopWord)


# In[15]:


df.head()


# ## Lemmatization

# In[16]:


# Here the stem words or lema would be obtained through the process of Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(pos_data):
    raw_lemma = " "
    for word,pos in pos_data:
        if not pos:
            raw_lemma = raw_lemma+ " "+ word
        else:
            lemma = lemmatizer.lemmatize(word , pos = pos)
            raw_lemma = raw_lemma+ " "+ lemma
    return raw_lemma
df['Lemma'] = df['Pos_tags'].apply(lemmatize)
        


# In[17]:


df.head()


# In[44]:


df_reviews = df[['cleaned_Review','Lemma']]
df_reviews.head()


# In[45]:


document = []
for row in df_reviews.Lemma:
    for word in row.split():
        document.append(word)
print(document[:50])


# In[46]:


word_counts = nltk.FreqDist(document)


# In[47]:


words = []
counts = []
for key in word_counts:
    words.append(key)
    counts.append(word_counts[key])
counts_df = pd.DataFrame({'Word':words,'Counts':counts})[:50]


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (12,6))
#sns.barplot(x = 'Word',y = 'Counts', data = counts_df)
plt.bar(counts_df.Word,counts_df.Counts, color = 'y')
plt.xticks(rotation = 90)
plt.xlabel('word counts')
plt.ylabel('Words')
plt.title('Top 50 Word Frequency Counts')
plt.show()


# In[49]:


# Creating a text variable
text = ''.join([lemma for lemma in df.Lemma])
text[:1000]


# In[52]:


#!pip install --upgrade pip
#!pip install --upgrade Pillow


# In[53]:


# Creating word_cloud with text as argument in .generate() method
#To generate a word cloud for the column df.reviews using Python 3, you can use the WordCloud library. Here is an example code snippet that shows how to do this:

#Python
#This code is AI-generated. Review and use carefully. Visit our FAQ for more information.

from wordcloud import WordCloud,STOPWORDS
stopwords = set(STOPWORDS)
import matplotlib.pyplot as plt

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=800, background_color='white', stopwords=stopwords, min_font_size=10).generate(' '.join(df.Lemma))

# Plot the WordCloud image
plt.figure(figsize=(8, 4), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)

# Show the plot
plt.show()


# ## VADER's Sentiment Analysis

# In[54]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# In[55]:


sia = SentimentIntensityAnalyzer()
sia.polarity_scores(df.Lemma[0])


# In[56]:


#for i, j in enumerate([1,2,5,6]),returns each in a list and its index
#for index,row in df_reviews.iterrows() use to interate through a dataframe it return each row in a dataframe and its index
#d1 = {'m':1,'n':3} ,d2 = {'m':7,'n':6}
#pd.DataFrame({1:d1,2:d2}) returns a data frame with indices m and n and columns columns 1 and 2

scores = {}
for i, row in df_reviews.iterrows():
    text = row['Lemma']
    ID = i
    scores[i] = sia.polarity_scores(text)
sentiment_df = pd.DataFrame(scores).T


# In[57]:


# the sentiment dataframe has 2100 records corressponding to the 2100 reviews
sentiment_df.head()


# In[58]:


df_reviews['id'] = [idx for idx in range(len(df_reviews))]
sentiment_df['id'] = [idx for idx in range(len(sentiment_df))]
merged_Revscores_df =  df_reviews.merge(sentiment_df, on = 'id',how = 'right').drop('id',axis = 1)
merged_Revscores_df.head()


# The _compound_ column gives the combined sentiment of each  of the reviews. The question which range of values for the combined sentiment is to be consider positive,neutral or negative?
# 
# Below is the standard scoring metric followed by most of the sentiment analyzers.
# 
# - Positive sentiment: compound score >= 0.05
# - Neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
# - Negative sentiment: compound score <= -0.05

# In[65]:


def sentimentAnalysis(compound):
    if compound > 0.05:
        return 'positive'
    elif compound < -0.05:
        return 'negative'
    else:
        return 'neutral'
merged_Revscores_df['Sentiment'] = merged_Revscores_df['compound'].apply(sentimentAnalysis)

Sent_df = merged_Revscores_df[['cleaned_Review','compound','Sentiment']]
Sent_df.head()


# In[66]:


# Getting the total number of positive,negative and neutral sentiments
positive = sentiment_df[sentiment_df['compound'] >= 0.05]
neutral = sentiment_df[(sentiment_df['compound'] > -0.05)&(sentiment_df['compound'] < 0.05)]
negative = sentiment_df[sentiment_df['compound'] <= -0.05]
print('pos:{} neu:{} neg:{}'.format(len(positive),len(neutral),len(negative)))


# In[67]:


#Displaying the information on a pie chart
import warnings
warnings.filterwarnings('ignore')
plt.figure(figsize = (12,6))
values = [len(positive),len(neutral),len(negative)]
labels = ['Positve','Neutral','Negative']
#colors = [ "orange","cyan", "brown","grey"] 
plt.pie(values,labels = labels, autopct = '%.1f%%',shadow = True,
       textprops={'fontsize': 14,},startangle = 40)
plt.legend()
plt.show()


# ### Task 2 : Building a Predictive Model to Understand Factors Influencing Customers Buying Behaviour 

# ### Importing important libraries

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Importing and Exploring Dataset

# In[4]:


# Read data using pandas
df = pd.read_csv('D:\sqldatasets\Foragedatasets\customer_booking.csv')


# In[6]:


# display first 5 rows
df.head()


# In[7]:


#dataset size
df.shape


# In[10]:


# column names
print(list(df.columns))


# In[11]:


# Display datatypes for each column
df.info()


# In[13]:


# Check columns containing NaN values
df.isna().sum()


# Data contains no NaN values.

# In[18]:


# Checking if dataset is balanced or not.

sns.barplot(x = df['booking_complete'].value_counts().index, y = df['booking_complete'].value_counts().values)


# The dataset is not balanced thus LogisticRegression will not be suitable for this task.Instead, RandomForest any other tree algorithm could be used.

# In[19]:


# Summary of descriptive analysis
df.describe()


# It could be observed from the descriptive analysis some features such num_passengers, purchase_lead, lead_of_stay are right skweed since their mean values are greated than their median values while feature, flight_hour is follows normal distribution.

# ### Numerical and Categorical Features

# In[29]:


#Numerical columns
num_features = df.select_dtypes( exclude = 'object').columns.to_list()
print(num_features)


# In[28]:


# Categorical columns
cat_features =   df.select_dtypes( include = 'object').columns.to_list()


# ### Correlation matrix

# In[34]:


sns.heatmap(df.corr(),cmap = 'coolwarm',annot = True)


# ### Building a RandomForest Classifier

# In[38]:


# Splitting the dataset into train and test sets.
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size = 0.25,random_state = 42)


# In[41]:


# displaying train and test size
print('train_size :{}  test_size:{}'.format(len(train_df), len(test_df)))


# In[48]:


import warnings
warnings.filterwarnings('ignore')


# ### Data preprocessing

# In[100]:


#Numerical feature scaling
from sklearn.preprocessing import MinMaxScaler

train_num_cols = list([feature for feature in num_features if feature not in ['booking_complete']])
train_target = train_df['booking_complete']
scaler = MinMaxScaler()
scaler.fit(train_df[train_num_cols])
train_df[train_num_cols] = scaler.transform(train_df[train_num_cols])


# In[101]:


train_df[train_num_cols].head()


# In[102]:


test_df[train_num_cols] = scaler.transform(test_df[train_num_cols])
test_df[train_num_cols].head()


# In[103]:


# One-hot Encoding of Categorical features
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse = False, handle_unknown = 'ignore')
encoder.fit(train_df[cat_features])
#encoder.transform(train_df[cat_features])
encoded_cols = encoder.get_feature_names(cat_features)
train_df[encoded_cols] = encoder.transform(train_df[cat_features])


# In[104]:


train_df[encoded_cols]


# In[105]:


#encoder.transform(train_df[cat_features])
encoded_cols = list(encoder.get_feature_names(cat_features))
test_df[encoded_cols] = encoder.transform(test_df[cat_features])


# In[97]:


test_df[encoded_cols]


# In[98]:


#Pocessed train and test sets
train_df[encoded_cols]


# In[99]:


train_df[encoded_cols]


# In[109]:


x_train = train_df[train_num_cols + encoded_cols]
y_train = train_df['booking_complete']
x_test = test_df[train_num_cols + encoded_cols]
y_test = test_df['booking_complete']


# ### Model Training

# In[123]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train,y_train)
print('Train_score: {}%'.format(round(model.score(x_train,y_train)*100),2))


# ### Model validation on test_set

# In[124]:


print('Train_score: {}%'.format(round(model.score(x_test,y_test)*100),2))


# ### Features Selection

# In[120]:


importances = model.feature_importances_
importance_df = pd.DataFrame({'feature':x_train.columns, 'importances':importances}).sort_values('importances',ascending = False)[:10]
sns.barplot(x = 'importances',y = 'feature',data = importance_df)
plt.title('Top 10 important features influencing customers buying behaviour')


# ## Conclusion
# From above, the model on the training set is 100% and that on the validation is 85% with a difference of 15%. This means that model overfitting occured during training. To solve this problem of model overfitting, hyperparameter tuning and feature selection using PCA will be performed.

# In[ ]:




