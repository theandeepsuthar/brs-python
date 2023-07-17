#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


books=pd.read_csv('books.csv')
users=pd.read_csv('users.csv')
ratings=pd.read_csv('ratings.csv')


# In[3]:


books


# In[4]:


users


# In[5]:


ratings


# In[6]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[7]:


books.isnull().sum()


# In[8]:


users.isnull().sum()


# In[9]:


ratings.isnull().sum()

books.duplicated().sum()
# In[10]:


users.duplicated().sum()


# In[11]:


ratings.duplicated().sum()


# # popularity based recommander system

# In[12]:


ratings_with_name=ratings.merge(books,on='ISBN')


# In[13]:


num_rating_df=ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
num_rating_df


# In[14]:


avg_rating_df=ratings_with_name.groupby('Book-Title').mean(['Book-Ratings']).reset_index()
avg_rating_df.rename(columns={'Book-Rating':'avg_rating'},inplace=True)
print(avg_rating_df)


# In[15]:


avg_rating_df.drop('User-ID',axis=1,inplace=True)


# In[16]:


avg_rating_df


# In[17]:


popular_df=num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[18]:


popular_df=popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)


# In[ ]:





# In[19]:


popular_df=popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# In[20]:


popular_df['Image-URL-M'][0]


# ##collaborative filtering based on recommandar system

# In[21]:


x=ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
padhe_likhe_user=x[x].index


# In[22]:


filtered_rating=ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_user)]


# In[23]:


y=filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books=y[y].index


# In[24]:


final_rating=filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[25]:


pt=final_rating.pivot_table(index='Book-Title',columns='User-ID',values="Book-Rating")


# In[26]:


pt.fillna(0,inplace=True)


# In[27]:


pt


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity


# In[29]:


similarity_score=cosine_similarity(pt)


# In[30]:


similarity_score.shape


# In[31]:


def recommend(book_name):
  #index fetch
  index=np.where(pt.index==book_name)[0][0]
  similar_item=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
  data=[]
  for i in similar_item:
        item=[]
        print(pt.index[i[0]])
        temp_df=books[books['Book-Title']==pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
        
  return data


# In[32]:


recommend('1984')


# In[33]:


import pickle 
pickle.dump(popular_df,open('popular.pkl','wb'))


# In[34]:


pickle.dump(pt,open('pt.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))
pickle.dump(similarity_score,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




