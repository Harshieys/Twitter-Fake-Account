#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tweepy
import pandas as pd

bearer_token ="###############bearer token####################"
client = tweepy.Client(bearer_token)


# In[ ]:


user=input("Enter username:")
response = client.get_users(usernames=user, user_fields=["profile_image_url",
                                                         "created_at",
                                                         "description",
                                                         "location",
                                                         "public_metrics",
                                                         "verified",
                                                         "protected",
                                                         "pinned_tweet_id",
                                                         "url"
                                                        ],tweet_fields=[])


# In[ ]:


users_df = pd.DataFrame()
for tweet in response.data:
    users_df = users_df.append(pd.DataFrame({
        'id':tweet.data["id"],
        'name':tweet.data["name"],
        'username': tweet.data["username"], 
        'created_date': tweet.data.get("created_at"),
        'profile_image':tweet.data.get("profile_image_url"),
        'bio': tweet.data.get("description"),
        'entities': tweet.entities,
        'location':tweet.data.get("location"),
        'pinned_tweet':tweet.pinned_tweet_id,
        'protected':tweet.data.get("protected"),
        'followers':tweet.public_metrics['followers_count'],
        'following':tweet.public_metrics['following_count'],
        'listed':tweet.public_metrics['listed_count'],
        'tweets':tweet.public_metrics['tweet_count'],
        'likes':tweet.data.get('likes_count'),
        'verified':tweet.data.get("verified"),
        'url':tweet.url
    },index=[1]))
    
# show the dataframe
users_df .head()


# In[ ]:


users_df.to_csv('file3.csv', index=False)


# In[ ]:


import csv
li=[]
li1=[]
with open('file3.csv','r')as file:
    csv_reader=csv.reader(file)
    for line in csv_reader:
        li.append(line[1])
        li.append(line[2])
        li.append(line[10])
        li.append(line[11])
        li.append(line[12])
        li.append(line[13])
        li.append(line[14])
print(li)
screen_name, name,followers_count,friends_count,listed_count,tweet_count,likes_count=li[7],li[8],li[9],li[10],li[11],li[12],len(li[13])
print(screen_name, name,followers_count,friends_count,listed_count,tweet_count,likes_count)


# In[ ]:


import numpy as np
import pickle
def maxSubsequence(screen_name, name): 
# find the length of the strings
    global m,n 
    m = len(screen_name) 
    n = len(name) 

# declaring the array for storing the dp values 
    global L
    L = [[None]*(n + 1) for i in range(m + 1)] 

    for i in range(m + 1): 
        for j in range(n + 1): 
            if i == 0 or j == 0 : 
                L[i][j] = 0
            elif screen_name[i-1] == name[j-1]: 
                L[i][j] = L[i-1][j-1]+1
            else: 
                L[i][j] = max(L[i-1][j], L[i][j-1]) 

# L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1] 
    return L[m][n]

def normaliseNameWeight():
    subseq = maxSubsequence(screen_name,name)
    return (subseq/len(name))

name_wt=normaliseNameWeight()

arr = np.array([[name_wt,tweet_count,followers_count,friends_count,likes_count,listed_count]])
global z
z=pd.DataFrame(arr)
print(z)
with open('pickleOutput', 'rb') as f:
    mp = pickle.load(f)

pickleTest = mp.predict(z)
if  pickleTest==0:
    auth='Real'
    print("The value of pickleTest is", auth)
else:
    auth='Fake'
    print("The value of pickleTest is", auth)


# In[ ]:




