#!/usr/bin/env python
# coding: utf-8

# ## BERCHMANS KEVIN S

# ## `Micro Project - Data and Visual Analytics`

# ## `Marketing Analytics`

# ### Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings('ignore')


# ### Loading Data into Dataframe

# In[2]:


df = pd.read_csv("ifood_df.csv")


# ### Exploratory Data Analysis

# In[3]:


#printing top-5 rows of matches dataset

df.head()


# In[4]:


#printing last 5 rows of the dataset

df.tail()


# In[5]:


#printing shape of the dataset

df.shape


# In[6]:


#count the values of the dataset

df.size


# In[7]:


#print info method

df.info()


# In[8]:


#printing the columns of the dataset

df.columns


# In[9]:


#printing describe method

df.describe()


# In[10]:


df.describe(include='all')


# In[11]:


#check the datatype of the dataset

df.dtypes


# In[12]:


#find duplicate values
print('Number of duplicates in the dataset:{}'.format(sum(df.duplicated())))


# In[13]:


#Checking the database for null values
null = df.isnull().values.sum()
print("Null Values: ",null)


# In[14]:


df['education_Graduation'].groupby(df['education_Graduation']).count()


# # Visualizations

# 1. Heat map ?

# In[15]:


corr = df.corr()
corr = (corr)
sns.heatmap(corr,
 xticklabels=corr.columns.values,
 yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')
corr


# 2.How many total number of plot purchases by country?

# In[16]:


df.groupby('MntGoldProds')['education_Master'].sum().sort_values(ascending=False).plot(kind='bar')
plt.title('Total Number of Purchases by Country', size=16)
plt.ylabel('Number of purchases');


# 3.Visualize the ducation_Graduation of the dataset?

# In[17]:


plt.figure(figsize=(6,6))
sizes = df.education_Graduation.value_counts()
labels = df.education_Graduation.value_counts().index
plt.title('Education_Graduation',fontsize=20)
plt.pie(sizes,colors = ['#ff9999','#66b3ff'],labels=labels,autopct='%1.1f%%',startangle=90,pctdistance=0.75,explode = (0.025,0.025))

#draw white circle
centre_circle = plt.Circle((0,0),0.60,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()


# 4. What other factors are significantly related to the amount spent on fish ?

# In[18]:


plt.figure(figsize=(8,4))
sns.distplot(df['MntFishProducts'], kde=False, hist=True, bins=12);
plt.title('MntFishProducts distribution', size=18)
plt.ylabel('count');


# 5. How many receipts in each the dataset of the marketing analytics ?

# In[19]:


plt.figure(figsize=(15,7))
sns.countplot(x='education_2n Cycle',data=df,order = df['education_2n Cycle'].value_counts().index.sort_values())
plt.xlabel('education_2n Cycle',fontsize=15)
plt.ylabel('No: of matches',fontsize=15)
plt.title('education_2n Cycle',fontsize=20)
plt.show()


# 6. Which marketing campaign is most successful?

# In[20]:


cam_success=pd.DataFrame(df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','Response']].mean()*100, columns=['Percent']).reset_index()

#plot
sns.countplot( data=cam_success.sort_values('Percent'),palette='gist_rainbow_r');
plt.xlabel('Accepted(%)');
plt.ylabel('Campaign');
plt.title('Marketing campaign success rate',size=20);


# 7. Which marketing got most of the match rewards in datasets ?

# In[21]:


plt.figure(figsize=(15,7))
player_of_match=df['MntWines'].value_counts()[:10]
sns.scatterplot(player_of_match.index,player_of_match.values,palette='Blues')
plt.title("MntWines",fontsize=22)
plt.xlabel('No of MntWines',fontsize=18)
plt.ylabel('Count',fontsize=18)
plt.show()


# 8. Display the Top 10 items of it?

# In[22]:


plt.figure(figsize = (12,8))
sns.countplot(y = 'AcceptedCmp2',data = df,order = df['AcceptedCmp2'].value_counts().iloc[:10].index,palette='hot_r')
plt.xlabel('No. of AcceptedCmp2  ',fontsize=15)
plt.ylabel('AcceptedCmp2',fontsize=15)


# 9.Income VS Kidhome ?

# In[23]:


import seaborn as sns
sns.boxplot(x="Kidhome", y="Income", hue="MntFishProducts", data=df)


# 10. education_Maste VS education_PhD ?

# In[24]:


sns.lmplot(x='education_Master', y='education_PhD', data=df, fit_reg=False, hue='MntTotal')


# In[ ]:




