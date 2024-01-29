#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


file_path= "C:\\Users\\Best Center\\Desktop\\Fitness.csv";
ffile=pd.read_csv(file_path);


# In[3]:


print(ffile.head())


# In[4]:


print(ffile.tail(10))


# In[5]:


print(ffile.duplicated().sum())


# In[6]:


print(ffile.isnull().sum())


# In[7]:


numeric_df = ffile.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
plt.show()


# In[8]:


print(ffile.info())


# In[9]:


print(ffile.describe())


# In[10]:


mn=pd.pivot_table(ffile, 
               values=['Age','Education','Usage','Fitness','Income','Miles'], 
               index=['MaritalStatus'],
               aggfunc={
                        'Age': [min, max, np.mean, np.median],
                        'Education': [min, max, np.mean, np.median],
                        'Usage': [min, max, np.mean, np.median],
                        'Fitness': [min, max, np.mean, np.median],
                        'Income': [min, max, np.mean, np.median],
                        'Miles': [min, max, np.mean, np.median]
                       }
               )
mn.T


# In[11]:


plt.scatter(ffile['Fitness'], ffile['Miles'],color='green')
plt.xlabel('Fitness')
plt.ylabel('Miles')
plt.title('Scatter Plot: Age vs. Miles')
plt.show()


# In[12]:


plt.scatter(ffile['Age'], ffile['Miles'],color='g')
plt.xlabel('Age')
plt.ylabel('Miles')
plt.title('Scatter Plot: Age vs. Miles')
plt.show()


# In[13]:


plt.scatter(ffile['Usage'], ffile['Miles'],color='g')
plt.xlabel('Usage')
plt.ylabel('Miles')
plt.title('Scatter Plot: Age vs. Miles')
plt.show()


# In[14]:


import matplotlib.pyplot as plt
gender_counts = ffile['Gender'].value_counts()
max_gender = gender_counts.idxmax()
colors = ['orange' if gender == max_gender else 'green' for gender in gender_counts.index]
gender_counts.plot(kind='bar', color=colors)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')

plt.show()


# In[15]:


import matplotlib.pyplot as plt
product_counts = ffile['Product'].value_counts()
max_product = product_counts.idxmax()
colors = ['green' if product != max_product else 'orange' for product in product_counts.index]
product_counts.plot(kind='bar', color=colors)
plt.xlabel('Product')
plt.ylabel('Count')
plt.title('Product Distribution')

plt.show()


# In[16]:


import matplotlib.pyplot as plt
marital_counts = ffile['MaritalStatus'].value_counts()
max_marital_status = marital_counts.idxmax()
colors = ['green' if status != max_marital_status else 'orange' for status in marital_counts.index]
marital_counts.plot(kind='bar', color=colors)
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.title('Marital Status Distribution')
plt.show()



# In[17]:


sns.boxplot(x='Income', data=ffile, color='green')
plt.xlabel('Income')
plt.title('Income Distribution')
plt.show()


# In[18]:


print(ffile['Income'].mean())


# In[19]:


sns.boxplot(x='Age', data=ffile, color='green')
plt.xlabel('Age')
plt.title('Age Distribution')
plt.show()


# In[20]:


print(ffile['Age'].mean())


# In[21]:


sns.boxplot(x='Education', data=ffile, color='green')
plt.xlabel('Education')
plt.title('Education Distribution')
plt.show()


# In[22]:


print(ffile['Education'].mean())


# In[23]:


sns.violinplot(x='Education', y='Usage', data=ffile)
plt.xlabel('Education')
plt.ylabel('Usage')
plt.title('Violin Plot: Education vs. Usage')
plt.show()


# In[24]:


sns.lineplot(x='Age', y='Income', data=ffile, marker='o', color='green')
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Income Trend with Age')
plt.show()


# In[25]:


sns.regplot(x=ffile['Income'], y=ffile['Miles'])


# In[26]:


sns.lmplot(x="Income", y="Miles", hue="Gender", data=ffile);


# In[27]:


sns.lmplot(x="Income", y="Miles", hue="MaritalStatus", data=ffile);


# In[28]:


sns.countplot(x="Product", hue="Gender", data=ffile)


# In[29]:


pr_cp = sns.countplot(x="Product", hue="MaritalStatus", data=ffile)


# In[30]:


import plotly.express as px


# In[31]:


px.scatter_3d(
    ffile, 
    x='Age', 
    y='Usage', 
    z='Income',
    color='MaritalStatus'
)


# In[32]:


plt.figure(figsize=(12, 8))
markers = {"Male": "^", "Female": "o"}
sns.scatterplot(data=ffile, x='Age', y='Income', hue='Product', style='Gender', palette=['#1b0c41', '#fb9b06', '#cf4446'], s= 120, markers=markers)

plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Age and Income Distribution by Product and Gender')

plt.legend(title='Product')
plt.show()


# In[33]:


plt.figure(figsize=(12, 8))
markers = {"Male": "^", "Female": "o"}
sns.scatterplot(data=ffile, x='Fitness', y='Income', hue='Product', style='Gender', palette=['#1b0c41', '#fb9b06', '#cf4446'], s= 120, markers=markers)

plt.xlabel('Fitness')
plt.ylabel('Income')
plt.title('Age and Income Distribution by Product and Gender')

plt.legend(title='Product')
plt.show()


# In[34]:


plt.figure(figsize=(12, 8))
markers = {"Male": "^", "Female": "o"}
sns.scatterplot(data=ffile, x='Miles', y='Income', hue='Product', style='Gender', palette=['#1b0c41', '#fb9b06', '#cf4446'], s= 120, markers=markers)

plt.xlabel('Miles')
plt.ylabel('Income')
plt.title('Age and Income Distribution by Product and Gender')

plt.legend(title='Product')
plt.show()


# In[ ]:





# In[ ]:




