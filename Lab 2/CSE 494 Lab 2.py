#!/usr/bin/env python
# coding: utf-8

# ## Lab 2: Malware analysis using VirusTotal reports on malware samples
# 
# ### Objectives of the lab:
#    - See where to extract MD5 hashes from and use them to access VirusTotal API – obtain reports on the malware samples.
#    - Learn about various attributes of malwares that would be used to characterize the samples based on their families or types
#    - Learn how to convert various attributes of the malwares into feature vectors suitable for data mining
# 

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[30]:


df = pd.read_csv('../../data/lab2/lab2_md5_families.csv')


# In[31]:


df.head()


# #### The columns represent the malware families and each cell is a md5 hash

# In[128]:


df = df.iloc[:,1:]
df.head()


# #### Iterate over the columns and retrieve the VirusTotal Reports for each md5

# In[33]:


def load_requests(md5, family):
    '''
    Input: md5 hash
    Output: dictionary with the metadata key-value pairs
    
    '''
    metadata = {}
    metadata['md5'] = md5
    metadata['family'] = family
    
    params = {'apikey': 'f2f9bb5419691e82420cc5cba8ec39285eeec5a3a1bed5a459d79f66cc542b62',
              'resource': md5, 'allinfo': 1}
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "gzip,  My Python requests library example client or username"
    }
    response = requests.get('https://www.virustotal.com/vtapi/v2/file/report',
                            params=params, headers=headers)
    json_response = response.json()
    
    # if no metadata present, just skip the md5
    if 'additional_info' not in json_response:
        return {} 
    
    for f in json_response['additional_info']['exiftool']:
        metadata[f] = json_response['additional_info']['exiftool'][f]
    
    return metadata


# In[34]:


import requests

feature_list = set() # set of features (exhaustive)
md5_data = [] # list of md5 dictionaries for temporary storage


for column in df.columns:
    print(column)
    malware_family = column
    md5_list = list(df[column])
    
#     print(md5_list)
    
    for md5 in md5_list:
        print(md5)
        md5_api_report = load_requests(md5, malware_family)
        
        if len(md5_api_report) == 0:
            continue
            
        for m in md5_api_report:
            if m not in feature_list:
                feature_list.add(m)
                
        md5_data.append(md5_api_report)

# Store the metadata of all md5 in a dataframe
df_metadata = pd.DataFrame(columns=list(feature_list))
for idx in range(len(md5_data)):
    md5_dict = md5_data[idx]
    for f in feature_list:
        if f in md5_dict:
            df_metadata.loc[idx, f] = md5_dict[f]
        else:
            df_metadata.loc[idx, f] = np.nan # NaN for missing values
        
df_metadata.to_csv('../../data/lab2/md5_lab2_feat.csv')
df_metadata.head()

            
        
    


# ### Missing Value Imputation
# 
# There are a multitude of ways to impute missing values - refer http://www.stat.columbia.edu/~gelman/arm/missing.pdf for moreinformation - for example, building predictors for missing values
# 
#  scikit learn's SimpleImputerMethod is quite useful https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
# 
# $$
# \textbf{SimpleImputer}(missing\_values=nan, strategy= 'mean', fill\_value=None, verbose=0, copy=True)
# $$
# 
# 
# ### NOTE: From 0.20 onwards the Scikit learn uses this SimpleImputer Method, for versions before the equivalent method is Imputer().
# 
# For this lab,we will use a generic imputer class that handles both numeric and non-numeric attributes 

# In[59]:


columns_to_select = []

for column in df_metadata.columns:
    num_missing = (df_metadata[column]).isnull().sum()
#     print(column, num_missing)

    if num_missing < 10:
        columns_to_select.append(column)

df_metadata = df_metadata[columns_to_select]
df_metadata.to_csv('../../data/lab2/lab2_md5_feat_filtered.csv')

# manually curate features
columns_feat = ['md5', 'EntryPoint', 'family', 'LinkerVersion', 'CodeSize', 'InitializedDataSize']
df_metadata = df_metadata[columns_feat]
        
df_metadata.to_csv('../../data/lab2/lab2_md5_feat_v1.csv')

    


# In[71]:


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# In[113]:


# Impute missing values grouped by each family
from sklearn.preprocessing import Imputer

feat_data_types = ['EntryPoint', 'LinkerVersion', 'CodeSize', 'InitializedDataSize']


families = list(set(df_metadata['family']))

df_imputed = pd.DataFrame(columns=df_metadata.columns)
for f in families:
    df_family = df_metadata[df_metadata['family'] == f]
    
    df_family = DataFrameImputer().fit_transform(df_family)
    
    df_imputed = df_imputed.append(df_family)
    
# df_imputed.to_csv('../../data/lab2/lab2_md5_feat_imputed.csv')
    


# ### Malware Analysis of features by family type - comparison by family types

# In[135]:


# Analyze the numerical features family wise

df_imputed['CodeSize'] = pd.to_numeric(df_imputed['CodeSize'])
df_plot = df_imputed.groupby(['family'])['CodeSize'].mean()

df_plot.describe

# df_plot.plot.bar()


# In[120]:


df_imputed['InitializedDataSize'] = pd.to_numeric(df_imputed['InitializedDataSize'])
df_plot = df_imputed.groupby(['family'])['InitializedDataSize'].mean()

df_plot.plot.bar()


# ### Feature vectorization - for different type of features
# 

# - #### For extracting n-grams or text features, you can look into scikit-learn's methods: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# 

# In[127]:


from collections import defaultdict

def get_ngrams_dict(text, n):
    ngrams = defaultdict(int)
    for i in range(len(text)):
        ngrams[text[i:i+n]] += 1
    
    return ngrams
    


# In[131]:


# N grams for hexadecimal entry point

from collections import Counter
import operator 

data_hex = df_imputed['EntryPoint'].tolist()

ngrams_dict = defaultdict(int)
for i in range(len(data_hex)):
    data_hex[i] = (data_hex[i])[2:]
    
    ngrams_curr = get_ngrams_dict(data_hex[i], 2)
    
    input = [ngrams_dict, ngrams_curr]
    ngrams_dict = sum((Counter(ng) for ng in input),  Counter())

sorted_ngrams = sorted(ngrams_dict.items(), key=operator.itemgetter(1))[:10]

# Take the first 10  most frequent n grams and make them binary features


# In[ ]:




