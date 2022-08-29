#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


attrs = ["amount", "freq", "avg_interval"]


# In[3]:


raw_data = pd.read_csv("2022AIGO_H_LAB測試資料_雲端行動科技.csv")


# In[4]:


data_2020 = raw_data[raw_data["datetime"].str.startswith("2020")]


# In[5]:


all_user = (data_2020.user_id.unique())


# In[6]:


data_2020.loc[:, "datetime"] = pd.to_datetime(data_2020["datetime"], format="%Y-%m-%d")
data_2020["month"] = data_2020["datetime"].dt.month


# In[7]:


df_2020 = data_2020[data_2020["item_tag"] != "優惠活動/折扣/集點"]
df_2020 = df_2020[df_2020["item_tag"] != "無法分類"]
df_2020 = df_2020[df_2020["item_tag"] != "餐飲需求"]


# In[8]:


df_2020_group = df_2020.groupby(["month", "user_id", ])


# In[9]:


df_agg = df_2020.groupby(["month", "user_id", "item_tag"])[["amount", "datetime"]].agg(
    amount = ("amount", "sum"),
    freq = ("amount", "count"),
    avg_interval = ("datetime", lambda x : x.sort_values().diff().sum().days / len(x))
)


# In[10]:



midx = pd.MultiIndex.from_product([range(1, 13), all_user])
df_ouput = pd.DataFrame(index = midx, columns=[f'{i}_{j}'  for i in df_2020.item_tag.unique() for j in attrs])
def make_output_matrix(x):
    (m, id) = (x.index.get_level_values("month")[0], x.index.get_level_values("user_id")[0])
    x = x.droplevel((0, 1))
    df_temp = x.stack()
    df_temp.index = [f"{x}_{y}" for x,y in df_temp.index]
    df_temp = df_temp.to_frame("").T
    df_ouput.loc[(m, id), df_temp.columns] = df_temp.values

df_agg.groupby(level = (0, 1)).apply(make_output_matrix)


# In[11]:


df_ouput.fillna(0, inplace=True)


# In[13]:


df_ouput.to_csv("input_matrix_2020.csv")

