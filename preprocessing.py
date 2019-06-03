# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:10:32 2019

@author: lisha
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


reread = pd.HDFStore('book_events_total_view_2017-01-03.h5')


# In[3]:


reread.keys()


# In[4]:


BAC = reread['/BAC']
print(BAC.shape)
BAC.to_csv('BAC_book_events_sip-nyse_2017-01-03.csv', sep=',')


# In[5]:


BAC = BAC.sort_values('timestamp',ascending = True)
BAC


# ## Data Preprocessing: 
# ## Drop Unknown Value, Convert Timestamp, Calculate Mid-Price, Set Labels and Normalization

# In[6]:


BAC['side'].unique()


# In[7]:


BAC = BAC[BAC['side'] != 'U']


# In[8]:


BAC = BAC[BAC['book_event_type']!= 'C']


# In[9]:


BAC['side'].unique()


# In[10]:


BAC.shape


# In[11]:


df = BAC.copy()


# In[12]:


import numpy as np
df['ts'] = df.timestamp / 1000000
df['ts'] = np.floor(df['ts'])


# In[13]:


df.head()


# In[14]:


df['ts'] = pd.to_datetime(df['ts'],unit='ms')
df['ts'] = df['ts'].apply(lambda t: t.strftime('%Y-%m-%d %H:%M:%S.%f'))


# In[15]:


df.head()


# In[16]:


df = df.set_index('ts')
df.head()


# In[17]:


df.shape


# In[18]:


df1 = df.copy()


# In[19]:


df1 = df[-((df.book_event_type == 'T') & (df.aux_quantity == 0))]


# In[20]:


df1.shape


# In[21]:


df2 = df1.sort_index(ascending = True)


# In[22]:


df2['time'] = df2.index
df2.head()


# In[23]:


df3 = df2.groupby(['time','side'])['price'].max()
df4 = df2.groupby(['time','side'])['price'].min()


# In[24]:


df3.tail()


# In[25]:


df4.tail()


# In[26]:


df34 = pd.concat([df3,df4],axis=1)


# In[27]:


df34.columns = ['max','min']


# In[28]:


df34.head()


# In[30]:


midprice = [0] * len(np.unique(df34.index.get_level_values('time')))
timestamps = np.unique(df34.index.get_level_values('time'))


for i in range(len(timestamps)):
    if len(df34.loc[timestamps[i]]) == 1:
        midprice[i] = midprice[i-1]
    elif len(df34.loc[timestamps[i]]) == 2:
        sub = df34.loc[timestamps[i]]
        if sub.loc['A', 'min'] >= sub.loc['B', 'max']: 
            midprice[i] = (sub.loc['A', 'min'] + sub.loc['B', 'max']) / 2
        else: 
            midprice[i] = midprice[i-1]


# In[31]:


midprice


# In[32]:


df_new = pd.concat([pd.Series(timestamps),pd.Series(midprice)],axis=1)
df_new.columns=['time','midprice']
df_new.head()


# In[33]:


new = pd.merge(df2, df_new, left_on='ts', right_on='time', how='left', sort=False)
new.head()


# In[34]:


new = new.drop(columns='time_x')


# In[35]:


new.shape


# In[36]:


label = [0] * len(new)
label[0] = 0
m = new.midprice

for i in range(len(new)-1):
    if m[i+1] > m[i]:
        label[i+1] = 1
    elif m[i+1] == m[i]:
        label[i+1] = 0
    else:
        label[i+1] = -1


# In[37]:


label


# In[38]:


new['label'] = label


# In[39]:


new[new['label']==1]


# In[40]:


new[new['label']==-1]


# In[41]:


new['label'].value_counts()


# In[42]:


7635/new.shape[0]


# In[43]:


7557/new.shape[0]


# In[156]:


387365/new.shape[0]


# In[157]:


new.head()


# In[202]:


new['EWMA1'] = new['price'].ewm(span=30,adjust=False).mean()
new['EWMA2'] = new['price'].ewm(span=60,adjust=False).mean()
new['macd'] = new['EWMA1'] - new['EWMA2']


# In[247]:


new['sma'] = new.price.rolling(window=30).mean()
a = new['price'][:30]
new['sma'][:30] = a


# In[204]:


new = new.drop(columns='EWMA1')
new = new.drop(columns='EWMA2')


# In[404]:


new.head()


# ## There exists problems of imbalanced class.  Next step: downsample the stationary ones.

# In[405]:


down = pd.DataFrame(columns=new.columns)
down.head()


# In[406]:


len(new.timestamp)


# In[407]:


# Check whether there are continuous non-stationary values
new_label = new.label 
empty = [0] * 402712

for i in range(len(new.timestamp) - 1): 
    empty[i] = abs(new_label[i]) + abs(new_label[i + 1])
    
set(empty)


# In[408]:


new_label = new.label 

for i in range(len(new.timestamp)):
    if (new_label[i] == 1) or (new_label[i] == -1):
        down = down.append(new.iloc[(i - 1):(i + 1),:])
        
down.head()


# In[409]:


down['label'].value_counts()


# In[410]:


7635/down.shape[0]


# In[411]:


7557/down.shape[0]


# In[412]:


15192/down.shape[0]


# ## Normalization : Price level difference and Midprice change

# In[413]:


norm = down.copy()


# In[414]:


norm = norm.sort_values(by='price')
norm = norm[2:]
norm = norm.reset_index(drop=True)
norm.head()


# In[415]:


norm['diff'] = (norm['price']/norm['midprice'])-1


# In[416]:


norm.shape


# In[417]:


change = [0] * len(norm.midprice)
change[0] = 0
mid = norm['midprice']

for i in range(1,len(norm.midprice)):
    change[i] = (mid[i]/mid[i-1])-1


# In[418]:


norm['change'] = change


# In[419]:


norm['diff'] = (norm['diff'] - norm['diff'].mean())/(norm['diff'].std())


# In[420]:


norm['change'] = (norm['change'] - norm['change'].mean())/(norm['change'].std())


# In[421]:


norm.head()


# In[504]:


norm.shape


# # Baseline Model
# 

# In[422]:


from sklearn import svm


# In[423]:


new1 = norm.sort_values(by='time_y',ascending = True)


# In[424]:


new1.head()


# In[425]:


type_dummies = pd.get_dummies(new1['book_event_type'], prefix = 'type')
side_dummies = pd.get_dummies(new1['side'], prefix = 'side')


# In[426]:


new2 = new1.drop(columns = ['book_event_type', 'side'])
new2 = new2.join(type_dummies)
new2 = new2.join(side_dummies)
new2.head()


# In[427]:


new2.shape


# In[428]:


new2 = new2.reset_index()


# In[429]:


train_df = new2[:20000]
test_df = new2[20000:]


# In[430]:


# Drop aux2 because its all zeros
y_train = train_df.label
X_train = train_df.drop(['index','timestamp','time_y','order_id','label','aux2','aux1'], axis=1)
y_test = test_df.label
X_test = test_df.drop(['index','timestamp','time_y','order_id','label','aux2','aux1'], axis=1)
y_train = y_train.astype('int')


# In[505]:


X_test.shape


# # Random Forest 

# In[538]:

X_train.to_csv('x_train.csv')
y_train.to_csv('y_train.csv',header=['label'])
X_test.to_csv('x_test.csv')
y_test.to_csv('y_test.csv',header=['label'])