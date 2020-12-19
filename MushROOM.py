#!/usr/bin/env python
# coding: utf-8

# ## Download data

# In[1]:


# imports required for the whole script to run
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import plot_confusion_matrix

import seaborn as sn


# In[5]:


mushrooms = pd.read_csv('mushrooms.csv')


# ## Data preparation

# In[10]:


def df_preliminary_proc(df):

    # rename the columns
    df.columns = ['classes', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor',
                         'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
                         'stalk_shape', 'stalk_root', 'stalk_surface_above_ring',
                         'stalk_surface_below_ring', 'stalk_color_above_ring',
                         'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number',
                         'ring_type', 'spore_print_color', 'population', 'habitat']

    # Drop unclear columns:
    df = df.drop(['odor', 'gill_size', 'veil_type', 'spore_print_color'], axis = 1)

    # replace '?' sign as a value
    df['stalk_root'] = df['stalk_root'].str.replace('?', 'm')

    #replace letters with meaningfull words:

    classes = {'edible': 'e', 'poisonous': 'p'}
    cap_shape = {'bell': 'b', 'conical': 'c', 'convex': 'x', 'flat': 'f', 'knobbed': 'k', 'sunken': 's'}
    cap_surface = {'fibrous': 'f', 'grooves': 'g', 'scaly': 'y', 'smooth': 's'}
    cap_color = {'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'green': 'r', 
                 'pink': 'p', 'purple': 'u', 'red': 'e', 'white': 'w', 'yellow': 'y'}
    bruises = {'yes': 't', 'no': 'f'}
    gill_attachment = {'attached': 'a', 'descending': 'd', 'free': 'f', 'notched': 'n'}
    gill_spacing = {'close': 'c', 'crowded': 'w', 'distant': 'd'}
    gill_color = {'black': 'k', 'brown': 'n', 'buff': 'b', 'chocolate': 'h', 'gray': 'g', 'green': 'r', 
                  'orange': 'o', 'pink': 'p', 'purple': 'u', 'red': 'e', 'white': 'w', 'yellow': 'y'}
    stalk_shape = {'enlarging': 'e', 'tapering': 't'}
    stalk_root = {'bulbous': 'b', 'club': 'c', 'cup': 'u', 'equal': 'e', 'rhizomorphs': 'z', 'rooted': 'r', 
                  'missing': 'm'}
    stalk_surface_above_ring = {'fibrous': 'f', 'scaly': 'y', 'silky': 'k', 'smooth': 's'}
    stalk_surface_below_ring = {'fibrous': 'f', 'scaly': 'y', 'silky': 'k', 'smooth': 's'}
    stalk_color_above_ring = {'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'orange': 'o', 
                              'pink': 'p', 'red': 'e', 'white': 'w', 'yellow': 'y'}
    stalk_color_below_ring = {'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'orange': 'o', 
                              'pink': 'p', 'red': 'e', 'white': 'w', 'yellow': 'y'}
    veil_color = {'brown': 'n', 'orange': 'o', 'white': 'w', 'yellow': 'y'}
    ring_number = {'none': 'n', 'one': 'o', 'two': 't'}
    ring_type = {'cobwebby': 'c', 'evanescent': 'e', 'flaring': 'f', 
                 'large': 'l', 'none': 'n', 'pendant': 'p', 'sheathing': 's', 'zone': 'z'}
    population = {'abundant': 'a', 'clustered': 'c', 'numerous': 'n', 'scattered': 's', 
                  'several': 'v', 'solitary': 'y'}
    habitat = {'grasses': 'g', 'leaves': 'l', 'meadows': 'm', 'paths': 'p', 'urban': 'u', 
               'waste': 'w', 'woods': 'd'}


    dict_list = [classes, cap_shape, cap_surface, cap_color, bruises,
           gill_attachment, gill_spacing, gill_color,
           stalk_shape, stalk_root, stalk_surface_above_ring,
           stalk_surface_below_ring, stalk_color_above_ring,
           stalk_color_below_ring, veil_color, ring_number, ring_type, population, habitat]

    dict_list_rev = []
    for i in dict_list:
        new_dict = dict([(value, key) for key, value in i.items()])
        dict_list_rev.append(new_dict)

    for i in range(0,19):
        df.iloc[:, i] = df.iloc[:, i].apply(lambda row: dict_list_rev[i][row])


    return df


# In[11]:


def df_final_proc(df):
    
    classes = {'edible': True, 'poisonous': False}
    cap_shape = {'bell': 1, 'conical': 2, 'convex': 3, 'flat': 4, 'knobbed': 5, 'sunken': 6}
    cap_surface = {'fibrous': 1, 'grooves': 2, 'scaly': 3, 'smooth': 4}
    cap_color = {'brown': 1, 'buff': 2, 'cinnamon': 3, 'gray': 4, 'green': 5, 'pink': 6, 'purple': 7, 
                 'red': 8, 'white': 9, 'yellow': 10}
    bruises = {'yes': 1, 'no': 0}
    gill_attachment = {'attached': 1, 'descending': 2, 'free': 3, 'notched': 4}
    gill_spacing = {'close': 1, 'crowded': 2, 'distant': 3}
    gill_color = {'black': 1, 'brown': 2, 'buff': 3, 'chocolate': 4, 'gray': 5, 'green': 6, 
                  'orange': 7, 'pink': 8, 'purple': 9, 'red': 10, 'white': 11, 'yellow': 12}
    stalk_shape = {'enlarging': 1, 'tapering': 2}
    stalk_root = {'bulbous': 1, 'club': 2, 'cup': 3, 'equal': 4, 'rhizomorphs': 5, 'rooted': 6, 
                  'missing': 0}
    stalk_surface_above_ring = {'fibrous': 1, 'scaly': 2, 'silky': 3, 'smooth': 4}
    stalk_surface_below_ring = {'fibrous': 1, 'scaly': 2, 'silky': 3, 'smooth': 4}
    stalk_color_above_ring = {'brown': 1, 'buff': 2, 'cinnamon': 3, 'gray': 4, 'orange': 5, 
                              'pink': 6, 'red': 7, 'white': 8, 'yellow': 9}
    stalk_color_below_ring = {'brown': 1, 'buff': 2, 'cinnamon': 3, 'gray': 4, 'orange': 5, 
                              'pink': 6, 'red': 7, 'white': 8, 'yellow': 9}
    veil_color = {'brown': 1, 'orange': 2, 'white': 3, 'yellow': 4}
    ring_number = {'none': 0, 'one': 1, 'two': 2}
    ring_type = {'cobwebby': 1, 'evanescent': 2, 'flaring': 3, 'large': 4, 
                 'none': 5, 'pendant': 6, 'sheathing': 7, 'zone': 8}
    population = {'abundant': 1, 'clustered': 2, 'numerous': 3, 'scattered': 4, 'several': 5, 'solitary': 6}
    habitat = {'grasses': 1, 'leaves': 2, 'meadows': 3, 'paths': 4, 'urban': 5, 'waste': 6, 'woods': 7}




    dict_list = [classes, cap_shape, cap_surface, cap_color, bruises, 
           gill_attachment, gill_spacing,  gill_color,
           stalk_shape, stalk_root, stalk_surface_above_ring,
           stalk_surface_below_ring, stalk_color_above_ring,
           stalk_color_below_ring, veil_color, ring_number, ring_type, population, habitat]

    for i in range(0,19):
        df.iloc[:, i] = df.iloc[:, i].apply(lambda row: dict_list[i][row])


    return df, dict_list


# In[12]:


def data_splitting(df):

    # Split the data into train and test dfs:
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    x_train = df_train.drop(['classes'], axis =1)  # take all variables except the price 
    y_train = df_train['classes']  # the price
    x_test = df_test.drop(['classes'], axis =1)  # take all variables except the price 
    y_test = df_test['classes']  # the price

    return x_train, y_train, x_test, y_test


# In[13]:


df = df_preliminary_proc(mushrooms)
df, dict_list = df_final_proc(df)
x_train, y_train, x_test, y_test = data_splitting(df)


# ## Modelling

# In[14]:


# Instantiate KNN learning model
model_knn = KNeighborsClassifier(n_neighbors=6)

# fit the model
model_knn.fit(x_train, y_train)

# Accuracy Score
accuracy_train = accuracy_score(y_train, model_knn.predict(x_train)) # for train
accuracy_test = accuracy_score(y_test, model_knn.predict(x_test))   # for test
f1_test = f1_score(y_test, model_knn.predict(x_test))
kappa = cohen_kappa_score(y_test, model_knn.predict(x_test))
print('Accuracy on train data: %.2f' %accuracy_train)
print('Accuracy on test data: %.2f' %accuracy_test)
print('F1-score: %.2f' %f1_test)
print('Kappa: %.2f' %kappa)
plot_confusion_matrix(model_knn, x_test, y_test, cmap = plt.cm.PuBuGn)
plt.show()




# ## Making the app

# In[2]:


import streamlit as st


# In[57]:

st.image('logo.png', channels="BGR", use_column_width = 1)

st.write("""
# Mushroom poisonousness prediction app
#### This app will help you predict if the mushroom is **edible** or not!

""")

st.write('''
##### 
##### NOTICE: 
The prediction will be updated everytime you change at least one setting.
Your final prediction is the one you see after all settings have been applied!
''')

st.sidebar.header('User Input Parameters')


# In[134]:


def user_input_features():
    cap_shape = st.sidebar.selectbox('Cap Shape', ('convex', 'flat', 'knobbed', 'bell', 'sunken', 'conical'))
    
    cap_surface = st.sidebar.selectbox('Cap Surface', ('scaly', 'smooth', 'fibrous', 'grooves'))
    
    cap_color = st.sidebar.selectbox('Cap Color', ('brown', 'gray', 'red', 'yellow', 'white', 'buff', 
                                  'pink', 'cinnamon', 'green', 'purple'))
    
    bruises = st.sidebar.selectbox('Bruises', ('no', 'yes'))
    
    gill_attachment = st.sidebar.selectbox('Gill Attachment', ('free', 'attached'))
    
    gill_spacing = st.sidebar.selectbox('Gill Spacing', ('crowded', 'close'))
    
    gill_color = st.sidebar.selectbox('Gill Color', ('white', 'pink', 'black', 'chocolate', 'purple', 'buff', 
                                   'green', 'gray', 'red', 'yellow', 'brown', 'orange'))
    
    stalk_shape = st.sidebar.selectbox('Stalk Shape',('tapering', 'enlarging'))
    
    stalk_root = st.sidebar.selectbox('Stalk Root', ('equal', 'bulbous', 'missing', 'rooted', 'club'))
    
    stalk_surface_above_ring = st.sidebar.selectbox('Stalk Surface Above Ring',( 'fibrous', 'scaly', 
                                                 'smooth', 'silky'))
    
    stalk_surface_below_ring = st.sidebar.selectbox('Stalk Surface Below Ring', ('fibrous', 'scaly', 
                                                 'smooth', 'silky'))
    
    stalk_color_above_ring = st.sidebar.selectbox('Stalk color Above Ring', ('white', 'pink', 'buff', 
                                               'cinnamon', 'gray', 'red', 'yellow', 'brown', 'orange'))
    
    stalk_color_below_ring = st.sidebar.selectbox('Stalk color Below Ring', ('white', 'pink', 'buff', 
                                               'cinnamon', 'gray', 'red', 'yellow', 'brown', 'orange'))
    
    
    veil_color = st.sidebar.selectbox('Veil Color',('brown', 'yellow', 'orange', 'white'))
    
    ring_number = st.sidebar.selectbox('Ring Number', ('one', 'none', 'two'))
    
    ring_type = st.sidebar.selectbox('Ring Type', ('flaring', 'pendant', 'large', 'evanescent', 'none'))
    
    
    population = st.sidebar.selectbox('Population', ('clustered', 'numerous', 'several', 'solitary', 
                                     'scattered', 'abundant'))
    
    habitat = st.sidebar.selectbox('Habitat', ('urban', 'waste', 'woods', 'paths', 
                                  'grasses', 'meadows', 'leaves'))
    
    
    
    data = {
        'cap_shape': cap_shape,
        'cap_surface': cap_surface,
        'cap_color': cap_color,
        'bruises': bruises,
        'gill_attachment': gill_attachment,
        'gill_spacing': gill_spacing,
        'gill_color': gill_color,
        'stalk_shape': stalk_shape,
        'stalk_root': stalk_root,
        'stalk_surface_above_ring': stalk_surface_above_ring,
        'stalk_surface_below_ring': stalk_surface_below_ring,
        'stalk_color_above_ring': stalk_color_above_ring,
        'stalk_color_below_ring': stalk_color_below_ring,
        'veil_color': veil_color,
        'ring_number': ring_number,
        'ring_type': ring_type,
        'population': population,
        'habitat': habitat,
    }
    
    features = pd.DataFrame(data, index=[0])
    return features


# In[135]:


def df_user_proc(df):
    
    dict_all = {
        'cap_shape': {'bell': 1, 'conical': 2, 'convex': 3, 'flat': 4, 'knobbed': 5, 'sunken': 6},
        'cap_surface': {'fibrous': 1, 'grooves': 2, 'scaly': 3, 'smooth': 4},
        'cap_color': {'brown': 1, 'buff': 2, 'cinnamon': 3, 'gray': 4, 'green': 5, 'pink': 6, 'purple': 7, 
                     'red': 8, 'white': 9, 'yellow': 10},
        'bruises': {'yes': 1, 'no': 0},
        'gill_attachment': {'attached': 1, 'descending': 2, 'free': 3, 'notched': 4},
        'gill_spacing': {'close': 1, 'crowded': 2, 'distant': 3},
        'gill_color': {'black': 1, 'brown': 2, 'buff': 3, 'chocolate': 4, 'gray': 5, 'green': 6, 
                      'orange': 7, 'pink': 8, 'purple': 9, 'red': 10, 'white': 11, 'yellow': 12},
        'stalk_shape': {'enlarging': 1, 'tapering': 2},
        'stalk_root': {'bulbous': 1, 'club': 2, 'cup': 3, 'equal': 4, 'rhizomorphs': 5, 'rooted': 6, 
                      'missing': 0},
        'stalk_surface_above_ring': {'fibrous': 1, 'scaly': 2, 'silky': 3, 'smooth': 4},
        'stalk_surface_below_ring': {'fibrous': 1, 'scaly': 2, 'silky': 3, 'smooth': 4},
        'stalk_color_above_ring': {'brown': 1, 'buff': 2, 'cinnamon': 3, 'gray': 4, 'orange': 5, 
                                  'pink': 6, 'red': 7, 'white': 8, 'yellow': 9},
        'stalk_color_below_ring': {'brown': 1, 'buff': 2, 'cinnamon': 3, 'gray': 4, 'orange': 5, 
                                  'pink': 6, 'red': 7, 'white': 8, 'yellow': 9},
        'veil_color': {'brown': 1, 'orange': 2, 'white': 3, 'yellow': 4},
        'ring_number': {'none': 0, 'one': 1, 'two': 2},
        'ring_type': {'cobwebby': 1, 'evanescent': 2, 'flaring': 3, 'large': 4, 
                     'none': 5, 'pendant': 6, 'sheathing': 7, 'zone': 8},
        'population': {'abundant': 1, 'clustered': 2, 'numerous': 3, 'scattered': 4, 'several': 5, 'solitary': 6},
        'habitat': {'grasses': 1, 'leaves': 2, 'meadows': 3, 'paths': 4, 'urban': 5, 'waste': 6, 'woods': 7}

        }
    

    for i in range(0,18):
        user_df.iloc[:,i] = dict_all[user_df.iloc[:,i].name][user_df.iloc[:,i].values.tolist()[0]]


    return df, dict_list


# In[ ]:


user_df = user_input_features()


# In[137]:


user_df, user_dict_list = df_user_proc(user_df)



# In[142]:


prediction = model_knn.predict(user_df)[0]


# In[ ]:


st.subheader('Our verdict for you:')


# In[145]:


if prediction == True:
    output = "This mushroom is edible! :) "
    st.success(output)
    st.image('edible.png', channels="BGR")
    
else:
    output = "This mushroom is poisonous :( "
    st.error(output)
    st.image('poisonous.png', channels="BGR", use_column_width = 1)
        

st.write('''
ATTENTION!
This prediction does not guarantee your safety!''')
    
    

    
    
    
    
    
    
    
    
