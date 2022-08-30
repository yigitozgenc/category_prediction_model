#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from anytree import Node, RenderTree, PreOrderIter, find_by_attr, search
from gensim.models import FastText
from gensim.test.utils import common_texts  # some example sentences


# In[17]:


#df = pd.read_csv("ProductData.csv")


# In[21]:


def category_condition(df):
    cats = df.category.apply(str).tolist()
    cats2 = []
    for i in cats:
        if len(i)>30:

            cats2.append(i.lower())
    return cats2


# In[41]:


# Creating the Category Tree as Python Data structure - Anytree
def creating_cat_graph(cats2):
    lines = cats2
    root = Node("Categories")
    all_leafs = []
    for line in lines:

        line = line.split(":")
        if len(line)==1:
            line = line[0].split(" - ")
        if len(line)==1:
            line = line[0].split(" > ") 
        if len(line)==1:
            continue
        for idx, cats in enumerate(line):

            if idx == 0:
                node_old = search.find(root, lambda node: node.name == cats)
                if node_old == None:
                    node = Node(cats, parent=root)
                    all_leafs.append(cats)

                else: 
                    node = node_old
            else:
                node_old = search.find(root, lambda node: node.name == cats)
                if node_old == None:
                    node = Node(cats, parent=node)
                    all_leafs.append(cats)

                else: 
                    node = node_old    

    return root,all_leafs


# In[38]:


def show_category_graph(root):
    for pre, _, node in RenderTree(root):
                    print("%s%s" % (pre, node.name))


# In[15]:


def load_fastext(path):
    fast= FastText.load(f"{path}")
    return fast


# In[121]:


def find_category_leaf(query,all_leafs,fast,root):
    all_leafs = set(all_leafs)
    query=query.lower()
    score = 0
    final_leaf = None
    for leaf in all_leafs:
        sim_score = fast.wv.similarity(leaf, query)
        sim_score = sim_score*100
        if sim_score > score:
            final_leaf = leaf
            score += sim_score
    node = search.find(root, lambda node: node.name == f"{final_leaf}")
    if score < 60:
        return "unclassify"
    else:
        return node,score

