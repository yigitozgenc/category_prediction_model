#!/usr/bin/env python
# coding: utf-8

# In[1]:


from graph_helper import *
import pandas as pd
import numpy as np
from bertopic import bertopic
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import argparse


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument("--product_name", default="jordan shoes", type=str)
args, unknown = parser.parse_known_args()


# In[ ]:


product_name = args.product_name


# In[ ]:


fastext_model = load_fastext("fastext1")
with open('root.pickle', 'rb') as handle:
    root = pickle.load(handle)
with open('all_leafs.pickle', 'rb') as handle:
    all_leafs = pickle.load(handle)


# In[ ]:


def main(product_name,fastext_model,root,all_leafs):
    return find_category_leaf(product_name,all_leafs,fastext_model,root)


# In[47]:


main(product_name,fastext_model,root,all_leafs)


# In[ ]:




