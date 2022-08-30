#!/usr/bin/env python
# coding: utf-8

# In[40]:


from graph_helper import *
import pandas as pd
import numpy as np


# In[41]:


freq = pd.read_csv("category_predictions.csv")


# In[42]:


bertopic = {"book & magazine":["book","hardcover","paperback","novel",'crewneck',
 'waking',
 'rising',
 'wolven',
 'deceptive',
 'wayfinder',
 'oath',
 'overwatch',
 'leo',
 'twelve'],
 "jewelry & accessories":["ring","diamond","bracelet","locket","quartz","earrings"],
 "Electronics":["clockwork","wayfinder","android","moderoid","code"],
 "women's fashion":["dress","offshoulder","polka","sparkly","maxi"],
 "skin care":["moisturizer","retinol","serum","hyaluronic","acne","toner","antiaging","skin",
              "spots","lotion",'shampoo',
 'conditioner',
 'hair',
 'keratin',
 'redken',
 '338',
 'taming',
 'gkhair',
 'biolage',
 'dht'],
 "vitamins & dietary supplements":["vitamin","calcium","capsules","magnesium","tablets","d3","mg","iu","immune","supplement","nurse","11oz"],
"home & garden":['homewares',
 'atlas',
 'successi',
 'aged',
 'eurotech',
 'craftsman',
 'browning',
 'fulcrum',
 'oskar',
 'sutton'],
"yoga":['yoga',
 'leggings',
 'capri',
 'gimnasio',
 'spiderweb',
 'scorpio',
 'rawr',
 'dreamcatcher',
 'scattered',
 'cracked'],
"tea & infusions":['tea',
 'salada',
 '20ct',
 'teabags',
 'iced',
 'chai',
 'tazo',
 'guayaki',
 'mate',
 'matcha']}

