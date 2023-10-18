# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import numpy as np

import streamlit as st
from streamlit.hello.utils import show_code

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from transformers import pipeline
# create pipeline for NER
ner = pipeline('ner', aggregation_strategy = 'simple', model="iguanodon-ai/bert-base-finnish-uncased-ner")


### Real deal 
#### Get an article
import requests
import json

def article_demo():

    def get_article_data():
        response_API = requests.get('https://articles.api.yle.fi/v2/articles.json?app_id=HIEKKALAATIKKO&app_key=HIEKKALAATIKKO&id=74-20054878')
        jsondata = response_API.text
        parse_json = json.loads(jsondata)
        
        data=[]
        for q in parse_json['data'][0]['content']:
            if q['type']=='text':
                data.append(q['text'])

        return data

    def get_ner():
        # using list comprehension
        listToStr = ' '.join(map(str, data))

        tulos = ner(listToStr)
        tulospd =pd.DataFrame(tulos)

        return tulospd.head()
    

    try:
        #### get only text
        data = get_article_data()
        tulos = get_ner()
        
    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**
            Connection error: %s
        """
            % e.reason
        )       
        
    

st.set_page_config(page_title="Article Demo", page_icon="📊")
st.markdown("# Plotting Demo")
st.sidebar.header("Plotting Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

article_demo()

show_code(article_demo)
