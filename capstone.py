import os
import json

import numpy as np
import pandas as pd
import random
# for plot
import matplotlib.pyplot as plt
import seaborn as sns
from pyprnt import prnt

from graphviz import Digraph 

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


class preprocess:
    """ arg 
    path : AI-Hub에서 다운 받으신 데이터 위치를 지정해 주시면 되요
    data_list : 위에서 위차한 폴더에 있는 모든 json파일의 이름을 리스트 형식으로 지정해준거에요
    aspect : data_list에 들어있는 모든 데이터들의 리뷰에서 뽑아낸 aspect들을 리스트 형식으로 저장했어요
    unique_asepct : 위의 aspect리스트에서 중복값 제거
    result : unique_asepct에 들어있는 값들의 언급 횟수를 {aspect : count}  형식으로 저장
    recommend_ksy : 위의 딕셔너리에서 언급횟수 상위 N개 추출
    """
    def __init__(
        self,
        path,
        category:str
    ) -> None:

        folders = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]
        
        if category in folders:
            self.path = os.path.join(path,category)
            
        self.data_list = [file for file in os.listdir(self.path) if file.endswith('.json') ]
        
        
        self.product = category
        self.result = {}
        self.recommend_key = None
        
        self.product_name = []
        self.categorized_products = {}
        self.Polarity = {}
    
    def recommend(self) -> None:
        
        aspect = []
        unique_aspect = []
        
    
        for file in self.data_list:
            with open(os.path.join(self.path,file),'r',encoding='utf-8') as f:
                data = json.load(f)
                
                for i in range(0,len(data)):
                    for ii in range(0,len(data[i]['Aspects'])):
                        aspect.append(data[i]['Aspects'][ii]['Aspect'])
                        
        for value in aspect:
            if value not in unique_aspect:
                unique_aspect.append(value)
                        
        for c in unique_aspect:
            self.result[c] = aspect.count(c)
        
        sorted_data = dict(sorted(self.result.items(), key=lambda item: item[1], reverse=True))
        self.recommend_key = dict(list(sorted_data.items())[:3])
    
    def print_review(self,aspect:str) -> list:
        
        text = []
        productname = []
        review = {}
        random_review = []
        
        for file in self.data_list:
            with open(os.path.join(self.path,file),'r',encoding='utf-8') as f:
                data = json.load(f)
                
                for i in range(0,len(data)):
                    for ii in range(0,len(data[i]['Aspects'])):
                        if data[i]['Aspects'][ii]['Aspect'] == aspect:
                            text.append(data[i]['RawText'])
                            productname.append(data[i]['ProductName'])
        
        for name,t in zip(productname,text):
            
            if name in list(review.keys()):
                review[name].append(t)
            else:
                review[name] = [t]
        # 여기는 고쳐야 해요!
        random_name = random.sample(list(review.keys()),3)
        
        for name in random_name:
            random_review.append(review[name])
        
        
        return random_name,random_review
        

        

# 수정 필요!
    def count_polarity(self):
        key1,key2,key3 = [],[],[]
        pn1,pn2,pn3 = {},{},{}
        Polarity = {}
        Sentiment = ['긍정','부정']
        key_list = list(self.recommend_key.keys())
        
        for file in self.data_list:
            with open(os.path.join(self.path,file)) as f:
                data = json.load(f)
                
                for i in range(0,len(data)):
                    for ii in range(0,len(data[i]['Aspects'])):
                        if data[i]['Aspects'][ii]['Aspect'] == key_list[0]:
                            key1.append(data[i]['Aspects'][ii]['SentimentPolarity'])
                        if data[i]['Aspects'][ii]['Aspect'] == key_list[1]:
                            key2.append(data[i]['Aspects'][ii]['SentimentPolarity'])
                        if data[i]['Aspects'][ii]['Aspect'] == key_list[2]:
                            key3.append(data[i]['Aspects'][ii]['SentimentPolarity'])
        for v in Sentiment:
            if v == '긍정':
                pn1[v] = key1.count('1')
            if v == '부정':
                pn1[v] = key1.count('-1')
                
        for v in Sentiment:
            if v == '긍정':
                pn2[v] = key2.count('1')
            if v == '부정':
                pn2[v] = key2.count('-1')
                
        for v in Sentiment:
            if v == '긍정':
                pn3[v] = key3.count('1')
            if v == '부정':
                pn3[v] = key3.count('-1')

        for key in key_list:
            if key == key_list[0]:
                self.Polarity[key] = pn1
            if key == key_list[1]:
                self.Polarity[key] = pn2
            if key == key_list[2]:
                self.Polarity[key] = pn3
    
    def plot(self):
        df = pd.DataFrame(self.Polarity)
        df = df.reset_index().melt(id_vars='index', var_name='항목', value_name='리뷰 수')
        df.columns = ['평가', '항목', '개수']
        sns.barplot(x='항목', y='개수', hue='평가', data=df)
        plt.title('항목별 긍정/부정 리뷰 수')
        plt.show()
        
    def select(self) -> None:
        
        product = []
        unique_product = []        
        for file in self.data_list:
            with open(os.path.join(self.path,file),'r',encoding='utf-8') as f:
                data = json.load(f)
                
                for i in range(0,len(data)):
                    product.append(data[i]['ProductName'])
                
                for value in product:
                    if value not in unique_product:
                        unique_product.append(value)
        self.product_name = unique_product
        
    def simple_labelling(self,product_category):
        product_category = product_category
        for category in product_category:
            for product in self.product_name:
                if category in product:
                    self.categorized_products[product] = category
            
    def to_df(self) -> pd.DataFrame:
        col = ['제품명','제품종류']
        df = pd.DataFrame(list(self.categorized_products.items()),columns=col)
        return df
        
    
    
    def summary(self):
        print('# ======================================')
        print(f'#                   {self.product}               ')
        print('# ======================================')
        print('#                   언급횟수          ')
        prnt(self.result)
        print('# ================= 상위 N ==============')
        print(self.recommend_key)
        
        