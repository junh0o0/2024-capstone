import os
import json
from typing import List,Literal

import numpy as np
import pandas as pd
import random
from dataclasses import dataclass, field

UserChoiceType = (
    Literal[
        "가습기",
        "공기청정기",
        "매트",
        "보일러",
        "선풍기",
        "에어컨",
        "온수기",
        "제습기",
        "히터"
    ] | None
)


@dataclass
class preprocess:
    path : str
    category : str 
    choice_type : str
    product_type : UserChoiceType = None 
    
    def __post_init__(self) -> None:
        
        folders = [x for x in os.listdir(self.path) if os.path.isdir(os.path.join(self.path,x))]
        
        if self.category in folders:
            type = [x for x in os.listdir(os.path.join(self.path,self.category)) if os.path.isdir(os.path.join(self.path,self.category,x))]
    
        if self.choice_type in type:
            self.path = os.path.join(self.path,self.category,self.choice_type)
             
        if  self.product_type is not None:
            self.path = os.path.join(self.path,self.product_type)
        
        
        data_list =  [file for file in os.listdir(self.path) if file.endswith('.json')]
        with open(os.path.join(self.path,data_list[0]),"r",encoding='utf-8') as f:
            self.data = json.load(f)    
    
    
    def review(self) -> List[str]:
        
        review_list = []
        
        for d in self.data:
            review_list.append(d.get("RawText"))
            
            
        return review_list
    
    
    def aspect(self) -> List[str]:
        
        aspect = []
        unique_aspect = []
        
    
        for d in self.data:
            for a in d.get("Aspects", []):
                aspect.append(a.get('Aspect'))
                
                
        for value in aspect:
            if value not in unique_aspect:
                unique_aspect.append(value)
        
        return unique_aspect
    
    
    def product_name(self) -> List[str]:
        
        name = []
        unique_name = []
        
        for d in self.data:
            name.append(d.get('ProductName'))
            
            
        for value in name:
            if value not in unique_name:
                unique_name.append(value)
                
        return unique_name
@dataclass
class recomend(preprocess):    
    def __post_init__(self):
        return super().__post_init__() 
    
    
    def recomend(self):
        col = self.aspect()
        row = self.product_name()
        table = np.zeros((len(row),len(col)))
        
        for i in range(len(row)):
            aspect_dict = {} 
            for d in self.data:
                if d.get("ProductName") == row[i]:
                    for a in d.get("Aspects"):
                        key = a.get('Aspect')
                        value = int(a.get('SentimentPolarity'))           
                    if key in aspect_dict:
                        aspect_dict[key] += value
                    else:
                        aspect_dict[key] = value
                            
            for index,aspect in enumerate(col):
                for key,val in aspect_dict.items():
                    if key == aspect:
                        table[i][index] = val
    
                
        return table
    
    
    def normalize(self):
        table = self.recomend()
        row_sums = table.sum(axis=1, keepdims=True)
        normalized_data = np.zeros_like(table, dtype=float)
        for i in range(len(row_sums)):
            if row_sums[i] != 0:
                normalized_data[i] = table[i] / row_sums[i]
            else:
                normalized_data[i] = table[i]
                
        return normalized_data
    
    
    
    
    def final_recomend(self,user_input):
        data = self.normalize()
        a = np.dot(data,user_input.T)
        rr = np.argsort(a)[-3:]
        
        for ind,product in enumerate(self.row):
            for i in rr:
                if ind == i:
                    print(product)