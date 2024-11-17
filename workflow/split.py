import os
import re
import numpy as np 
import pandas as pd
import openai
import requests


from tqdm import tqdm 
from .Core import preprocess
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import List,Literal


from jobflow import job

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
class split_sentence(preprocess):
    
    def __post_init__(self):
        super().__post_init__()
        
        self.products = []
        self.reviews = []
        current_product = None
        if os.path.isfile(os.path.join(self.path,f'{self.product_type}.txt')):
            with open(os.path.join(self.path,f'{self.product_type}.txt'), 'r', encoding='utf-8') as f:
                data = f.readlines()

                for line in data:
                    if line.startswith("제품명:"):
                        current_product = line.strip().split("제품명: ")[1]
                    elif line.startswith("날짜:") and "리뷰:" in line:
                        review = re.search(r'리뷰:\s*(.*)', line).group(1)
                        self.products.append(current_product)
                        self.reviews.append(review)
        
    def gpt(self):
        load_dotenv()
        api_key = os.getenv('API_KEY')
        openai.api_key = api_key
        
        response = []
        aspect = self.aspect()
        for review in tqdm(self.reviews):
            messages = [
                {"role": "system", "content": "너는 리뷰 데이터를 {aspect} 이라는 aspect로만 분리하여 <aspect: 문장> 이 형식으로 보여주는 챗봇이야. aspect에 없는 aspect는 절대 알려주지마. 다른 aspect는 언급하지마. "},
                {"role": "user", "content": f"리뷰 데이터: '{review}' \n이 리뷰를 {aspect} 로만 분리해서 알려줘. 각각 <aspect: 문장> 무조건 이 형태로 분리해서 응답해줘. 절대 < > 이것을 빼먹지마."}
            ]
            completion = openai.ChatCompletion.create(
                    model="ft:gpt-3.5-turbo-0125:personal:3:AQVJbNZG",
                    messages=messages,
                    max_tokens=150,
                    n=1,
                    stop=None,
                    temperature=0
                )
            response.append(completion.choices[0].message.content)
            


        return response 
    
    
    def R2dict(self):
        review_dict = {}
        response = self.gpt()
        for review, response_text in zip(self.reviews, response):
    
            aspects = {}
            matches = re.findall(r'<\s*["\']?([\w가-힣/]+)["\']?\s*:\s*["\']?([^<>]+?)["\']?\s*>', response_text)

            for aspect, detail in matches:
                aspects[aspect.strip()] = detail.strip()
    
    
        review_dict[review] = aspects
        
        return review_dict
    def R2df(self):
        response = self.gpt()
        data = []
        for review, response_text in zip(self.reviews, response):
    
            aspects = {}
            matches = re.findall(r'<\s*["\']?([\w가-힣/]+)["\']?\s*:\s*["\']?([^<>]+?)["\']?\s*>', response_text)


            for aspect, detail in matches:
                if len(detail.strip()) > 5: 
                    data.append({
                    "제품명" : self.product_name,
                    "원본리뷰": self.reviews,
                    "특성": aspect,
                    "나뉜리뷰": detail
                    })
                
        df = pd.DataFrame(data)
        df.to_csv('{self.product_type}.csv')
    
    