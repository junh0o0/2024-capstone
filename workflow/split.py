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


load_dotenv()
api_key = os.getenv('API_KEY')

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
        self.reviws = []
        current_product = None
        
        with open(os.path.join(self.path,'{args.product_type}.txt'), 'r', encoding='utf-8') as f:
            data = f.readlines()

            for line in data:
                if line.startswith("제품명:"):
                    current_product = line.strip().split("제품명: ")[1]
                elif line.startswith("날짜:") and "리뷰:" in line:
                    review = re.search(r'리뷰:\s*(.*)', line).group(1)
                    self.products.append(current_product)
                    self.reviws.append(review)
    
    def gpt(self):
        response = []
        aspect = self.aspect()
        product,review_list = self.read_txt()
        for review in tqdm(self.reviws):
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
        pass
    
    
    def R2df(self):
        pass
    
    