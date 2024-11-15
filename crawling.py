from core import preprocess
from bs4 import BeautifulSoup
from datetime import datetime
import time
 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


import os
import json
import re
from typing import List,Literal

import numpy as np
import pandas as pd
import random
from dataclasses import dataclass, field


from jobflow import job,Flow,run_locally

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
class Crawling_for_naver(preprocess):
    threshold : int = 8
    n_pages : int = 3
    
    def __post_init__(self):
        return super().__post_init__()
    
    def chrome_driver(self,headless:bool):
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", True)
        options.add_argument("--start-maximized")
        if headless:
            options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        return driver

    def setupBeforeCrawling(self,chromeDriver):
        driver = chromeDriver
        driver.get("https://shopping.naver.com/ns/home")
        driver.implicitly_wait(5)
        
        return driver
    
    def target_product(self) -> List[str]:

        name = []
        product_name = self.product_name()

        for d in self.data:
            name.append(d.get('ProductName'))
        
        target = []
        for n in product_name:
            if name.count(n) < self.threshold:
                target.append(n)
        
        return target
    
    @job
    def get_url(self) -> dict:
        
        driver = self.chrome_driver(headless=False)
        shopping = self.setupBeforeCrawling(chromeDriver=driver)
        product_name = self.target_product()
        
        all_product_links = {}

        for product_name in product_name:
            try:
                search = shopping.find_element(By.CSS_SELECTOR, "input._searchInput_search_text_83jy9")
                search.click()
                search.clear()
                search.send_keys(product_name)
                search.send_keys(Keys.ENTER)
                shopping.implicitly_wait(3)

                product_links = []
                index = 1

                while True:
                    try:
                        product = shopping.find_element(By.XPATH, f"/html/body/div/div[4]/div[2]/div[2]/div/div/ul/li[{index}]/div/a")
                        product_link = product.get_attribute("href")
                        product_links.append(product_link)
                        index += 1
                    except NoSuchElementException:
                        break  

                all_product_links[product_name] = product_links

            except NoSuchElementException:
                all_product_links[product_name] = []
                
                
        shopping.close()
        
        return  all_product_links
    
    
    @job
    def Crawling(self,product_dict,output_filename):
        driver = self.chrome_driver(headless=False)
        shopping = self.setupBeforeCrawling(chromeDriver=driver)
        
        with open(os.path.join(self.path,output_filename), 'w', encoding='utf-8') as file:
            for product_name, url_list in product_dict.items():
                for url in url_list:
                    shopping.get(url)
                    time.sleep(3)

                    try:
                        shopping.find_element(By.CSS_SELECTOR, '#content > div > div.z7cS6-TO7X > div._27jmWaPaKy > ul > li:nth-child(2) > a').click()
                        time.sleep(3)
                    except NoSuchElementException:
                        print(f"No review tab for {product_name}")
                        continue

                    page_num = 1

                    while page_num <= self.n_pages:

                        html_source = shopping.page_source
                        soup = BeautifulSoup(html_source, 'html.parser')
                        time.sleep(3)

                        reviews = soup.findAll('li', {'class': 'BnwL_cs1av'})
                        if not reviews:
                            break

                        for review in reviews:
                            write_dt_raw = review.findAll('span', {'class': '_2L3vDiadT9'})[0].get_text()
                            write_dt = datetime.strptime(write_dt_raw, '%y.%m.%d.').strftime('%Y%m%d')

                            review_content_raw = review.find('div', {'class': '_1kMfD5ErZ6'}).find('span', {'class': '_2L3vDiadT9'}).get_text()
                            review_content = re.sub(' +', ' ', re.sub('\n', ' ', review_content_raw))

                            file.write(f"제품명: {product_name}\n")
                            file.write(f"날짜: {write_dt}, 리뷰: {review_content}\n\n")

                        try:
                            next_button = WebDriverWait(driver, 20).until(
                                EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.OWN4IvaQza[data-shp-inventory="revlist"]'))
                            )
                            next_button.click()
                            time.sleep(3)
                        except (NoSuchElementException, TimeoutException):
                            print(f"No more pages available for {product_name} or timeout")
                            break

                        page_num += 1

        shopping.close()

        
