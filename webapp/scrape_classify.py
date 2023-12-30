#general scraper function with summaries
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlsplit
from sqlalchemy import create_engine
import pandas as pd
from dateutil import parser
import datefinder
from datetime import datetime
import json
#try with sumy
import sumy
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
#cleaning html text
from bs4 import BeautifulSoup
from bs4.element import Comment
import urllib.request

#keywords
from summa import keywords

import os

model_path = os.path.join("classifier", "saved_models", "1701367313")

import tensorflow as tf
model=tf.saved_model.load(model_path)
#print(model)

infer=model.signatures["serving_default"]

def get_rating(input_text):
    tensortext = tf.constant([input_text], dtype=tf.string)
    labeling = infer(tensortext)
    rating=labeling["prediction"].numpy()[0][0]
    return rating

def get_gov_news(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.content,'html.parser')
    org=soup.find('dd', attrs = {'class':'gem-c-metadata__definition'}).text
    org=org.replace("\n","")
    summary=soup.find('p', attrs = {'class':'gem-c-lead-paragraph'}).text
    return org,summary

today = datetime.today()

def get_summary(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    img_tag = soup.find('img')
    if img_tag:
        img_url = img_tag.get('src')
    else:
        print("No image found on the given URL.",url)
        img_url = None
    document1 = response.content
    parser = HtmlParser.from_string(document1,url,Tokenizer("english"))
    # Using LexRank
    summarizer = LexRankSummarizer()
    #Summarize the document with 3 sentences
    summary = summarizer(parser.document, 2)
    if len(summary) >80:
        summary = summarizer(parser.document, 1)
    wordcount=len(parser.document.words)
    TR_keywords=keywords.keywords(str(parser.document.words),split=True)[:11]
    #print(TR_keywords)
    together= "".join([str(sentence) for sentence in summary])
    return [together, TR_keywords, wordcount,img_url]

def DatesToList(x):
    
    
    def nearest(items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    dates = datefinder.find_dates(x)
    
    lists = []
    
    for adate in dates:
        if today > adate:
            lists.append(adate)
    
    if len(lists) >0:
        this=nearest(lists,today)
        if this.date() > today.date():
            this=today
            return this.strftime('%d-%m-%Y')
        else:
            return this.strftime('%d-%m-%Y')
    else:
        #formatted=today.strftime('%Y-%m-%d')
        formatted="could not find date"
        return [formatted]


# printing result
#element 1 is the container, element 2 is the title with link

def get_news(url,element1,element2=None,attrib1=None,attrib2=None,cut1=0,cut2=0,cut3=0,baseurl=None,theme="blank",sourcetype="think tank"):
    r = requests.get(url)
    soup = BeautifulSoup(r.content,features="lxml")
    table = soup.findAll(element1,attrs=attrib1)
    dates=[]
    for date in table:
        if date.text is not None:
            test_string=str(date.text.strip(' ').replace('\n', ' '))
            try:
                #onlydate=parser.parse(test_string, fuzzy=True)
                onlydate = DatesToList(test_string)
                #print(test_string,onlydate[0])
                dates.append(onlydate)
            except Exception as error:
                #print(test_string,"-------------")
                #print(str(Exception))
                print(error)
                dates.append(onlydate)
    dates=dates[cut3:]
    if element2 is not None:
        elements=[title.find(element2, attrs= attrib2) for title in table]
    else:
        elements=table
    titles=[]
    for i in range(len(elements)):
        if elements[i] is not None:
            titles.append(elements[i].text.replace('\n',' '))
    links=[link.find('a', href=True) for link in elements if link is not None]
    titles=titles[cut1:]
    links=[link['href'] for link in links if link is not None]
    links=links[cut2:]
    #print(links)
    if baseurl is not None:
        fulllinks=[baseurl + link if url[:8] not in link else link for link in links]
    else:
        fulllinks=[url + link if url[:8] not in link else link for link in links] 
    package=list(zip(titles,fulllinks))
    package = list(dict.fromkeys(package))
    articles=[]
    list1, list2=list(zip(*package))
    split_url = urlsplit(url)
    source=split_url.hostname



    for i in range(len(package)):
        article_data = {}
        article_data['url'] = list2[i]
        article_data['title'] = list1[i]
        #print(list2[i])
        article_meta=get_summary(list2[i])
        article_data['summary'] = article_meta[0]
        #article_data['theme']= article_meta[1]
        article_data['theme']= theme
        article_data['source']=source
        article_data['date']=str(dates[i])
        article_data['type']=sourcetype
        article_data['wordcount']=article_meta[2]
        article_data['likes']=0
        article_data['img_url']=article_meta[3]
        #article_data['scraped_on']=str(today)
        if sourcetype=="gov.uk news":
            result=get_gov_news(list2[i])
            article_data['mlinput']=result[0]+ " " + result[1] + " " + list1[i]
            string=result[0]+ " " + result[1] + " " + list1[i]
            article_data['binary_interest']=get_rating(string)
        else:
            article_data['mlinput']="ignore"
            article_data['binary_interest']=0.5
        articles.append(article_data)
    return articles




#will need to deupe these gov ones
gov=get_news("https://www.gov.uk/search/news-and-communications?page=1","li","div",attrib1={"class":"gem-c-document-list__item"},attrib2={"class":"gem-c-document-list__item-title"},baseurl="https://www.gov.uk/",sourcetype="gov.uk news")

rawdf=pd.DataFrame.from_dict(gov)


#//[username]:[password]@[host name]/[db name]
engine = create_engine('mysql+mysqlconnector://u598492255_olliejgooding :Benoljoe99!@srv1039.hstgr.io/u598492255_policyarticles')

#print("connected")
con= engine.connect()

currentdf=pd.read_sql('articlesdb',con)

con.close()

wholedf=pd.concat([rawdf,currentdf]).drop_duplicates(['url'], keep='last')


engine = create_engine('mysql+mysqlconnector://u598492255_olliejgooding :Benoljoe99!@srv1039.hstgr.io/u598492255_policyarticles')

con1= engine.connect()

wholedf.to_sql('articlesdb',con1, if_exists='replace', index=False)

con1.close()