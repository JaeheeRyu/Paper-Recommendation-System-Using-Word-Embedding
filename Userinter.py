import requests
from bs4 import BeautifulSoup
import re
import nltk

def craw(url_list):
    '''
    :param url: 문서명 리스트를 받아옴
    :return: 문서 별 링크, 제목, 저자, 초록을 담은 2차원 리스트 반환
    '''
    inter_list = []
    for link in url_list:
        doclink = 'https://arxiv.org/abs/' + link.replace('.txt','')
        result = requests.get(doclink)
        soup = BeautifulSoup(result.text, 'html.parser')
        Title = soup.find('h1', class_='title mathjax').text.replace("Title:",'')
        Authers = soup.find('div', class_='authors').text.replace("Authors:", '')
        Abstract = soup.find('blockquote', class_='abstract mathjax').text.replace("Abstract:  ", '').replace("\n", " ")
        doc_sum = [Title, Authers, Abstract, doclink]
        inter_list.append(doc_sum)
    return inter_list

def preprocess(doc):
    # Step2: Convert to Lowercase
    #lowertext = doc.lower()

    # Step3: Normalize (remove numbers, some punctuation)
    wo_nl = re.sub(r"-\n+", "", doc)  # 하이픈 뒤에 같이 따라오는 개행문자 삭제
    wo_nl = wo_nl.replace("\n", " ")  # 개행문자 띄어쓰기로 대체
    only_text = ''.join(i for i in doc if i.isalpha() or i in ['.', ' ', '-', '?', '!'])  # 알파벳과 해당 기호들만 남기고 삭제

    # step4: 문장 및 단어 단위로 토큰화
    #tokenedtext = re.findall(r"[\w]+-[\w]+|[\w']+", wo_nl)
    #result = ' '.join(tokenedtext)
    return only_text

def summary(doc_link):
    '''
    :param doc_link: 추천된 문서명을 받아옴
    :return: 해당 문서를 요약하여 제공
    '''
    doc_link = ["1911.09309n.txt"]
    #load_text
    dir = './tfidf'
    doc_list = []
    for filename in doc_link:
        doc = ''
        for line in open(dir + '/' + filename, 'r', encoding="utf8"):
            doc += line
        doc = preprocess(doc)
        sentence_list = nltk.sent_tokenize(doc)
        sentence_str = '\n'.join(sentence_list)
        doc_list.append(sentence_str)
    return doc_list
