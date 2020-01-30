from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
from nltk.stem import PorterStemmer
import numpy as np


def max_extract(x, num):
    li = []
    x = x.tolist()
    for i in range(num):
        max_ = max(x)
        index_ = x.index(max_)
        x[index_] = -100
        li.append(index_)
    return li


def preprocess(doc):
    # Step2: Convert to Lowercase
    lowertext = doc.lower()

    # Step3: Normalize (remove numbers, some punctuation)
    wo_nl = re.sub(r"-\n+", "", lowertext)  # 하이픈 뒤에 같이 따라오는 개행문자 삭제
    wo_nl = wo_nl.replace("\n", " ")  # 개행문자 띄어쓰기로 대체
    only_text = ''.join(i for i in wo_nl if i.isalpha() or i in ['.', ' ', '-', '?', '!'])  # 알파벳과 해당 기호들만 남기고 삭제

    # step4: 문장 및 단어 단위로 토큰화
    tokenedtext = re.findall(r"[\w]+-[\w]+|[\w']+", only_text)

    # Step5: Remove stopwords
    # http://www.lemurproject.org/stopwords/stoplist.dft
    stopwords2 = "a about above according across after afterwards again against albeit all almost alone along already also although always am among amongst an and another any anybody anyhow anyone anything anyway anywhere apart are around as at av be became because become becomes becoming been before beforehand behind being below beside besides between beyond both but by can cannot canst certain cf choose contrariwise cos could cu day do does doesn't doing dost doth double down dual during each either else elsewhere enough et etc even ever every everybody everyone everything everywhere except excepted excepting exception exclude excluding exclusive far farther farthest few ff first for formerly forth forward from front further furthermore furthest get go had halves hardly has hast hath have he hence henceforth her here hereabouts hereafter hereby herein hereto hereupon hers herself him himself hindmost his hither hitherto how however howsoever i ie if in inasmuch inc include included including indeed indoors inside insomuch instead into inward inwards is it its itself just kind kg km last latter latterly less lest let like little ltd many may maybe me meantime meanwhile might moreover most mostly more mr mrs ms much must my myself namely need neither never nevertheless next no nobody none nonetheless noone nope nor not nothing notwithstanding now nowadays nowhere of off often ok on once one only onto or other others otherwise ought our ours ourselves out outside over own per perhaps plenty provide quite rather really round said sake same sang save saw see seeing seem seemed seeming seems seen seldom selves sent several shalt she should shown sideways since slept slew slung slunk smote so some somebody somehow someone something sometime sometimes somewhat somewhere spake spat spoke spoken sprang sprung stave staves still such supposing than that the thee their them themselves then thence thenceforth there thereabout thereabouts thereafter thereby therefore therein thereof thereon thereto thereupon these they this those thou though thrice through throughout thru thus thy thyself till to together too toward towards ugh unable under underneath unless unlike until up upon upward upwards us use used using very via vs want was we week well were what whatever whatsoever when whence whenever whensoever where whereabouts whereafter whereas whereat whereby wherefore wherefrom wherein whereinto whereof whereon wheresoever whereto whereunto whereupon wherever wherewith whether whew which whichever whichsoever while whilst whither who whoa whoever whole whom whomever whomsoever whose whosoever why will wilt with within without worse worst would wow ye yet year yippee you your yours yourself yourselves".split()
    stopwords3 = "one two three four five six seven eight nine ten eleven twelve thirtheen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four twenty-five twenty-six twenty-seven twenty-eight twenty-nine thirty thirty-one forty fifty sixty seventy eighty ninety hundred thousandth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirtheenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth twentieth twenty-first twenty-second twenty-third twenty-fourth twenty-fifth twenty-sixth twenty-seventh twenty-eighth twenty-ninth thirtieth thirty-first fortieth fiftieth sixtieth seventieth eightieth ninetieth hundredth thousand".split()
    stopwords4 = ['cid', 'al', 'figur', 'pp', 'fig', 'arxiv', 'ours']

    stopWords = []
    for line in open('./stopwords_nltk.txt'):
        stopWords.append(line.split()[0])

    stopWords = set(stopWords + stopwords2 + stopwords3 + stopwords4)
    tokenedtext2 = [x for x in tokenedtext if x not in stopWords]

    # Step6: Stemming(어간 추출)
    ps = PorterStemmer()
    stemmedtext = [ps.stem(x) for x in tokenedtext2]

    # Step7: 너무 짧은 단어와 너무 긴 단어 삭제
    stemmedtext = [x for x in stemmedtext if len(x) > 1]
    # 15글자 이상인 문자열 제거하되 하이픈 포함 문자는 남겨둠
    longword = [x for x in stemmedtext if len(x) > 15 and '-' not in x]
    stemmedtext2 = [x for x in stemmedtext if x not in longword]

    result = ' '.join(stemmedtext2)
    # print(result)
    return result


def load_text_list():
    # Step1: 모든 데이터 불러오기
    dir = './tfidf'
    i = 0
    j = 0
    text_list = []
    paper_list = []
    all_paper_list = os.listdir(dir)
    for filename in all_paper_list:
        i += 1
        # print(i)

        doc = ''
        try:
            for line in open(dir + '/' + filename, 'r', encoding="utf8"):
                doc += line
        except:
            j += 1
            continue
        if doc == '':
            continue
        paper_list.append(filename)
        text_list.append(preprocess(doc))
    # print(j)
    return text_list, paper_list #text_list : 전체 문서 집합(리스트, 각 요소는 str) / paper_list : 문서 파일명


def create_top7(text_list):
    # TF-IDF matrix 생성
    tfidv = TfidfVectorizer(min_df=2).fit(text_list)  # 2개 미만의 문서에서 등장하는 단어는 무시하고 TF-IDF 생성
    tf_matrix = tfidv.transform(text_list).toarray()  # numpy array로 변환
    features = np.array(tfidv.get_feature_names())  # term-doc 매트릭스를 구성하는 단어

    # TF-IDF 값 상위 7개 단어에 대한 딕셔너리 생성
    top7_list = []
    for example in tf_matrix:
        tf_dict = {}
        top7 = max_extract(example, 7)  # TF-IDF 값 상위 7개 단어의 인덱스
        top_word = features[top7]  # TF-IDF 값 상위 7개 단어
        top_tfidf = example[top7]  # 상위 7개 단어의 TF-IDF 값
        for w, tf in zip(top_word, top_tfidf):
            tf_dict[w] = tf
        top7_list.append(tf_dict)
    return top7_list


def show_proportion(text_list):
    # 상위 7개의 TF-IDF 값이 전체에서 차지하는 비율
    tfidv = TfidfVectorizer(min_df=2).fit(text_list)  # 2개 미만의 문서에서 등장하는 단어는 무시하고 TF-IDF 생성
    tf_matrix = tfidv.transform(text_list).toarray()  # numpy array로 변환
    prop_list = []
    for example in tf_matrix:
        top7 = max_extract(example, 7)
        prop_list.append(example[top7].sum() / example.sum())
    print(np.array(prop_list).mean())
