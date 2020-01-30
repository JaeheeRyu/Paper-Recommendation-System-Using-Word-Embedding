import TFIDF
import Userinter
from gensim.models import Word2Vec, KeyedVectors, FastText
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import numpy as np
from beautifultable import BeautifulTable

def max_extract(x, num):
    li = []
    for i in range(num):
        max_ = max(x)
        index_ = x.index(max_)
        x[index_] = -100
        li.append(index_)
    return li


def show_2d_pca():
    # 단어를 2차원 공간상에 시각화
    ft_model = FastText.load('./fasttext/eng_ft')  # ft 모델 로드
    w2v_model = KeyedVectors.load_word2vec_format("eng_w2v")  # w2v 모델 로드
    words = ['lstm', 'boy', 'girl', 'bert', 'xlnet', 'pytorch', 'tensorflow', 'waterfall', 'agile', 'tcp', 'udp', 'keras']

    # fast text 일 때
    pca = PCA(n_components=2)
    xys = pca.fit_transform([ft_model.wv.word_vec(w) for w in words])
    xs = xys[:, 0]
    ys = xys[:, 1]
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, marker='o')
    for i, v in enumerate(words):
        plt.annotate(v, xy=(xs[i], ys[i]))

    # word2vec 일 때
    xys = pca.fit_transform([w2v_model.wv.word_vec(w) for w in words])
    xs = xys[:, 0]
    ys = xys[:, 1]
    plt.figure(figsize=(8, 5))
    plt.scatter(xs, ys, marker='o')
    for i, v in enumerate(words):
        plt.annotate(v, xy=(xs[i], ys[i]))


def extract_simdoc_list(WE_loaded_model, top7_list, paper_list, user_input, num):
    '''
    :param WE_loaded_model:  Loaded Word embedding Model
    :param top7_list:  TF-IDF 상위 7개 단어를 담고 있는 리스트
    :param paper_list:  문서의 이름을 담은 리스트
    :param user_input:  유저로부터 입력받은 키워드
    :param num:  추천 문서 수
    :return:  추천 문서 명
    '''
    final_mean_sim = []
    for top7_entry in top7_list:
        mean_sim = []
        for ui in user_input:
            k = 0.0
            sim = 0.0
            for keys in top7_entry:
                k += top7_entry[keys]
                sim += top7_entry[keys] * WE_loaded_model.wv.similarity(ui, keys)
            mean_sim.append(sim / k)
        final_mean_sim.append(max(mean_sim))

    sim_doc_top5 = max_extract(final_mean_sim, num)  # 유저키워드-논문토픽 유사도 상위 5개 논문의 인덱스
    doc_link = np.array(paper_list)[sim_doc_top5].tolist()  # 유사도 상위 5개 논문의 링크
    return doc_link

def doc_show(user_pick):
    '''
    :param user_pick: user가 선택한 논문 내용을 받음
    :return: 논문을 간략한 테이블로 출력
    '''
    table = BeautifulTable()
    table.append_row(["Title", user_pick[0]])
    table.append_row(["Authors", user_pick[1]])
    table.append_row(["Abstract", user_pick[2]])
    table.append_row(["Link", user_pick[3]])
    print(table)
    print('\n')

def main():
    # TF-IDF로 문서의 주제가 될만한 단어 선택
    text_list, paper_list = TFIDF.load_text_list()
    top7_list = TFIDF.create_top7(text_list)
    ft_model = FastText.load('./fasttext/eng_ft')  # ft 모델 로드
    user_input = str(input('▶키워드를 입력하세요:'))
    user_num = int(input('▶추천받을 논문 수를 입력하세요:'))
    user_input = user_input.split() # 유저 인풋
    user_input = [w.lower() for w in user_input]
    doc_link = extract_simdoc_list(ft_model, top7_list, paper_list, user_input, user_num)
    user_inter = Userinter.craw(doc_link)
    while(True):
        print('\n[추천 논문 리스트]')
        for i in user_inter:
            print(str(user_inter.index(i)+1), i[0])
        user_pick = int(input('\n▶원하는 논문번호를 입력하세요:'))
        user_pick = user_inter[user_pick-1]
        doc_show(user_pick)
        choice = int(input('▶다른 추천 논문을 보시려면 0, 키워드를 다시 입력하시려면 1 입력:'))
        if choice == 0:
            continue
        else:
            user_input = str(input('▶키워드를 입력하세요:'))
            user_num = int(input('▶추천받을 논문 수를 입력하세요:'))
            user_input = user_input.split()  # 유저 인풋
            user_input = [w.lower() for w in user_input]
            doc_link = extract_simdoc_list(ft_model, top7_list, paper_list, user_input, user_num)
            user_inter = Userinter.craw(doc_link)
#논문 링크, 초록, 저자, 타이틀(요약X)

if __name__ == '__main__':
    main()