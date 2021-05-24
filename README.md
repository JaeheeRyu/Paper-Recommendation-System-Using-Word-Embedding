# Paper Recommendation System Using Word Embedding

## ✨주요 내용

최근 1년간 arXiv(https://arxiv.org/) 에 공유된 Computer Science 분야의 논문을 수집하여 논문 데이터 corpus를 구축하고, Word-Embedding을 기반으로 한 추천 알고리즘을 제안, 사용자에게 적합한 논문 추천 제공

## 🎞분석 과정

![image-20210525012016547](https://github.com/JaeheeRyu/Paper-Recommendation-System-Using-Word-Embedding/blob/master/img/image-20210525012016547.png)

## 🎨시스템 개요

![image-20210525012150375](https://github.com/JaeheeRyu/Paper-Recommendation-System-Using-Word-Embedding/blob/master/img/image-20210525012150375.png)

- 키워드가 단순히 ‘등장’하는 문서보다, 문서와 키워드간 관련도를 파악, 비교해서 더 관련이 높은 문서를 추천
- 키워드 완전 일치보다 키워드간 거리벡터를 고려, ‘이음 동의어’ 혹은 ‘뜻이 비슷한 단어’의 문서 내 중요도까지 고려하여 추천

## 🎯추천 알고리즘 도출

![image-20210525012320047](https://github.com/JaeheeRyu/Paper-Recommendation-System-Using-Word-Embedding/blob/master/img/image-20210525012320047.png)

1. simil(t,u) : User의 키워드가 문서의 토픽 단어들과 얼마나 유사한지 Similarity 계산

2. 위 유사도에 문서별 토픽 단어 TF-IDF 값을 가중치로 두어 가중 평균을 계산

3. 위 과정을 모든 유저 키워드에 대해 진행, 추천 점수가 가장 높은 키워드의 점수를 문서 i의 추천 점수로 결정

   💫 **각 문서의 추천 점수를 비교해 추천 점수가 가장 높은 상위 N개의 논문을 추천함**

#### 💾Examples

![image-20210525012735736](https://github.com/JaeheeRyu/Paper-Recommendation-System-Using-Word-Embedding/blob/master/img/image-20210525012735736.png)

## 💡논문 추천 데모

![image-20210525012834073](https://github.com/JaeheeRyu/Paper-Recommendation-System-Using-Word-Embedding/blob/master/img/image-20210525012834073.png)![image-20210525012857304](https://github.com/JaeheeRyu/Paper-Recommendation-System-Using-Word-Embedding/blob/master/img/image-20210525012857304.png)

### 📖[분석 보고서](https://drive.google.com/file/d/1n_xb2lUp4jR5UUw6jDkdh50yStYAtRgO/view?usp=sharing)


