# 존중톡

## ***Team234*** 

![Team234](https://user-images.githubusercontent.com/57997672/84602970-e7dc8280-aec5-11ea-8aa3-9da5f8288388.png)

- 김태호 : Web Scrapping, Data Labeling, PyTorch-RNN Modeling
- 박혜진 : Data Labeling, Django 기반 웹페이지 제작
- 위정훈 : 기획 및 업무분배, Web Scrapping, Data Labeling, Tokenizing, Keras-CNN, KoBERT Modeling

---

## 기획의도

### *'별거 아닌 채팅 말투 때문에...' '한번만 더 생각하고 채팅할걸...'*

- 딱히 예민한 사람이 아니더라도, 업무를 하다 보면 업무 채팅 때문에 사소한 오해가 생기고 일을 때려치고 싶을 때가 있습니다
- 저 사람이 말을 왜 ~~*저 따위*~~.. 저렇게 하나 싶은데, 막상 만나서 얘기하면 별일 없을 때도 종종 있죠
- 홧김에 타자를 치고나서 심호흡 한번만 더하고 문자 보낼 걸 후회하기도 하죠..

### 당신의 젠틀함을 지켜주기 위해

![](https://user-images.githubusercontent.com/57997672/84603924-507b2d80-aecd-11ea-867e-d2b53b25c015.png)

사소한 감정 소모로 체면을 구기고 업무에 지장을 주는 상황을 모면할 수 있도록, '존중톡'을 기획하게 되었습니다. 

분노의 타자를 치고나서도 한번더 생각할 수 있도록, 내가 무심코 한 말이 상대에게 오해가 되지 않도록, 존중톡이 도와드립니다.




---

## Portfolio

- WebPage : https://jonjung.herokuapp.com
- Github : https://github.com/gitgitWi/Jon_Jung_Talk_Django
- DeepNote : https://beta.deepnote.com/project/35dabe6d-f7f9-4f66-ae2f-9b6e9c23b56e#%2F01_notebooks%2F0522_WebScrapping.ipynb


## 참고자료

- <김기현의 자연어 처리 딥러닝 캠프; 파이토치편(2019)>
- <한국어 임베딩(2019)>
- <텐서플로와 머신러닝으로 시작하는 자연어 처리(2019)>
- <파이썬을 이용한 머신러닝, 딥러닝 실전 개발 입문(2017)>
- <파이썬 머신러닝 완벽 가이드(2020)>
- <PyTorch로 시작하는 딥 러닝 입문(wikidocs)>
- https://github.com/kimwoonggon/publicservantAI
- https://www.youtube.com/watch?v=Vj55zaDvn4Q 등 다수 유튜브 세미나/강의


---

## 개발환경

### Python

![COLAB](https://blog.nerdfactory.ai/assets/images/posts/learn-bert-with-colab/colab_logo.png)

***Google Colab***
  - OS : Ubuntu 18.04.3 LTS
  - Python : 3.6
  - Java : OpenJDK 11
  - GPU : Tesla K80, P100
  - CPU : Xeon(R) CPU @ 2.00 ~ 2.30GHz

![DeepNote](https://pycon-assets.s3.amazonaws.com/2020/media/startuprow_logos/deepnote_logo.png)

***DeepNote***
  - OS : Debian 9.12
  - Python : 3.7
  - CPU : Xeon(R) CPU @ 2.30GHz

![Azure](http://www.epnc.co.kr/news/photo/201909/92123_82837_1944.jpg)

***MS Azure***

  - Python : 3.6
  - GPU : Tesla K80


#### Packages

***Machine Learning Framework***

- Tensorflow-Keras
- PyTorch

***Data Science***

- Numpy 
- Pandas 
- Seaborn
- Matplotlib

***NLP***

- KoNLPy : Okt, Mecab
- SoyNLP
- Sentencepiece
- Khaiii
- transformers : KoBERT

***Web Scrapping***

- BeutifulSoup4
- Selenium

***Web Framework***

- Django
- Gunicorn
- psycopg2 for PostgreSQL

### Web Deploy

- ***Heroku***

---

## 개발 과정

### 01. Team234 팀 결성, 기획의도 공유 (5.21.) 

- 심리학자 폴 에크만의 6가지 보편적 감정 모델 기반 감정 분류
- CNN, RNN 기반 텍스트 Multi-class 분류
- 데이터 수집 대상 결정 : Google PlayStore (Korea)

### 02. Web Scrapping (5.22. ~ 5.25.) 

- BeutifulSoup4, Selenium 활용 
- https://beta.deepnote.com/project/35dabe6d-f7f9-4f66-ae2f-9b6e9c23b56e#%2F01_notebooks%2F0522_WebScrapping.ipynb
- 최고 매출 기준 상위 50개 앱에 대한 리뷰 69493개(중복 3501개 제외) 수집

### 03. Word Tokenizing & Embedding, Data Labeling (5.26. ~ 5.29.)

#### Tokenizing & Embedding

![Screen Shot 2020-06-15 at 06 25 21](https://user-images.githubusercontent.com/57997672/84604451-1e6bca80-aed1-11ea-81a8-0c756d669e61.png)

- KoNLPy Okt, SoyNLP 활용
- 단어 수준 임베딩을 위한 토크나이징

#### Sample Data Labeling

![Screen Shot 2020-06-15 at 06 23 13](https://user-images.githubusercontent.com/57997672/84604402-c6cd5f00-aed0-11ea-9201-dbc8eb849f82.png)

![Screen Shot 2020-06-15 at 06 25 30](https://user-images.githubusercontent.com/57997672/84604474-393e3f00-aed1-11ea-9730-d200d91c5598.png)

- 카카오톡 리뷰 2015개 샘플로 추출
- label 데이터 review_response, reader_response
- 각각 3개 카테고리로 분류 : 중립, 긍정, 부정
- GCP AutoML Table에서 전처리 없이 학습 결과 정확도 70% 내외

![](https://user-images.githubusercontent.com/57997672/84604357-50c8f800-aed0-11ea-83d0-300559cb40bc.png)

### 04-1. Modeling, Word Tokenizing 추가 학습 (6.1.~6.12.)

***Word Tokenizers***

![Screen Shot 2020-06-15 at 07 06 47](https://user-images.githubusercontent.com/57997672/84605237-d64fa680-aed6-11ea-8510-316fa591a761.png)

- 지도학습 방식 : Mecab (KoNLPy)
- 비지도학습 방식 : SentencePiece (Google), Khaiii (Kakao Brain)



#### Modeling

***CNN, KoBERT***


***RNN***

https://beta.deepnote.com/project/35dabe6d-f7f9-4f66-ae2f-9b6e9c23b56e

LSTM Model

```python
SentimentRNN(
  (embedding): Embedding(70193, 380)
  (lstm): LSTM(380, 2048, num_layers=2, batch_first=True, dropout=0.5)
  (dropout): Dropout(p=0.3, inplace=False)
  (fc): Linear(in_features=2048, out_features=1, bias=True)
  (sig): Sigmoid()
)
```

- Okt corpus
- Sentencepiece corpus



### 04-2. 웹 페이지 구현 (6.1.~6.14.)


---

## 아쉬운 점

- Labeling Data가 부족해, 만족할만한 정확도를 만들어 낼 수 없었습니다.
  - 정확도 문제 & 시간 부족으로, 초기 계획했던 것처럼, 2천개 샘플로 학습 - 나머지 데이터 중 일부 예측 - 틀린 부분만 수정해 모델 업데이트 - 정확도 높인 모델로 또 남은 데이터 예측 반복하는 작업은 아예 시도하지도 못했습니다.

- 채팅 보조 기능 추가 불가
  - Heroku 호스팅의 무료 용량 한도는 500MB
  - 이 부분을 늦게 알게 되어 KoBERT 모델(320MB)은 물론, Tensorflow(520MB) 설치도 불가
  - 용량 문제만 해결되면, 간단한 채팅 앱을 구현해 메시지를 띄워주는 기능은 가능할 것으로 보입니다.

## 결론

- 한국어 NLP를 너무 쉽게 생각하고 도전했는데, 과정 하나하나가 쉽지 않고 고려해야할 것들이 너무 많았습니다.
- 딥러닝, NLP 관련 다양한 라이브러리를 경험할 수 있었습니다.