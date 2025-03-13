# Best-seller-prediction-based-on-deep-learning-with-book-information
![image.png](attachment:de6982a9-32dd-4602-816a-592c0048203c:image.png)
- **2024.12.10 ~ 현재**
- 줄거리, 책 제목 등의 도서 정보를 활용하여 도서의 베스트 셀러를 예측하는 딥러닝 알고리즘 개발

> **핵심 전략**
> 
> 1. **데이터 불균형 해소**
>     - **문제 :** 베스트셀러 도서가 일반 도서에 비해 적어 학습 데이터에서 불균형이 발생
>     1. **Downsampling** 기법 적용: 베스트셀러 도서와 일반 도서 샘플 수를 3:1 비율로 균형 맞춤
>     2. 균형 잡힌 데이터셋으로 학습하여 베스트셀러 예측 성능 향상
> 2. **텍스트 임베딩 성능 최적화**
>     - **문제 :** Word2Vec은 문맥을 반영하지 못하고, KoBERT는 긴 문장에서 정보 손실이 발생
>     1. **Word2Vec**과 **KoBERT** 임베딩 성능 비교
>     2. KoBERT의 **Truncation Strategy** 최적화로 긴 문장 정보 손실 방지
>     3. 줄거리, 책 제목, 장르, 작가 등 다양한 텍스트 특징을 개별 임베딩 후 결합하여 학습

### **1. 개발 배경**

- ‘책을 출간하기 전 내가 쓴 도서가 베스트 셀러가 될지 안될지 확인 할 수 있으면 좋겠다’ 라는 생각을 통해 해당 프로젝트를 시작함
- 원고 작성하기 전 줄거리를 통해서 베스트 셀러 여부를 확인할 수 있기 때문에 빠르고 간편하게 아이디어를 수정할 수 있음
- 이 과정을 통해 베스트 셀러에 가까워질 수 있음

### **2. 기대 효과**

- 작가의 입장
    - 해당 알고리즘을 통해 자신의 도서가 베스트 셀러가 될지 안될지 출간 전에 미리 알 수 있음. 이 점을 활용하여 작가는 자신의 원고를 수정할 수 있음.
- 출판사의 입장
    - 원고 매입 전에 도서의 흥행 여부를 알 수 있음. 원고 매입 여부에 관한 결정에 도움을 줌. 출판사는 이를 활용하여 불필요한 투자를 줄이는 데 도움을 줄 수 있음

### **3. 아키텍쳐**

- 데이터 수집 및 전처리 → Kobert와 word2vec을 활용한 임베딩 생성 → 모델링(RNN, LSTM, GRU, Transformer) → 모델 학습 및 평가(accuracy, precision, recall, f1 score) → 하이퍼 파라미터 튜닝(GA)을 통한 최적화

### **4. 사용 기술**

- Python, Crawling, NLP, Embedding (KoBERT, Word2Vec), Deep Learning (RNN, LSTM, Transformer, GRU)

### **5. 라이브러리**

- Numpy, Pandas, KoBERT-Transformers, KoNLPy, Gensim, Pytorch, Sklearn, Matplotlib

### **6. 개발 환경**

- 언어 : Python 3.9.12
- OS : CentOS 7
- IDE : Jupyter Notebook, VS Code
- GPU : RTX 3070 (CUDA 12.4, PyTorch)
