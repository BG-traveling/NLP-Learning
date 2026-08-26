# NLP Learning

자연어처리(NLP) 학습 기록 저장소. 청년취업사관학교 AI 교육과정에서 진행하는 NLP 수업 실습 코드와 학습 내용을 정리합니다.

## 학습 배경

CNN(ResNet, LeNet, AlexNet)을 통한 이미지 특징추출·분류·객체탐지·세그멘테이션 학습에 이어, 순차 데이터를 다루는 **RNN → LSTM/GRU → Transformer(Attention)** 계열로 확장하며 자연어처리를 학습 중입니다.

## NLP란?

사람이 쓰는 언어(텍스트/음성)를 컴퓨터가 이해·분석·생성할 수 있도록 처리하는 기술 분야. 이미지가 픽셀 행렬로 표현되듯, 텍스트는 숫자(벡터)로 변환되어야 모델이 학습할 수 있습니다.

## 전체 프로세스 흐름

1. **데이터 수집** — 공개 데이터셋, 크롤링 등
2. **전처리** — 정제(노이즈 제거) → 토큰화 → 불용어 제거 → 정규화(어간/표제어 추출)
3. **텍스트 표현(임베딩)** — Bag of Words, TF-IDF (통계 기반) → Word2Vec/GloVe (예측 기반) → BERT/GPT 등 문맥 기반 임베딩
4. **모델링** — RNN → LSTM/GRU → Encoder-Decoder → Transformer(Attention) → 사전학습 모델 파인튜닝
5. **평가** — Accuracy, F1-score(분류) / BLEU, ROUGE(생성·번역)
6. **배포** — FastAPI 등으로 API화하여 서비스에 연결

### Padding(패딩)

배치 학습 시 문장마다 길이가 다르므로, 정해진 최대 길이(max_len)에 맞춰 부족한 부분을 `<PAD>` 토큰(보통 0)으로 채우는 작업. 모델이 패딩값을 실제 단어로 착각하지 않도록 마스킹(masking)을 함께 적용하는 경우가 많습니다.

```
"나는 밥을 먹었다"          → [3, 15, 42, 0, 0]      (post-padding)
"나는 어제 친구랑 밥을 먹었다" → [3, 8, 21, 15, 42]
```

## 실습: SMS Spam Classification (RNN)

`main.py` + `models/rnn.py` — UCI SMS Spam Collection 데이터셋으로 스팸 여부를 분류하는 텍스트 분류 모델을 전처리부터 학습 루프까지 직접 구현.

```
원본 텍스트(.tsv)
  → tokenize()        소문자화 → 정규식으로 영문/숫자 외 제거 → 공백 분리
  → build_vocab()     빈도(min_freq) 기반 단어 사전, <PAD>=0, <UNK>=1
  → SpamDataset        문장 → 정수 시퀀스 변환 + max_len(50) 패딩/자르기
  → random_split        Train 70% / Valid 15% / Test 15%
  → DataLoader          batch_size=32
  → SpamRNN             Embedding → RNN → Dropout → Linear
  → train()             CrossEntropy + Optimizer, epoch별 loss/acc 기록
```

- **데이터**: `SMSSpamCollection` (ham/spam 라벨 + 원문 텍스트, 5,574건). 없을 경우 UCI 저장소에서 자동 다운로드 후 압축 해제
- **`SpamDataset` (커스텀 PyTorch Dataset)**: `__len__`, `__getitem__` 구현. 텍스트를 vocab 기준 정수 시퀀스로 바꾸고 `max_len`에 맞춰 패딩(부족분 0)/자르기(초과분 제거)
- **`SpamRNN` 모델**: `nn.Embedding(padding_idx=0)`으로 `<PAD>` 토큰은 학습에서 제외, `nn.RNN`의 마지막 타임스텝 hidden state를 문장 전체 맥락 벡터로 사용해 `nn.Linear`로 분류
- **`train()`**: epoch별 train loss/accuracy, validation accuracy 기록 및 best validation accuracy 추적
- **다음 단계**: 실제 학습 실행 및 결과 기록 → RNN을 LSTM/GRU로 교체해 성능 비교 → F1-score 등 평가지표 보강

## 폴더 구성

```
├── main.py                # 전처리 + Dataset/DataLoader + 학습 루프
├── models/
│   └── rnn.py              # SpamRNN (Embedding + RNN + FC)
├── SMSSpamCollection        # 실습 데이터셋
├── requirements.txt         # 의존성 (PyTorch, pandas, scikit-learn 등)
└── README.md
```

## 개발 환경

`requirements.txt` 참고 — PyTorch(CUDA), pandas, scikit-learn, numpy 등 데이터 처리·딥러닝 스택 사용.

## 관련 정리

학습 개념 정리는 Notion(학습내용 → 개발사전 → NLP)에 병행 기록 중입니다.
