#RNN, LSTM, GRU
#ResNet -> Residual(잔차) 구조가 특징
#RNN -> Recurrent 구조
#https://arxiv.org/pdf/1912.05911 -> 논문

#RNN, LSTM, GRU 만들때도 -> 토큰화 + vocab 

#os -> 파일, 경로
#re -> 정규표현식(regex)
#urllib.request -> 인터넷 자료 다운로드 라이브러리
import os, re, urllib.request, zipfile
import pandas as pd
from collections import Counter #워드클라우드 만들 때 '단어 수' 함수


def load_data(data_path='SMSSpamCollection', batch_size=32):
    #os.path.exists(경로) : '경로'가 존재하는가?
    if not os.path.exists(data_path):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        #정해놓은 url에 가서 zip 다운로드
        urllib.request.urlretrieve(url, 'smsspam.zip')
        with zipfile.ZipFile('smsspam.zip') as z:
            z.extractall('.')
        print('완료')

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.split()
    return text

#문자열 -> 정수, 어떤 문자열이 몇 번 정수로 바뀌었나? 기억
def build_vocab(df, min_freq=2):
    #나는 딥러닝을 공부하고 있어. 딥러닝은 정말 많은 작업을 할 수 있어.
    # 나 0 는 1 딥러닝 2 을 3 공부하고 4 있어 5 딥러닝 2 은 5 정말 6 많은 7 작업 8 을 3
    counter = Counter(tok for text in df for tok in tokenize(text))

    #패딩
    #알려지지 않음
    vocab = {'<PAD>': 0, '<UNK>': 1}
    #counter.items (딥러닝, 2)
    for word, freq in counter.items():
        if freq >= min_freq: 
            vocab[word] = len(vocab)
    return vocab

def preprocessing(data_path='SMSSpamCollection'):

    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'text'])
    # print(df.head())

    df['label'] = (df['label'] == 'spam').astype(int) #label이라는 열을 숫자로 변환
    print(f'스팸이 아닌 것: {(df.label == 0).sum()}, 스팸인 것: {(df.label == 1).sum()}')

    #vocab으로 변환!
    vocab = build_vocab(df['text'])
    print(vocab)

if __name__ == "__main__":
    preprocessing()