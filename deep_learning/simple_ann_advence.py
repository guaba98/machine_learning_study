"""
문제 1. 두 종류의 물고기를 구분하는 딥러닝을 설계하라.
* 사용 데이터는 fish/fishes_data.csv(type column은 종을 나타내며 1은 도미, 0은 빙어이다.)
1. keras의 random을 0번 seed로 고정한다.
2. Sequential을 이용하여 모델 작성한다.
3. 모델을 최소 2개층까지 사용하여 만든다.
4. 이진분류를 실시한다.
5. 학습 결과가 보이도록 출력한다.
* 데이터 전처리는 마음대로, seed는 0번
* 최종적으로 [[25., 150.], [20., 100.]] 이 두개의 도미 데이터를 맞추도록 하이퍼 파라미터를 조정한다.
"""
# 모듈 import
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

# 자료 불러오기
data = pd.read_csv('../data/fish/fishes_data.csv')
data.head() # 1은 도미, 0은 빙어

# 랜덤시드 고정
seed = 0
keras.utils.set_random_seed(seed)

# 학습 데이터와 타겟 데이터 분류하기
train_data = data.drop(columns = 'type')
train_target = data['type']

# 모델 정의
model = keras.models.Sequential(
    [
        keras.layers.Dense(units = 20, input_dim=2, activation = "relu"),
        keras.layers.Dense(units = 1, activation = "sigmoid")
    ])

# 모델에 학습할 손실 함수와 최적화 함수, 옵션 지정
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

# fit을 통해 학습 시작
model.fit(train_data, train_target, epochs=30, verbose=1)

# 모델로 예측하기
predictions = model.predict([[25., 150.], [20., 100.]])

# 확률 값을 이진 결과로 변환하기 (0.5를 기준으로)
binary_predictions = np.where(predictions > 0.5, 1, 0)

print(binary_predictions)