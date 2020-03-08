# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {} # dict 자료형 정의 (O(1)연산 수행으로 list로 만들지 않음)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # randn은 가우시안 표준 정규분포에서 난수 생성
        self.params['b1'] = np.zeros(hidden_size) # bias b1 초기값은 0으로 가정함
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size) # bias b2 초기값은 0으로 가정함

    # predict 함수를 거치면 2계층으로 구성된 신경망에 대해 입력 x를 받고 현재 설정된 weight를 기반으로 y값을 도출함    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        # L1 계층 hypothesis 수행
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        
        # L2 계층 hypothesis 수행 (마지막은 softmax 함수를 사용)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        # predict 함수를 통해 현재 weight, bias 값을 기준으로 y값을 예측함
        y = self.predict(x)
        # 예측된 결과에 대해 cross entropy error 손실함수 값을 리턴함 (실측치와 비교 시, 엔트로피 손실함수 차원에서 에러 판정)
        # 크로스 엔트로피 경우 원-핫 인코딩에 의해 정답에 해당되는 예측값의 확률이 높을 때 에러가 줄고, 확률이 낮으면 에러가 높음 (지수로그함수)
        return cross_entropy_error(y, t)
    
    # 정확도를 계산함
    def accuracy(self, x, t):
        # 현재 weight, bias 텐서값을 기준으로 신경망의 y 값을 도출
        y = self.predict(x)
        # y값 중에서 가장 큰 y 인덱스를 취하고 t에 대해서도 가장 큰 t 인덱스를 리턴 (t 값은 one-hot 인코딩으로 되어 있음)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        # y와 t의 인덱스가 같다면 (x 입력에 대해 신경망의 결과가 정답과 동일하다면), 입력값 X 개수대비 정확도를 구함
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    # 수치미분 방식으로 매개변수의 기울기를 계산함
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        # 학습 수행 시, 편미분 결과에 대해 저장하는 dict 자료형 선언
        grads = {}
        # 각 변수별로 기울기를 구하고 저장함 (key:value pair)
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
