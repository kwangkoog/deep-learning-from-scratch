# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        
        # 가중치 초기화 (순전파 당시와 동일하게 dictionary 변수에 W1, b1, W2, b2를 초기화)
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성 
        # 순전파에서는 predict 함수에서 행렬곱 연산을 수행하여 현재 저장된 weight, bias 값 기반으로 y값을 도출함
        # 역전파에서는 초기 클래스를 생성할 때, 계층을 생성해 준다
        # 계층을 orderedDict 기반으로 보관한다
        # 입력층 => affine 계층 => relu => affine 계층 => 출력층 => 손실함수
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        
        # 입력되는 x에 대해 layer를 꺼내면서 최종 x를 계산함 (출력층은 손실함수라고 본다)
        for layer in self.layers.values():
            x = layer.forward(x)
        
        return x
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        
        # 현재 입력된 x 데이터 셋을 기준으로 y 출력
        y = self.predict(x)
        # 출력된 y 배열값들과 정답지 t 배열값을 비교하며, 이때 softmax-with-loss의 forward를 사용함
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        
        # 현재 입력된 x 데이터 셋을 기준으로 y 출력
        y = self.predict(x)
        # 현재 출력 배열에서 가장 큰 값들만 취함
        y = np.argmax(y, axis=1)
        # t는 one-hot 인코딩 (아래 코드에 대해 좀더 분석 필요)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        # 전체 x 개수중에서 맞춘 개수를 정확도로 판정
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        
        # 현재 입력 관점에서 손실함수를 계산함 (순전파 수행)
        loss_W = lambda W: self.loss(x, t)
        
        # 경사 하강법을 이용하여 weight, bias를 갱신함
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):
        
        # forward를 통해서 현재 손실함수 값을 lastLayer에 저장 (손실값, y값, t값이 저장됨)
        self.loss(x, t)

        # backward
        # 마지막 계층에 대해 backward 수행하여 dout을 계산함
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        # 계층별로 현재 저장되어 있는 값들을 가져오고 뒤집는다
        layers = list(self.layers.values())
        layers.reverse()
        # 각 계층별로 돌면서 역전파를 수행하여 dout을 앞으로 건네줌
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        # 경사 하강법에 의해서 수치미분를 사용하지 않고
        # 현재 계층별로 저장되어 있는 국소 미분값들을 그대로 가져옴 (역전파 하면서 저장된 값들만 가져온다)
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
