# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기 (훈련 데이터는 6만개)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# mnist 데이터셋에 대해 784 픽셀 입력에 대해 hidden layer 노드 50개를 거쳐, output을 10개 도출하는 신경망 구성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기, 60000개의 데이터에서 임의로 100개의 데이터를 추려내기 위함
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
# 60000개 데이터에서 100개의 미니배치를 나누면 600개가 한 에폭 당 반복 수가 됨
iter_per_epoch = max(train_size / batch_size, 1)

# 훈련 횟수인 10000번 동안 경사 하강법을 수행함
# 기울기 계산이 들어갈 때 batch 갯수를 가져가서 손실함수를 구하고,
# 그 다음 경사 하강법에 의해 현재 기울기를 구하고,
# 다음 스텝에 필요한 weight, bias를 갱신한다

for i in range(iters_num):

    # 매 반복마다 미니배치 획득
    # 6만개 데이터중 100개에 대한 인덱스를 배열로 저장 (랜덤하게 가져오기 때문에 1만번 수행에 대해 랜덤하게 학습 대상을 가져옴)
    batch_mask = np.random.choice(train_size, batch_size)
    # 배치에 해당하는 훈련 데이터 x와 정답지 t값을 가져옴
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch)
    
    # 매개변수 갱신
    # 각 key 값 별로 weight, bias를 갱신함
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    # 학습 경과 기록
    # 10,000번 수행되면서 error 값이 얼마나 줄어드는지 기록함
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # 1에폭당 정확도 계산
    # 신경망 학습의 목표는 범용적인 능력이므로 학습 도중 정기적으로 훈련 데이터와 시험 데이터를 대상으로 정확도를 기록함
    # 1에폭은 학습에서 훈련 데이터를 모두 소진했을 때의 횟수에 해당
    # 현재 예에서는 600개가 하나의 에폭이므로 10000번 수행을 600으로 나누면 17번이 나옴 (17번마다 기록을 수행)
    if i % iter_per_epoch == 0:
        # 훈련데이터에 대한 정확도 산출
        train_acc = network.accuracy(x_train, t_train)
        # 테스트 데이터에 대한 정확도 산출
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        
# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
