# coding: utf-8

# 각 클래스에 대해 순전파 대비 역전파 관점에서 바라보면,
# 더 이상 weight에 대해 매개변수를 갱신하는 형태로 구성되어 있지 않음을 알 수 있음
# 각 계층들은 backward와 forward 함수를 모두 갖고 있음
# 순전파의 forward도 반드시 구현되어야 함 => 결국 입력 x에 대해 y를 계산하는 것임
# forward는 현재 결정된 weight 관점에서 계산을 수행하게 하고
# backward는 gradient 자체를 갱신하는 관점에서만 사용할 수 있음

# 입력 x와 y에 대한 forward / backward 연산을 수행하는 곱하기 노드 설계
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 입력 x와 y에 대한 곱    
    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    # 입력으로 dout이 들어오면 
    # dx는 dout에 y를 곱
    # dy는 dout에 x를 곱
    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy

# 입력 x와 y에 대한 덧셈
class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
