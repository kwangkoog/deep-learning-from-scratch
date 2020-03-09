# coding: utf-8

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
