import numpy as np
# 設定輸入維度
d_in = 10                                
# 設定輸出維度
d_out = 30                               

# 模擬神經網路輸入
x = np.ones((d_in, 1))                   
# 模擬神經網路權重
W = np.random.rand(d_out, d_in) * 10 - 5 
# 模擬神經網路偏差值
b = np.random.rand(d_out, 1) * 10 - 5  
# TODO
# 計算線性輸出
z = np.dot(W, x) + b
# 計算 ReLU Activation
a = np.maximum(z, 0)
print(a)
# 計算總和
print(np.sum(a))