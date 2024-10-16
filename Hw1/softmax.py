import numpy as np
# 設定總共類別
c = 10                    
# 模擬輸出 logits
x = np.random.rand(c) 
# 計算 softmax
exp_x = np.exp(x)
softmax_x = exp_x / np.sum(exp_x)
print(softmax_x)
print(np.sum(softmax_x))
