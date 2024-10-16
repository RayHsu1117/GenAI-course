# 練習 Hint
import pandas as pd
# 匯入填補缺失值的工具
from sklearn.impute import SimpleImputer          
# 匯入 Label Encoder
from sklearn.preprocessing import LabelEncoder    
# 匯入決策樹模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier   
# 匯入準確度計算工具
from sklearn.metrics import accuracy_score     
# 匯入 train_test_split 工具
from sklearn.model_selection import train_test_split ,ross_val_score  


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv('./data/train.csv')

# 取出訓練資料需要分析的資料欄位
df_x = df.loc[:,['Sex', 'Age', 'Fare','Pclass']]        
# 取出訓練資料的答案
df_y = df.loc[:,'Survived']

# 數值型態資料前處理
# 創造 imputer 並設定填補策略
imputer = SimpleImputer(strategy='median')        
age = df_x.loc[:,'Age'].to_numpy().reshape(-1, 1)
# 根據資料學習需要填補的值
imputer.fit(age)                                  
# 填補缺失值
df_x.loc[:,'Age'] = imputer.transform(age)           

# 類別型態資料前處理
# 創造 Label Encoder
le = LabelEncoder()                               
# 給予每個類別一個數值
le.fit(df_x.loc[:,'Sex']) 
                           
# 轉換所有類別成為數值
df_x.loc[:,'Sex'] = le.transform(df_x.loc[:,'Sex'])

# 分割 train and test sets，random_state 固定為 1012
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, train_size=0.8, random_state=1012)

# 初始化模型
models = {
    "Decision Tree(origin)": DecisionTreeClassifier(random_state=1012,splitter='random'), # "random_state" 是為了讓每次訓練結果一樣
    "Decision Tree(gini)": DecisionTreeClassifier(random_state=1012, criterion='gini',max_depth=10,splitter='random'),
    "Decision Tree(entropy)": DecisionTreeClassifier(random_state=1012,criterion='entropy',max_depth=10,splitter='random'),
    "Random Forest": RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 1012, max_depth=9,bootstrap=True,max_features='sqrt'),
    "SVC Linear": SVC(kernel='linear', random_state=1012,C=1),
}

# 訓練和評估模型
for name, model in models.items():
    model.fit(train_x, train_y)  
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    train_acc = accuracy_score(train_y, train_pred)             
    test_acc = accuracy_score(test_y, test_pred) 
    precision = precision_score(test_y, test_pred)
    # recall = recall_score(test_y, test_pred)
    # f1 = f1_score(test_y, test_pred)
    print(f"Model: {name}")
    print(f"Train Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")
    print("")
# # 創造決策樹模型
# model = DecisionTreeClassifier(random_state=1012) 
# model2 = DecisionTreeClassifier(random_state=1012,criterion='entropy',max_depth=6)   
# model3 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 1012, max_depth=6)
# # 訓練決策樹模型
# model.fit(train_x, train_y)                       
# model2.fit(train_x, train_y)
# model3.fit(train_x, train_y)
# # 確認模型是否訓練成功
# pred_train = model.predict(train_x)  
# pred_train2 = model2.predict(train_x)  
# pred_train3 = model3.predict(train_x)               
# # 計算準確度
# train_acc = accuracy_score(train_y, pred_train) 
# train_acc2 = accuracy_score(train_y, pred_train2)            
# train_acc3 = accuracy_score(train_y, pred_train3)

# # 輸出準確度
# print('train accuracy: {}'.format(train_acc)) 
# print('train accuracy2: {}'.format(train_acc2))
# print('train accuracy3: {}'.format(train_acc3))

# # 確認模型是否訓練成功
# pred_test = model.predict(test_x) 
# pred_test2 = model2.predict(test_x)   
# pred_test3 = model3.predict(test_x)               
# # 計算準確度
# test_acc = accuracy_score(test_y, pred_test) 
# test_acc2 = accuracy_score(test_y, pred_test2)  
# test_acc3 = accuracy_score(test_y, pred_test3)          

# # 輸出準確度
# print('test accuracy: {}'.format(test_acc)) 
# print('test accuracy2: {}'.format(test_acc2))
# print('test accuracy3: {}'.format(test_acc3))
