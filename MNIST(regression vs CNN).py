import pandas as pd
import numpy as np
import tensorflow
from sklearn import preprocessing

# Step1.資料的御處理(Preprocessing data)
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
    
print("\n\n訓練集資料：\n\n",train_data)
print("\n\n測試集資料：\n\n",test_data)

# 觀察知此資料無缺值(NAN)
train_data.info()

train_data.columns

train_data.shape 
# 觀察知有784個特徵及1個標籤

train_data.describe() 

np.sort(train_data['label'].unique())
#無缺值、無須編碼(也可one-hot)
#無須特徵篩選

train_data_feature = train_data.drop('label',axis=1)  #捨去label特徵
train_data_label = train_data['label']

train_data_feature = train_data_feature/255 #normalize
test_data_feature = test_data/255


# Step2.模型選擇與建立(Data choose and build)
##1
from sklearn.model_selection import train_test_split
from sklearn import svm
train_feature, val_feature, train_label, val_label = train_test_split(train_data_feature,train_data_label,test_size=0.2)

svm_model = svm.SVC()
svm_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("支持向量機(Support Vector Machines)模型準確度(訓練集):",svm_model.score(train_feature, train_label))
print ("支持向量機(Support Vector Machines)模型準確度(測試集):",svm_model.score(val_feature, val_label))
svm_model_acc = svm_model.score(val_feature, val_label)


##2
from sklearn.neighbors import KNeighborsClassifier

KNeighbors_model = KNeighborsClassifier(n_neighbers=2)
KNeighbors_model.fit(train_feature, train_label)

print ("最近的鄰居(Nearest Neighbors)模型準確度(訓練集)：",KNeighbors_model.score(train_feature, train_label))
print ("最近的鄰居(Nearest Neighbors)模型準確度(測試集)：",KNeighbors_model.score(val_feature, val_label))
KNeighbors_model_acc = KNeighbors_model.score(val_feature, val_label)

##3
from sklearn import tree

DecisionTree_model = tree.DecisionTreeClassifier()
DecisionTree_model.fit(train_feature, train_label)

print ("決策樹(Decision Trees)模型準確度(訓練集)：",DecisionTree_model.score(train_feature, train_label))
print ("決策樹(Decision Trees)模型準確度(測試集)：",DecisionTree_model.score(val_feature, val_label))
DecisionTree_model_acc = DecisionTree_model.score(val_feature, val_label)


##4
from sklearn.ensemble import RandomForestClassifier

RandomForest_model = RandomForestClassifier(n_estimators=10)
RandomForest_model.fit(train_feature, train_label)

print ("隨機森林(Forests of randomized trees)模型準確度(訓練集)：",RandomForest_model.score(train_feature, train_label))
print ("隨機森林(Forests of randomized trees)模型準確度(測試集)：",RandomForest_model.score(val_feature, val_label))
RandomForest_model_model_acc = RandomForest_model.score(val_feature, val_label)

##5
from sklearn.neural_network import MLPClassifier

MLP_model = MLPClassifier(solver='lbfgs', 
                                   alpha=1e-4,
                                   hidden_layer_sizes=(10, 10), 
                                   )
MLP_model.fit(train_feature, train_label)

#將原本train_data資料，分成訓練集與測試集合，並丟入訓練好的模型測試準確度
print ("神經網路(Neural Network models)模型準確度(訓練集)：",MLP_model.score(train_feature, train_label))
print ("神經網路(Neural Network models)模型準確度(測試集)：",MLP_model.score(val_feature, val_label))
MLP_model_acc = MLP_model.score(val_feature, val_label)

##6
# CNN 模型
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, Adam

# 建立 CNN 模型
CNN_model = keras.Sequential([
    # 輸入層，將 784 維的向量轉換為 28x28x1 的圖像結構
    layers.Reshape((28, 28, 1), input_shape=[784]),

    # 卷積層 1 + 批正規化 + Dropout
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.25),  # Dropout

    # 卷積層 2 + 批正規化 + Dropout
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.25),  # Dropout

    # 卷積層 3 + 批正規化 + Dropout
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.25),  # Dropout

    # 平坦層
    layers.Flatten(),

    # 全連接層 + 批正規化 + Dropout
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Dropout

    # 輸出層（假設分類問題有 10 個類別）
    layers.Dense(10, activation='softmax')
])

# 顯示模型架構
#CNN_model.summary()

optimizer = tensorflow.keras.optimizers.Adam(epsilon=0.001)

CNN_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  # 適合於整數標籤的分類問題
    metrics=['accuracy']
)


history = CNN_model.fit(
    train_feature, train_label,
    epochs=10,
    batch_size=32,
    validation_data=(val_feature, val_label)
)


CNN_model_train_loss, CNN_model_train_acc = CNN_model.evaluate(train_feature, train_label, verbose=0)
CNN_model_loss, CNN_model_acc = CNN_model.evaluate(val_feature, val_label, verbose=0)
print ("Convolution Neural Network )模型準確度(訓練集)：",CNN_model_train_acc)
print ("Convolution Neural Network )模型準確度(測試集)：",CNN_model_acc,"Loss:", CNN_model_loss)




models = pd.DataFrame({
    'Model': ['支持向量機(Support Vector Machines)', 
              '最近的鄰居(Nearest Neighbors)', 
              '決策樹(Decision Trees)',
              '隨機森林(Forests of randomized trees)', 
              '神經網路(Neural Network models)',
              'Convolution Neural Network (CNN)',
             ],
    'Score': [svm_model_acc,
              KNeighbors_model_acc,
              DecisionTree_model_acc,
              RandomForest_model_model_acc,
              MLP_model_acc,
              CNN_model_acc, 
              ]
                       })
models.sort_values(by='Score', ascending=False)

# STep3.模型驗證(Model validation)
#CNN
#test_data無需處理
CNN_model.fit(
    train_data_feature, train_data_label,
    epochs=10,
    batch_size=32,
)
label_predict_cnn = CNN_model.predict(test_data_feature)

label_predict_cnn = np.argmax(label_predict_cnn,axis = 1)
print(label_predict_cnn)

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
output = pd.DataFrame({'ImageId': sample_submission["ImageId"], 'Label': label_predict_cnn})
output.to_csv('Submission_cnn.csv', index=False)