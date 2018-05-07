#decision_tree
#encoding:utf-8

# 对原始数据进行分为训练数据和测试数据
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import pydotplus

##定义数据转换函数，在读入数据时进行预处理，将每一种属性的值转换为相应的数字值，适应numpy的计算
def outlook_type(s):
    it = {'sunny':1, 'overcast':2, 'rainy':3}
    return it[s]
def temperature(s):
    it = {'hot':1, 'mild':2, 'cool':3}
    return it[s]
def humidity(s):
    it = {'high':1, 'normal':0}
    return it[s]
def windy(s):
    it = {'TRUE':1, 'FALSE':0}
    return it[s]

def play_type(s):
    it = {'yes': 1, 'no': 0}
    return it[s]

play_feature_E = 'outlook', 'temperature', 'humidity', 'windy'
play_class = 'yes', 'no'

## 1、读入数据，并将原始数据中的数据转换为数字形式，其中，converters是一个字典, 表示第i列使用其后的函数来进行预处理
##data = np.loadtxt("play.tennies.txt", delimiter=" ", dtype=str,  converters={0:outlook_type, 1:temperature, 2:humidity, 3:windy,4:play_type})
data=[
 [1,1,1,0,0],
 [1,1,1,1,0],
 [2,1,1,0,1],
 [3,2,1,0,1],
 [3,3,0,0,1],
 [3,3,0,1,0],
 [2,3,0,1,1],
 [1,2,1,0,0],
 [1,3,0,0,1],
 [3,2,0,0,1],
 [1,2,0,1,1],
 [2,2,1,1,1],
 [2,1,0,0,1],
 [3,2,1,1,0]
 ]

x, y = np.split(data,(4,),axis=1)
## split(数据，分割位置，轴=1or0)axis=1,代表列，是要把data数据集中的所有数据按第四、五列之间分割为X集和Y集。

# 2、拆分训练数据与测试数据，为了进行交叉验证
'''sklearn.model_selection.train_test_split随机划分训练集与测试集,train_test_split(train_data,train_target,test_size=数字, random_state=0)

　　参数解释：

　　train_data：所要划分的样本特征集

　　train_target：所要划分的样本结果

　　test_size：样本占比，如果是整数的话就是样本的数量

　　random_state：是随机数的种子'''
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# 3、初始化决策树模型,使用信息熵作为划分标准，对决策树进行训练
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train, y_train)#调用模型中fit函数/模块训练模型

# 4、把决策树结构写入文件
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=play_feature_E, class_names=play_class, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('decision_tree1.pdf')

# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
print(clf.feature_importances_)

# 5、使用训练数据预测，预测结果完全正确
answer = clf.predict(x_train)
y_train = y_train.reshape(-1)
print(answer)
print(y_train)
print(np.mean(answer == y_train))

# 6、对测试数据进行预测，准确度较低，说明过拟合
answer = clf.predict(x_test)
y_test = y_test.reshape(-1)
print(answer)
print(y_test)
print(np.mean(answer == y_test))
