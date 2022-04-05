from sklearn import tree
import pydotplus
import csv
f = csv.reader(open('1111.csv','r'))
# Outlook (0:Rain, 1:Overcast, 2:Suuny)
# Temprature (0:Cool, 1:Mild, 2:Hot)
# Humidity (0:Normal, 1:High)
# Wind (0:Weak, 1:Strong)
X = [[2,2,1,0],
     [2,2,1,1],
     [1,2,1,0],
     [0,1,1,0],
     [0,0,0,0],
     [0,0,0,1],
     [1,0,0,1],
     [2,1,1,0],
     [2,0,0,0],
     [0,1,0,0],
     [2,1,0,0],
     [1,1,1,1],
     [1,2,0,0],
     [0,1,1,1]]
y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]


# 建立并训练决策树
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# 预测结果
dot_data = tree.export_graphviz(clf, out_file=None,
                     feature_names=['Outlook', 'Temprature','Humidity','Wind'],
                     class_names=['Yes', 'No'],
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_jpg("tree.jpg")	# 生成jpg文件