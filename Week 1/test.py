import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import csv

# read the data
with open('score.csv', newline='') as f:
    reader = csv.reader(f)
    s = list(reader)
tmp = s[0]
scores = np.array([float(item) for item in tmp])
with open('label.csv', newline='') as f:
    reader = csv.reader(f)
    l = list(reader)
tmp = l[0]
label = np.array([float(item) for item in tmp])

# build confusion matrix with a threshold of 0.05 a
threshold = 0.05
label_pred=scores.copy()
label_pred[scores>threshold]=1
label_pred[scores<threshold]=0
cm = metrics.confusion_matrix(label, label_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# get TP, FP, TN, FN from confusion matrix
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

# compute Precision, Recall, F1-score and Accuracy
P=TP/(TP+FP)
R=TP/(TP+FN)
F1=(2*P*R)/(P+R)
acc=(TP+TN)/len(label)
print("Precision: ",P,"\nRecall:",R,"\nF1-score: ",F1,"\nAccuracy: ",acc)

# compute FPR, TPR and draw ROC curve
fpr, tpr, thresholds = metrics.roc_curve(label, scores, pos_label=1)
print(thresholds)
roc_auc = metrics.auc(fpr, tpr)
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    label="ROC curve (area = %0.2f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

# compute AUC