print(__doc__)

import sys
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import matthews_corrcoef

random_state = np.random.RandomState(0)
f = open(sys.argv[1])
f.readline()#skip header
data = np.loadtxt(f, delimiter=",")
np.random.shuffle(data)
X = data[:,1:]
y = data[:,0]

# SVM Classification and ROC analysis
# with stratified cross-validation
# python roc.py tdat_cdhit_N25_100slt_infog.csv


cv = StratifiedKFold(y, n_folds=5)
#classifier = svm.SVC(kernel='rbf', probability=True,
                     #random_state=random_state)

classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

# k-fold cross-validation for svm  
for i, (train, test) in enumerate(cv):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

pred = []
for x in probas_[:,1]:
	if x > 0.5:
		pred.append(1)
	else:
		pred.append(-1)
print "MCC: ", matthews_corrcoef(y[test], pred)
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
