import os
'''
#print(os.listdir('/content'))
if 'SUSY.csv' in os.listdir('/content'):
  print('file exists')
else:
  print('file downloading')
  !wget http://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
  !gzip -d SUSY.csv.gz
# Link to dataset: http://archive.ics.uci.edu/ml/datasets/SUSY
# 0: background
# 1: signal
'''
#low_level = ["label","lepton1-pT", "lepton1-eta", "lepton1-phi", "lepton2-pT", "lepton2-eta", "lepton2-phi", "missing-energy-magnitude", "missing-energy-phi", "MET-rel"]

!ls

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.dpi']= 200

df = pd.read_csv('SUSY.csv')

df.columns = ["label","lepton1-pT", "lepton1-eta", "lepton1-phi", "lepton2-pT", "lepton2-eta", "lepton2-phi", "missing-energy-magnitude", "missing-energy-phi", "MET-rel", "axial-MET", "MR", "MTR2", "R", "MT2", "SR", "MDeltaR", "dPhirb", "cos(thetar1)"]

df

signal = df.loc[df['label']==1]
background = df.loc[df['label']==0]

import numpy as np

for col in df.columns:
  #min = min(signal[col].min(),background[col].min())
  #max = max(signal[col].max(),background[col].max()).min
  plt.hist(signal[col], alpha=0.4,bins=50,color='b',label='signal')#  range=[min,max]
  plt.hist(background[col],alpha=0.4,bins=50,color='r',label='background')#,range=[min,max]
  plt.xlabel(col)
  plt.yscale('log')
  plt.legend()
  plt.show()

df_low_level = df[["label","lepton1-pT", "lepton1-eta", "lepton1-phi", "lepton2-pT", "lepton2-eta", "lepton2-phi", "missing-energy-magnitude", "missing-energy-phi", "MET-rel"]]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

y = df_low_level[['label']]
X = df_low_level.drop('label',axis=1)
X=X.to_numpy()
y=y.to_numpy()
#y.ravel()

X_train, X_valid, y_train, y_valid =  train_test_split(X[:20_00_000],y[:20_00_000],random_state=1,test_size=.3)

#rand = RandomForestClassifier(criterion='gini',min_samples_leaf=5,max_depth=6,n_jobs=-1)
rand = xgb.XGBClassifier(max_depth=10,n_jobs=-1,random_state=1,)#criterion='gini',min_samples_leaf=5,max_depth=6,n_jobs=-1)
rand.fit(X_train,y_train.ravel())
y_pred_random = rand.predict_proba(X_valid)

#y_pred_random[1:10,1], y_valid[1:10]

plt.hist(y_pred_random[:,0],label='background',bins=50,histtype='step')
plt.hist(y_pred_random[:,1],label='signal',bins=50,histtype='step')
plt.legend()
plt.x_label('BDT Output')
plt.y_label('Counts`')
plt.yscale('log')
plt.show()

fpr_random, tpr_random, thresholds = roc_curve(y_valid.ravel(), y_pred_random[:,1].ravel())
auc_random = auc(fpr_random, tpr_random)
plt.plot(tpr_random, 1/(fpr_random+.000001),label=f'RandomForestClassifier, AUC={auc_random:.2f}')
plt.yscale('log')
plt.xlim([0.0, 1.0])
plt.legend()
plt.show()

S = 100*tpr_random
B = 1000*fpr_random
metric = S/np.sqrt(S+B)
print(np.amax(metric), thresholds[np.argmax(metric)])
plt.plot(thresholds,metric)
plt.x_label('BDT Cut')
plt.y_label('Significance')
plt.show()

'''ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(clf, X_valid, y_valid, ax=ax, alpha=0.8)
#rfc_disp.plot(ax=ax, alpha=0.8)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.show()


def roc_plot(y_pred):
  y_pred = clf.predict_proba(X_valid)
  fpr_random, tpr_random, _ = roc_curve(y_valid.ravel(), y_pred[:,1].ravel())
  auc_random = auc(fpr_random, tpr_random)
  plt.plot(tpr_random, 1/(fpr_random+.000001),label=f'RandomForestClassifier, AUC={auc_random:.2f}')


  plt.yscale('log')
  plt.xlim([0.0, 1.0])
  plt.legend()
  return None'''

#y_pred = clf.predict_proba(X_valid)
#y_pred = clf.predict(X_valid)
#print(y_pred)

#classification_report(y_valid, y_pred2,target_names=['background','signal']) #output_dict=True,

#from sklearn.preprocessing import label_binarize
#y = label_binarize(y, classes=[0, 1])
#n_classes = y.shape[1]
#y
