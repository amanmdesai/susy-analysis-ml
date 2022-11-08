import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.dpi']= 120


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Dense,
    Input,
    Layer,
)
from tensorflow.keras import Sequential,metrics
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import GlorotUniform

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#/content/SUSY.csv.gz
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

df = pd.read_csv('SUSY.csv')
df.columns = ["label","lepton1-pT", "lepton1-eta", "lepton1-phi", "lepton2-pT", "lepton2-eta", "lepton2-phi", "missing-energy-magnitude", "missing-energy-phi", "MET-rel", "axial-MET", "MR", "MTR2", "R", "MT2", "SR", "MDeltaR", "dPhirb", "cos(thetar1)"]

df_all = df
df_low_level=df[["label","lepton1-pT", "lepton1-eta", "lepton1-phi", "lepton2-pT", "lepton2-eta", "lepton2-phi", "missing-energy-magnitude", "missing-energy-phi"]]
df_high_level= df[["label","MET-rel", "axial-MET", "MR", "MTR2", "R", "MT2", "SR", "MDeltaR", "dPhirb", "cos(thetar1)"]]
del df

levels = ["all","low","high"]

def Model_NN(X_train, X_valid, y_train, y_valid):
  initializer = tf.keras.initializers.GlorotUniform()
  model = Sequential()
  model = keras.Sequential(name="my_sequential")
  model.add(Dense(300, activation="LeakyReLU"))
  model.add(Dense(300, activation="LeakyReLU"))
  model.add(Dense(200, activation="LeakyReLU"))
  model.add(Dense(50, activation="LeakyReLU"))
  model.add(Dense(1,activation="sigmoid"))
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss="binary_crossentropy")
  return model

i = 0
for df in [df_all, df_low_level, df_high_level]:
  y = df[['label']]
  X = df.drop('label',axis=1)
  X=X.to_numpy()
  y=y.to_numpy()
  X_train, X_valid, y_train, y_valid =  train_test_split(X,y,random_state=1,test_size=.35)
  model = Model_NN(X_train, X_valid, y_train, y_valid)
  model_name = "DNN"
  history = model.fit(X_train,y_train.ravel(),epochs=5,batch_size=1024,validation_data=(X_valid, y_valid))
  y_pred = model.predict(X_valid)
  save = pd.DataFrame({"y_pred" : y_pred.ravel(),"y_true" : y_valid.ravel()})
  save.to_csv("data_"+model_name+"_"+levels[i]+".csv", index=False)
  i = i + 1

i = 0
for df in [df_all, df_low_level, df_high_level]:
  y = df[['label']]
  X = df.drop('label',axis=1)
  X=X.to_numpy()
  y=y.to_numpy()
  X_train, X_valid, y_train, y_valid =  train_test_split(X,y,random_state=1,test_size=.35)
  model = xgb.XGBClassifier(max_depth=10,sampling_method='uniform',n_jobs=-1,random_state=1,tree_method='gpu_hist')#criterion='gini',min_samples_leaf=5,max_depth=6,n_jobs=-1)
  model_name = "XGBClassifier"
  model.fit(X_train,y_train.ravel())
  y_pred = model.predict_proba(X_valid)
  save = pd.DataFrame({"y_pred" : y_pred[:,1].ravel(),"y_true" : y_valid.ravel()})
  save.to_csv("data_"+model_name+"_"+levels[i]+".csv", index=False)
  i = i + 1

def make_plot(files):
  fig, ax =plt.subplots()
  fig2, ax2 =plt.subplots()
  fig3, ax3 =plt.subplots()
  for file in files:

    read_df = pd.read_csv(file)
    y_pred = read_df['y_pred'].to_numpy()
    y_true = read_df['y_true'].to_numpy()
    model_name = file.replace("data_","")
    model_name = model_name.replace(".csv","")
    fpr, tpr, thresholds = roc_curve(y_true.ravel(), y_pred.ravel())
    auc_measure = auc(fpr, tpr)

    S = 100*tpr
    B = 1000*fpr
    metric = S/np.sqrt(S+B+.000000001)
    opt_index = np.argmax(metric)
    
    ax3.hist(y_pred.ravel(),label=model_name+'_signal',bins=50,histtype='step')
    ax3.hist(1-(y_pred.ravel()),label=model_name+'_background',bins=50,histtype='step')
    
    ax.plot(tpr, 1-fpr,label=model_name+f' , AUC={auc_measure:.2f}')
    ax2.plot(thresholds,metric,label=model_name+f' OptCut={thresholds[opt_index]:.2f}, Significance={metric[opt_index]:.2f}')




  ax.set_xlabel('Signal Efficiency')
  ax.set_ylabel('Background Rejection')
  ax.set_xlim([0.0, 1.0])
  ax.legend(loc='lower left',title_fontsize='x-small')
  ax2.legend(bbox_to_anchor =(1.65, 1))
  print("")
  print("")

  ax2.set_xlim([0.0, 1.0])
  ax2.set_xlabel('BDT Cut')
  ax2.set_ylabel('Significance')
  ax2.legend(loc='lower left',title_fontsize='x-small',)
  ax2.legend(bbox_to_anchor =(1.1, .5))
  print("")
  print("")
  ax3.set_xlabel('BDT Output')
  ax3.set_ylabel('Counts`')
  ax3.set_yscale('log')
  ax3.legend(loc='lower left',title_fontsize='x-small',)
  ax3.legend(bbox_to_anchor =(1.65, 1))

make_plot(['data_DNN_low.csv','data_DNN_high.csv','data_DNN_all.csv'])

make_plot(['data_XGBClassifier_low.csv','data_XGBClassifier_high.csv','data_XGBClassifier_all.csv'])

files_all=['data_DNN_low.csv','data_DNN_high.csv','data_DNN_all.csv','data_XGBClassifier_low.csv','data_XGBClassifier_high.csv','data_XGBClassifier_all.csv']
make_plot(files_all)
