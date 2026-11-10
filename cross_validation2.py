from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,matthews_corrcoef,recall_score,roc_auc_score,roc_curve,confusion_matrix
import numpy as np
import math
from sklearn.model_selection import KFold, StratifiedKFold  #For K-fold cross validation\n
import random
import copy
import math
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import cross_val_predict

class cross_validation:
      def __init__(self,X_data,y_data, model,cv):
            self.X_data=X_data
            self.y_data=y_data
            self.cv=cv
            self.model=model
        
      def fit(self):
          cvp = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
          y_pred_CV_GBM = cross_val_predict(self.model, self.X_data, self.y_data, cv=cvp)
          cm1 = metrics.confusion_matrix(self.y_data, y_pred_CV_GBM, labels=[0,1])
          TNs, FPs, FNs, TPs = cm1.ravel()
          Sn=(TPs/(TPs+FNs))*100
          Sp=(TNs/(TNs+FPs))*100
          PRs=(TPs/(TPs+FPs))*100
          F1s=(2*TPs / (2*TPs + FPs + FNs))*100
          Acc=((TPs+TNs)/(TPs+TNs+FPs+FNs))*100
          MCC=(TPs*TNs-FPs*FNs)/math.sqrt((TPs+FPs)*(TPs+FNs)*(TNs+FPs)*(TNs+FNs))
          print('True positive: '+str(TPs))
          print('True negative: '+str(TNs))
          print('False postive: '+str(FPs))
          print('False negative: '+str(FNs))
          print('Sensitivity: '+str(Sn))
          print('Specificity: '+str(Sp))
          print('Accuracy: '+str(Acc))
          print('MCC: '+str(MCC))
          print('Precision: '+str(PRs))
          print('F1 score: '+str(F1s))
          return TPs,TNs,FPs,FNs,Sn,Sp,Acc,F1s

         



