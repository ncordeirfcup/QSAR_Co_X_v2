import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
#from PIL import ImageTk, Image
#import pymysql
import os
import shutil
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from testset_prediction import testset_prediction as tsp
from cross_validation2 import cross_validation as cv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import sys

import warnings
warnings.filterwarnings("ignore")



initialdir=os.getcwd()
#RF=RandomForestClassifier()
sc = StandardScaler()

def write_versions(filer):
    import sklearn
    import xgboost
    import lightgbm
    filer.write('Scikit-learn version: '+ str(sklearn.__version__)+'\n')
    filer.write('XGBoost version: '+ str(xgboost.__version__)+'\n')
    filer.write('LightGBM version: '+ str(lightgbm.__version__)+'\n')
    filer.write('Numpy version: '+str(np.__version__)+'\n')
    filer.write('Pandas version: '+str(pd.__version__)+'\n')
    filer.write('Python version:' + str(sys.version))
    filer.write('\n')

def data1():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    global col1
    col1 = list(file1.head(0))

def data2():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)

def data3():
    global filename3
    filename3 = askopenfilename(initialdir=initialdir,title = "Select validation file")
    fifthEntryTabOne.delete(0, END)
    fifthEntryTabOne.insert(0, filename3)
    global e_
    e_,f_=os.path.splitext(filename3)
    global file3
    file3 = pd.read_csv(filename3)

def data4():
    global filename4
    filename4 = askopenfilename(initialdir=initialdir,title = "Select parameter file")
    thirdEntryTabThree_x.delete(0, END)
    thirdEntryTabThree_x.insert(0, filename4)
    global e_x
    e_x,f_x=os.path.splitext(filename4)
    global file4
    file4 = pd.read_csv(filename4)

def correlation(X,cthreshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] > cthreshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in X.columns:
                    del X[colname] # deleting the column from the dataset
    return X

def variance(X,threshold):
    from sklearn.feature_selection import VarianceThreshold
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]
    return X

def pretreat(X,cthreshold,vthreshold):
    X=correlation(X,cthreshold)
    X=variance(X,vthreshold)
    return X

def parse_value(v):
    v = str(v).strip()
    #print(v)

    # Boolean
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False

    # None
    if v.lower() == "none":
        return None

    # Integer
    try:
        if "." in v:
            if v.split('.')[1]=='0':
                if int(v.split('.')[0])>0:
                   return int(v.split('.')[0])
            elif float(v)>0:
                   return float(v)
        else:
            return int(v)

    except:
        pass

    # Float

    # String
    return v

def csv_to_param_grid(csv_file):
    df = pd.read_csv(csv_file)

    param_grid = {}

    for col in df.columns:
        values = df[col].dropna().unique()
        param_grid[col] = [parse_value(v) for v in values]

    return param_grid

def mcc_score(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0  # avoid division by zero
    
    return numerator / denominator

def selected():
    param_grid=csv_to_param_grid(filename4)
    #print(param_grid)

    if Criterion.get()==1:
        L11T1 ['state']='normal'
        MaxFeatures1_None['state']='normal'
        MaxFeatures1_integer['state']='normal'
        E10T1['state']='normal'
        if Criterion26.get()=='none':
           rd=None
        elif Criterion26.get()=='integer':

             rd=int(E10T1.get())
        estimator1=RandomForestClassifier(random_state=rd)
        rn='RF'
        param_grid['bootstrap']=[True,False]

    elif Criterion.get()==2:
        thirdLabelTabThreer2c1_h['state']='disabled'
        thirdEntryTabThreer3c1_h['state']='disabled'
        thirdEntryTabThreer3c1_h1['state']='disabled'
        thirdEntryTabThreer3c1_h2['state']='disabled'
        L11T1 ['state']='disabled'
        MaxFeatures1_None['state']='disabled'
        MaxFeatures1_integer['state']='disabled'
        E10T1['state']='disabled'
        estimator1=KNeighborsClassifier()
        rn='KNN'
    elif Criterion.get()==3:
        thirdLabelTabThreer2c1_h['state']='disabled'
        thirdEntryTabThreer3c1_h['state']='disabled'
        thirdEntryTabThreer3c1_h1['state']='disabled'
        thirdEntryTabThreer3c1_h2['state']='disabled'
        L11T1 ['state']='disabled'
        MaxFeatures1_None['state']='disabled'
        MaxFeatures1_integer['state']='disabled'
        E10T1['state']='disabled'
        estimator1=BernoulliNB()
        rn='NB'
        param_grid['fit_prior']=[True,False]
    elif Criterion.get()==4:
        thirdLabelTabThreer2c1_h['state']='disabled'
        thirdEntryTabThreer3c1_h['state']='disabled'
        thirdEntryTabThreer3c1_h1['state']='disabled'
        thirdEntryTabThreer3c1_h2['state']='disabled'
        L11T1 ['state']='disabled'
        MaxFeatures1_None['state']='disabled'
        MaxFeatures1_integer['state']='disabled'
        E10T1['state']='disabled'
        estimator1=SVC(probability=True)
        rn='SVC'
    elif Criterion.get()==5:
         L11T1 ['state']='normal'
         MaxFeatures1_None['state']='normal'
         MaxFeatures1_integer['state']='normal'
         E10T1['state']='normal'
         if Criterion26.get()=='none':
            rd=None
         elif Criterion26.get()=='integer':
             rd=int(E10T1.get())
         estimator1=GradientBoostingClassifier(verbose=1,random_state=rd)
         rn='GB'
    elif Criterion.get()==7:
         L11T1 ['state']='normal'
         MaxFeatures1_None['state']='normal'
         MaxFeatures1_integer['state']='normal'
         E10T1['state']='normal'
         if Criterion26.get()=='none':
            rd=None
         elif Criterion26.get()=='integer':
             rd=int(E10T1.get())
         #estimator=GradientBoostingClassifier(verbose=1,random_state=rd)
         estimator1 = AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=42), algorithm="SAMME",random_state=rd)
         rn='AB'

    elif Criterion.get()==8:
         L11T1 ['state']='normal'
         MaxFeatures1_None['state']='normal'
         MaxFeatures1_integer['state']='normal'
         E10T1['state']='normal'
         if Criterion26.get()=='none':
            rd=None
         elif Criterion26.get()=='integer':
             rd=int(E10T1.get())
         estimator1=XGBClassifier(objective="binary:logistic", eval_metric="logloss",use_label_encoder=False,random_state=rd, error_score='raise')
         rn='XGB'

    elif Criterion.get()==6:
         thirdLabelTabThreer2c1_h['state']='normal'
         thirdEntryTabThreer3c1_h['state']='normal'
         thirdEntryTabThreer3c1_h1['state']='normal'
         thirdEntryTabThreer3c1_h2['state']='normal'
         L11T1 ['state']='normal'
         MaxFeatures1_None['state']='normal'
         MaxFeatures1_integer['state']='normal'
         E10T1['state']='normal'
         if Criterion26.get()=='none':
            rd=None
         elif Criterion26.get()=='integer':
              rd=int(E10T1.get())
         estimator1=MLPClassifier(random_state=rd)
         rn='MLP'
         if int(thirdEntryTabThreer3c1_h.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),)]
         elif 'hidden_layer_sizes' in file4.columns:
               lst = file4['hidden_layer_sizes'].values.tolist()
               result = [tuple(map(int, x.split(','))) for x in lst]
               param_grid['hidden_layer_sizes']=result
         else:
               pass
         if int(thirdEntryTabThreer3c1_h1.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()))]
         if int(thirdEntryTabThreer3c1_h2.get())!=0:
            param_grid ['hidden_layer_sizes']=[(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()),int(thirdEntryTabThreer3c1_h2.get()))]
         #print(param_grid)
    elif Criterion.get()==9:
        L11T1 ['state']='normal'
        MaxFeatures1_None['state']='normal'
        MaxFeatures1_integer['state']='normal'
        E10T1['state']='normal'
        if Criterion26.get()=='none':
           rd=None
        elif Criterion26.get()=='integer':
            rd=int(E10T1.get())
        estimator1=CatBoostClassifier(verbose=0,loss_function="Logloss", random_state=rd)
        rn='CB'
    elif Criterion.get()==10:
        L11T1 ['state']='normal'
        MaxFeatures1_None['state']='normal'
        MaxFeatures1_integer['state']='normal'
        E10T1['state']='normal'
        if Criterion26.get()=='none':
           rd=None
        elif Criterion26.get()=='integer':
            rd=int(E10T1.get())
        estimator1= ExtraTreesClassifier(random_state=rd, n_jobs=-1)
        rn='ET'


    else:
        pass
    print(param_grid)
    rn='g_'+rn
    return estimator1,param_grid,rn



def sol():
    X_train=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    X_test=file2.iloc[:,2:]
    yts=file2.iloc[:,1:2]
    nts=file2.iloc[:,0:1]
    global Xtr
    if stc1.get()=='yes':
       Xtr = pd.DataFrame(np.round(sc.fit_transform(X_train),4))
       Xts = pd.DataFrame(np.round(sc.transform(X_test),4))
       Xtr.columns=X_train.columns
       Xts.columns=X_test.columns
    elif stc1.get()=='no':
       Xtr=X_train
       Xts=X_test
    estimator,param_grid,rn=selected()
    cvg=thirdEntryTabOne.get()
    cvg=int(cvg)
    cvm=forthEntryTabOne.get()
    cvm=int(cvm)
    #param_grid=paramgrid()
    clf = GridSearchCV(cv=cvg, estimator=estimator, param_grid=param_grid, n_jobs=-1)
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    Xtr=pretreat(Xtr,cthreshold,vthreshold)
    Xts=Xts[Xtr.columns]
    Xtr.to_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    Xts.to_csv('Pretreat_test_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    clf.fit(Xtr,ytr)
    global clfb
    clfb=clf.best_estimator_
    clfb.fit(Xtr,ytr)
    pname=str(c_)+rn+"_tr.pkl"
    pickle.dump(clfb,open(pname,"wb"))
    #clf.best_estimator_.score(Xtr,ytr)
    #cvv=cv(Xtr,ytr,clf.best_estimator_,cvm)
    #print(clf.best_estimator_,clf.best_score_)
    #print(cvv.fit())
    #ts=tsp(Xts,yts,clf.best_estimator_)
    #print(ts.fit())
    #global filer

    if stc1.get()=='yes':
       filer = open(str(c_)+rn+"_wstd_tr.txt","w")
    elif stc1.get()=='no':
        filer = open(str(c_)+rn+"_tr.txt","w")
    write_versions(filer)

    filer.write('The best estimator is: '+'\n')
    filer.write(str(clf.best_estimator_))
    filer.write("\n")
    filer.write((str(cvm)+' fold cross validation statistics are: '+'\n'))
    filer.write("\n")
    writefile2(Xtr,ytr,clfb,cvm,filer)
    filer.write("\n")
    filer.write('The test set results are: '+'\n')
    filer.write("\n")
    a=writefile(Xts,yts,clfb,filer)
    filer.write("\n")
    #file2n=file2[file2.iloc[:,0:1].columns.tolist()+file2.iloc[:,2:].columns.tolist()+file2.iloc[:,1:2].columns.tolist()]
    file2n=pd.concat([file2.iloc[:,0:1],Xts,file2.iloc[:,1:2]], axis=1)
    b=pd.concat([file2n,a],axis=1)
    b['Set'] = 'Test'
    b.to_csv(str(c_)+str(rn)+'_tspred.csv', index=False)
    label='test'
    color='blue'
    global pyplot
    pyplot.figure(figsize=(15,10))

    ROCplot(clfb,Xts,yts,label,color,pyplot)


def sol2():
    #file,plt=sol()
    estimator,pg,rn=selected()
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    selDes=pd.read_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    Xvd=file3[list(selDes)[1:]]
    #Xvd=Xvd[Xtr.columns]
    if stc1.get()=='yes':
       Xvd = pd.DataFrame(np.round(sc.transform(Xvd),4))
       Xvd.columns=list(selDes)[1:]
    elif stc1.get()=='no':
       Xvd=Xvd
    Xvd.to_csv('Pretreat_valid_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    ytr=file1.iloc[:,1:2]

    if ytr.columns[0] in file3.columns:
        yvd=file3.iloc[:,1:2]
        nvd=file3.iloc[:,0:1]
        if stc1.get()=='yes':
           filer = open(str(c_)+rn+"_wstd_vd.txt","w")
        elif stc1.get()=='no':
            filer = open(str(c_)+rn+"_vd.txt","w")
        filer.write('The validation set results are: '+'\n')
        filer.write("\n")
        a=writefile(Xvd,yvd,clfb,filer)
        file3n=pd.concat([file3.iloc[:,0:1],Xvd,file3.iloc[:,1:2]], axis=1)
        b=pd.concat([file3n,a],axis=1)
        b['Set'] = 'Validation'
        b.to_csv(str(c_)+str(rn)+'_vdpred.csv', index=False)
        label='validation'
        color='red'
        ROCplot(clfb,Xvd,yvd,label,color,pyplot)
    else:
        #est.fit(Xtr,ytr)
        yprvd=pd.DataFrame(clfb.predict(Xvd))
        yprvd.columns=['Pred']
        yprvd2=pd.DataFrame(clfb.predict_proba(Xvd))
        yprvd2.columns=['%Prob(-1)','%Prob(+1)']
        dfsvd=pd.concat([file3.iloc[:,1:2],Xvd,yprvd,yprvd2],axis=1)
        dfsvd['Diff']=abs(dfsvd['%Prob(-1)']-dfsvd['%Prob(+1)'])
        dfsvd['Outlier_info(Confidence estimation approach, Threshold 0.5)']=['In' if x>=0.5 else 'Outlier' for x in dfsvd['Diff']]
        dfsvd['Set'] = 'Screening'
        dfsvd.to_csv(str(c_)+str(rn)+'_scpred.csv', index=False)

def ROCplot(model,X,y,label,color,plt):
    #model.fit(X,y)
    s,p,t=selected()
    lr_probs =model.predict_proba(X)
    lr_probs = lr_probs[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
    #pyplot.figure(figsize=(15,10))
    plt.plot(lr_fpr,lr_tpr, label=label, color=color, marker='.',  linewidth=1, markersize=10)
    plt.ylabel('True postive rate',fontsize=28)
    plt.xlabel('False postive rate',fontsize=28)
    plt.legend(fontsize=18)
    plt.tick_params(labelsize=18)
    rocn=str(c_)+str(t)+'_ROC.png'
    plt.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,metadata=None)


def writefile(X,y,model,filerw):
    ts=tsp(X,y,model)
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10=ts.fit()
    filerw.write('True Positive: '+str(a1)+"\n")
    filerw.write('True Negative: '+str(a2)+"\n")
    filerw.write('False Positive '+str(a3)+"\n")
    filerw.write('False Negative '+str(a4)+"\n")
    filerw.write('Sensitivity: '+str(a5)+"\n")
    filerw.write('Specificity: '+str(a6)+"\n")
    filerw.write('Accuracy: '+str(a7)+"\n")
    filerw.write('f1_score: '+str(a8)+"\n")
    #filer.write('Recall score: '+str(recall_score(self.y,ypred))
    filerw.write('MCC: '+str(a9)+"\n")
    filerw.write('ROC_AUC: '+str(a10)+"\n")
    ypred=pd.DataFrame(model.predict(X))
    ypred.columns=['Pred']
    ypr2=pd.DataFrame(model.predict_proba(X))
    ypr2.columns=['%Prob(-1)','%Prob(+1)']
    dfsvd=pd.concat([ypred,ypr2],axis=1)
    dfsvd['Diff']=abs(dfsvd['%Prob(-1)']-dfsvd['%Prob(+1)'])
    dfsvd['Outlier_info(Confidence estimation approach, Threshold 0.5)']=['In' if x>=0.5 else 'Outlier' for x in dfsvd['Diff']]
    return dfsvd

def writefile2(X,y,model,cvm,filerw):
    cvv=cv(X,y,model,cvm)
    a1,a2,a3,a4,a5,a6,a7,a8=cvv.fit()
    filerw.write('True Positive: '+str(a1)+"\n")
    filerw.write('True Negative: '+str(a2)+"\n")
    filerw.write('False Positive '+str(a3)+"\n")
    filerw.write('False Negative '+str(a4)+"\n")
    filerw.write('Sensitivity: '+str(a5)+"\n")
    filerw.write('Specificity: '+str(a6)+"\n")
    filerw.write('Accuracy: '+str(a7)+"\n")
    filerw.write('f1_score: '+str(a8)+"\n")
    filerw.write('MCC score: '+str(mcc_score(a1, a2, a3, a4))+"\n")
    #filer.write('Recall score: '+str(recall_score(self.y,ypred))
    #filerw.write('MCC: '+str(a9)+"\n")
    #filerw.write('ROC_AUC: '+str(a10)+"\n")

form = tk.Tk()
form.title("QSAR-Co-X (Module-2)")
form.geometry("750x750")
form.configure(bg="#f5f5f5")

style = ttk.Style()
style.configure("TLabel", font=("Helvetica", 11))
style.configure("TButton", font=("Helvetica", 10))

tab_parent = ttk.Notebook(form)
tab_parent.pack(fill="both", expand=True, padx=10, pady=10)

tab1 = ttk.Frame(tab_parent)
tab_parent.add(tab1, text="Grid search based non-linear model")

# ===============================
# DATA FILE SECTION
# ===============================

file_frame = ttk.LabelFrame(tab1, text="Input Files")
file_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

ttk.Label(file_frame, text="Sub-training set").grid(row=0, column=0, padx=10, pady=5, sticky="w")
firstEntryTabThree = ttk.Entry(file_frame, width=60)
firstEntryTabThree.grid(row=0, column=1, padx=10)
ttk.Button(file_frame, text="Browse", command=data1).grid(row=0, column=2)

ttk.Label(file_frame, text="Test set").grid(row=1, column=0, padx=10, pady=5, sticky="w")
secondEntryTabThree = ttk.Entry(file_frame, width=60)
secondEntryTabThree.grid(row=1, column=1, padx=10)
ttk.Button(file_frame, text="Browse", command=data2).grid(row=1, column=2)

ttk.Label(file_frame, text="Parameter file").grid(row=2, column=0, padx=10, pady=5, sticky="w")
thirdEntryTabThree_x = ttk.Entry(file_frame, width=60)
thirdEntryTabThree_x.grid(row=2, column=1, padx=10)
ttk.Button(file_frame, text="Browse", command=data4).grid(row=2, column=2)

# ===============================
# DESCRIPTOR STANDARDIZATION
# ===============================

ttk.Label(file_frame, text="Standardization:").grid(row=3, column=0, sticky="w", pady=10)

stc1 = StringVar()
ttk.Radiobutton(file_frame, text='Yes', variable=stc1, value='yes').grid(row=3, column=1, sticky="w")
ttk.Radiobutton(file_frame, text='No', variable=stc1, value='no').grid(row=3, column=1, padx=60, sticky="w")

# ===============================
# MODEL SELECTION
# ===============================

model_frame = ttk.LabelFrame(tab1, text="Model Selection")
model_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

Criterion = IntVar()

Criterion_RF = ttk.Radiobutton(model_frame, text='Random Forest', variable=Criterion, value=1, command=selected)
Criterion_KNN = ttk.Radiobutton(model_frame, text='k-Nearest Neighbor', variable=Criterion, value=2, command=selected)
Criterion_NB = ttk.Radiobutton(model_frame, text='BernoulliNB', variable=Criterion, value=3, command=selected)
Criterion_SVC = ttk.Radiobutton(model_frame, text='SVC', variable=Criterion, value=4, command=selected)
Criterion_AB = ttk.Radiobutton(model_frame, text='AdaBoost', variable=Criterion, value=7, command=selected)
Criterion_XGB = ttk.Radiobutton(model_frame, text='XGBoost', variable=Criterion, value=8, command=selected)
Criterion_GB = ttk.Radiobutton(model_frame, text='Gradient Boosting', variable=Criterion, value=5, command=selected)
Criterion_MLP = ttk.Radiobutton(model_frame, text='MLP', variable=Criterion, value=6, command=selected)
Criterion_CB = ttk.Radiobutton(model_frame, text='CatBoost', variable=Criterion, value=9, command=selected)
Criterion_ET = ttk.Radiobutton(model_frame, text='Extra Trees', variable=Criterion, value=10, command=selected)

Criterion_RF.grid(row=0, column=0, padx=10, pady=5)
Criterion_KNN.grid(row=0, column=1, padx=10)
Criterion_NB.grid(row=0, column=2, padx=10)
Criterion_SVC.grid(row=0, column=3, padx=10)

Criterion_AB.grid(row=1, column=0, padx=10)
Criterion_XGB.grid(row=1, column=1, padx=10)
Criterion_GB.grid(row=1, column=2, padx=10)
Criterion_MLP.grid(row=1, column=3, padx=10)
Criterion_CB.grid(row=0, column=4, padx=10)
Criterion_ET.grid(row=1, column=4, padx=10)

# ===============================
# MLP HIDDEN LAYER
# ===============================

mlp_frame = ttk.LabelFrame(tab1, text="MLP Hidden Layers")
mlp_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

v1=IntVar(value=0)
v2=IntVar(value=0)
v3=IntVar(value=0)

thirdLabelTabThreer2c1_h = ttk.Label(mlp_frame, text='Hidden layers')
thirdLabelTabThreer2c1_h.grid(row=0, column=0, padx=10)

thirdEntryTabThreer3c1_h = ttk.Entry(mlp_frame, textvariable=v1, width=6)
thirdEntryTabThreer3c1_h.grid(row=0, column=1)

thirdEntryTabThreer3c1_h1 = ttk.Entry(mlp_frame, textvariable=v2, width=6)
thirdEntryTabThreer3c1_h1.grid(row=0, column=2)

thirdEntryTabThreer3c1_h2 = ttk.Entry(mlp_frame, textvariable=v3, width=6)
thirdEntryTabThreer3c1_h2.grid(row=0, column=3)

# ===============================
# RANDOM STATE
# ===============================

random_frame = ttk.LabelFrame(tab1, text="Random State")
random_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

Criterion26 = StringVar()
Criterion26.set('none')

L11T1 = ttk.Label(random_frame, text="Random state (except KNN/BNB/SVC)")
L11T1.grid(row=0, column=0, padx=10)

MaxFeatures1_None = ttk.Radiobutton(
    random_frame, text='None', variable=Criterion26, value='none'
)
MaxFeatures1_None.grid(row=0, column=1)

MaxFeatures1_integer = ttk.Radiobutton(
    random_frame, text='Number', variable=Criterion26, value='integer'
)
MaxFeatures1_integer.grid(row=0, column=2)

E10T1 = ttk.Entry(random_frame, width=8)
E10T1.grid(row=0, column=3)

# ===============================
# PRETREATMENT
# ===============================

pretreat_frame = ttk.LabelFrame(tab1, text="Descriptor Pretreatment")
pretreat_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")

ttk.Label(pretreat_frame, text="Correlation cut-off").grid(row=0, column=0, padx=10, pady=5)
thirdEntryTabThreer3c1 = ttk.Entry(pretreat_frame, width=10)
thirdEntryTabThreer3c1.grid(row=0, column=1)

ttk.Label(pretreat_frame, text="Variance cut-off").grid(row=1, column=0, padx=10)
fourthEntryTabThreer5c1 = ttk.Entry(pretreat_frame, width=10)
fourthEntryTabThreer5c1.grid(row=1, column=1)

# ===============================
# CROSS VALIDATION
# ===============================

cv_frame = ttk.LabelFrame(tab1, text="Cross Validation")
cv_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")

ttk.Label(cv_frame, text="CV for Grid Search").grid(row=0, column=0, padx=10)
thirdEntryTabOne = ttk.Entry(cv_frame, width=10)
thirdEntryTabOne.grid(row=0, column=1)

ttk.Label(cv_frame, text="CV for Model Predictability").grid(row=1, column=0, padx=10)
forthEntryTabOne = ttk.Entry(cv_frame, width=10)
forthEntryTabOne.grid(row=1, column=1)

# ===============================
# RUN BUTTON
# ===============================

run_frame = ttk.Frame(tab1)
run_frame.grid(row=7, column=0, pady=10)

b2 = ttk.Button(run_frame, text="Generate Model", command=sol)
b2.grid(row=0, column=0, padx=10)

# ===============================
# VALIDATION
# ===============================

validation_frame = ttk.LabelFrame(tab1, text="Validation Set")
validation_frame.grid(row=8, column=0, padx=10, pady=10, sticky="ew")

fifthEntryTabOne = ttk.Entry(validation_frame, width=60)
fifthEntryTabOne.grid(row=0, column=0, padx=10)

ttk.Button(validation_frame, text="Browse", command=data3).grid(row=0, column=1)

b4 = ttk.Button(validation_frame, text="Submit", command=sol2)
b4.grid(row=1, column=0, pady=10)

form.mainloop()
