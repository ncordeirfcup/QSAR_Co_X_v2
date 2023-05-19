import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
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
from cross_validation import cross_validation as cv
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



initialdir=os.getcwd()
RF=RandomForestClassifier()

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


def sol():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    Xts=file2.iloc[:,2:]
    yts=file2.iloc[:,1:2]
    nts=file2.iloc[:,0:1]
    estimator,rn=Selected()
    cvm=forthEntryTabOne.get()
    cvm=int(cvm)
    #param_grid=paramgrid()
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    Xtr=pretreat(Xtr,cthreshold,vthreshold)
    Xts=Xts[Xtr.columns]
    Xtr.to_csv('Pretreat_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    Xts.to_csv('Pretreat_test_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    global clf
    clf = estimator

    clf.fit(Xtr,ytr)
    #clf.fit(Xtr,ytr)
    clf.score(Xtr,ytr)
    #cvv=cv(Xtr,ytr,clf.best_estimator_,cvm)
    #print(clf.best_estimator_,clf.best_score_)
    #print(cvv.fit())
    #ts=tsp(Xts,yts,clf.best_estimator_)
    #print(ts.fit())
    filer = open(str(c_)+rn+"_tr.txt","w")
    filer.write('The estimator is: '+'\n')
    filer.write(str(clf))
    filer.write("\n")
    filer.write((str(cvm)+' fold cross validation statistics are: '+'\n'))
    filer.write("\n")
    writefile2(Xtr,ytr,clf,cvm,filer)
    filer.write("\n")
    filer.write('The test set results are: '+'\n')
    filer.write("\n")
    a=writefile(Xts,yts,clf,filer)
    filer.write("\n")
    file2n=file2[file2.iloc[:,0:1].columns.tolist()+file2.iloc[:,2:].columns.tolist()+file2.iloc[:,1:2].columns.tolist()]
    b=pd.concat([file2n,a],axis=1)
    b['Set'] = 'Test'
    b.to_csv(str(c_)+str(rn)+'_tspred.csv',index=False)
    label='test'
    color='blue'

    global pyplot
    pyplot.figure(figsize=(15,10))
    ROCplot(clf,Xts,yts,label,color,pyplot)
    #return clf,filer,pyplot

def sol2():
    #est,file,plt=sol()
    est,rn=Selected()
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    cthreshold=float(thirdEntryTabThreer3c1.get())
    vthreshold=float(fourthEntryTabThreer5c1.get())
    Xtr=pretreat(Xtr,cthreshold,vthreshold)
    Xvd=file3[Xtr.columns]
    Xvd.to_csv('Pretreat_valid_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')

    if ytr.columns[0] in file3.columns:
        yvd=file3.iloc[:,1:2]
        nvd=file3.iloc[:,0:1]
        filer = open(str(c_)+rn+"_vd.txt","w")
        filer.write('The validation set results are: '+'\n')
        filer.write("\n")
        a=writefile(Xvd,yvd,clf,filer)
        file3n=file3[file3.iloc[:,0:1].columns.tolist()+file3.iloc[:,2:].columns.tolist()+file3.iloc[:,1:2].columns.tolist()]
        b=pd.concat([file3n,a],axis=1)
        b['Set'] = 'Validation'
        b.to_csv(str(c_)+str(rn)+'_vdpred.csv', index=False)
        label='validation'
        color='red'
        ROCplot(clf,Xvd,yvd,label,color,pyplot)


    else:
        clf.fit(Xtr,ytr)
        yprvd=pd.DataFrame(clf.predict(Xvd))
        yprvd.columns=['Pred']
        yprvd2=pd.DataFrame(clf.predict_proba(Xvd))
        yprvd2.columns=['%Prob(-1)','%Prob(+1)']
        dfsvd=pd.concat([file3,yprvd,yprvd2],axis=1)
        dfsvd['Diff']=abs(dfsvd['%Prob(-1)']-dfsvd['%Prob(+1)'])
        dfsvd['Outlier_info(Confidence estimation approach, Threshold 0.5)']=['In' if x>=0.5 else 'Outlier' for x in dfsvd['Diff']]
        dfsvd['Set'] = 'Screening'
        dfsvd.to_csv(str(c_)+str(rn)+'_scpred.csv',index=False)


def Selected():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    Xvd=file1.iloc[:,2:]
    yvd=file1.iloc[:,1:2]
    nvd=file1.iloc[:,0:1]
    if var1.get():
       wg=Criterion2.get()
       alg=Criterion3.get()
       rn='KNN'
       estimator=KNeighborsClassifier(n_neighbors=int(E1T1.get()),weights=wg, algorithm=alg)
       #clf1.fit(Xtr,ytr)
       #print(clf1.score(Xtr,ytr))
    elif var2.get():
         FP=Criterion4.get()
         rn='NB'
         estimator=BernoulliNB(alpha=float(E2T1.get()),fit_prior=FP)
         #clf2.fit(Xtr,ytr)
         #print(clf2.score(Xtr,ytr))
    elif var3.get():
         rn='SVC'
         estimator=SVC(C=float(E3T1.get()),gamma=float(E4T1.get()),kernel=Criterion5.get())
         #clf3.fit(Xtr,ytr)
         #print(clf3.score(Xtr,ytr))
    elif var4.get():
         alpha=float(E5T1.get())
         if Criterion26_mr.get()=='none':
            rd=None
         elif Criterion26_mr.get()=='integer':
             rd=int(XML_r.get())
         rn='MLP'
         if int(thirdEntryTabThreer3c1_h.get())!=0:
            hls=(int(thirdEntryTabThreer3c1_h.get()),)
         if int(thirdEntryTabThreer3c1_h1.get())!=0:
            hls=(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()))
         if int(thirdEntryTabThreer3c1_h2.get())!=0:
            hls=(int(thirdEntryTabThreer3c1_h.get()),int(thirdEntryTabThreer3c1_h1.get()),int(thirdEntryTabThreer3c1_h2.get()))
         estimator=MLPClassifier(hidden_layer_sizes=hls,alpha=alpha,activation=Criterion6.get(),solver=Criterion7.get(), learning_rate=Criterion8.get(),random_state=rd)

    elif var5.get():
         rn='RF'
         if Criterion25.get()=='none':
            md=None
         elif Criterion25.get()=='integer':
              md=int(E6T1.get())
         if Criterion25_r.get()=='none':
            rd=None
         elif Criterion25_r.get()=='integer':
             rd=int(E6T1_r.get())
         min_samples_leaf=int(E7T1.get())
         min_samples_split=int(E8T1.get())
         n_estimators=int(E9T1.get())

         estimator=RandomForestClassifier(bootstrap=Criterion11.get(), criterion=Criterion9.get(),max_features=Criterion10.get(),
                                     max_depth=md,min_samples_leaf=int(E7T1.get()),
                                     min_samples_split=int(E8T1.get()),n_estimators=int(E9T1.get()),random_state=rd)
         #clf5.fit(Xtr,ytr)
         #print(clf5.score(Xtr,ytr))

    elif var6.get():
         lr=float(E14T1.get())
         mss=float(E12T1.get())
         msl=float(E11T1.get())
         if Criterion26.get()=='none':
            md=None
         elif Criterion26.get()=='integer':
              md=int(E10T1.get())
         if Criterion26_r.get()=='none':
            rd=None
         elif Criterion26_r.get()=='integer':
             rd=int(E10T1_r.get())
         ss=float(E15T1.get())
         ne=int(E13T1.get())
         rn='GB'
         estimator=GradientBoostingClassifier(loss=Criterion23.get(),learning_rate=lr,min_samples_split=mss, min_samples_leaf=msl,
                                        max_depth=md, max_features=Criterion24.get(), criterion=Criterion22.get(), subsample=ss,
                                         n_estimators=ne,random_state=rd)
         #clf6.fit(Xtr,ytr)
         #print(clf6.score(Xtr,ytr))
    else:
        pass
    print(estimator)
    return estimator,rn

def ROCplot(model,X,y,label,color,plt):
    #model.fit(X,y)
    s,t=Selected()
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
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


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
    #filer.write('Recall score: '+str(recall_score(self.y,ypred))
    #filerw.write('MCC: '+str(a9)+"\n")
    #filerw.write('ROC_AUC: '+str(a10)+"\n")

def enable():
    E10T1['state']='normal'

def disable():
    E10T1['state']='disabled'

def enable1():
    E6T1['state']='normal'

def disable1():
    E6T1['state']='disabled'

form = tk.Tk()
form.title("QSAR-Co-X (Module-3)")
form.geometry("750x700")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)


tab_parent.add(tab1,text="User specific non-linear model")

###Tab1#####

####TAB2#####
firstLabelTabThree = tk.Label(tab1, text="Select sub-training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=60,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab1,text='Browse', command=data1,font=("Helvetica", 10))
b3.place(x=480,y=10)

secondLabelTabThree = tk.Label(tab1, text="Select test set",font=("Helvetica", 12))
secondLabelTabThree.place(x=120,y=40)
secondEntryTabThree = tk.Entry(tab1,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab1,text='Browse', command=data2,font=("Helvetica", 10))
b4.place(x=480,y=40)

thirdLabelTabThreer2c1=Label(tab1, text='Correlation cut-off',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=45,y=70)
thirdEntryTabThreer3c1=Entry(tab1, width=7)
thirdEntryTabThreer3c1.place(x=180,y=73)


fourthLabelTabThreer4c1=Label(tab1, text='Variance cut-off',font=("Helvetica", 12))
fourthLabelTabThreer4c1.place(x=250,y=70)
fourthEntryTabThreer5c1=Entry(tab1, width=7)
fourthEntryTabThreer5c1.place(x=370,y=73)

forthLabelTabOne=tk.Label(tab1, text="CV(model predictability)",font=("Helvetica", 12))
forthLabelTabOne.place(x=450,y=70)
forthEntryTabOne = tk.Entry(tab1, width=7)
forthEntryTabOne.place(x=630,y=73)

var1= IntVar()
CB1=Checkbutton(tab1, text = "k-Nearest Neighborhood",  variable=var1, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB1.place(x=30,y=100)

L1T1= tk.Label(tab1, text="Number of neighbors",font=("Helvetica", 10))
L1T1.place(x=25,y=130)
E1T1 = tk.Entry(tab1, width=10)
E1T1.place(x=153,y=133)

Criterion2 = StringVar()
Criterion3 = StringVar()

L2T1= tk.Label(tab1, text="Weights",font=("Helvetica", 10))
L2T1.place(x=25,y=160)
C1= ttk.Radiobutton(tab1, text='Uniform', variable=Criterion2, value='uniform')
C2= ttk.Radiobutton(tab1, text='Distance', variable=Criterion2,  value='distance')
C1.place(x=80,y=160)
C2.place(x=160,y=160)

L3T1= tk.Label(tab1, text="Algorithm",font=("Helvetica", 10))
L3T1.place(x=20,y=190)
C3= ttk.Radiobutton(tab1, text='Auto', variable=Criterion3, value='auto')
C4= ttk.Radiobutton(tab1, text='Ball_tree', variable=Criterion3, value='ball_tree')
C5= ttk.Radiobutton(tab1, text='KD_tree', variable=Criterion3, value='kd_tree')
C6= ttk.Radiobutton(tab1, text='Brute', variable=Criterion3, value='brute')

C3.place(x=80,y=190)
C4.place(x=130,y=190)
C5.place(x=80,y=210)
C6.place(x=150,y=210)

b5=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b5.place(x=100,y=235)

#############################

var2= IntVar()
CB2=Checkbutton(tab1, text = "Bernoulli Naive Bayes",  variable=var2, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB2.place(x=520,y=100)

L4T1= tk.Label(tab1, text="alpha",font=("Helvetica", 10))
L4T1.place(x=535,y=130)
E2T1 = tk.Entry(tab1, width=10)
E2T1.place(x=573,y=133)

Criterion4 = BooleanVar()
#Criterion4.set(True)
L5T1= tk.Label(tab1, text="Fit_prior",font=("Helvetica", 10))
L5T1.place(x=510,y=160)
C7= ttk.Radiobutton(tab1, text='True', variable=Criterion4, value=True)
C8= ttk.Radiobutton(tab1, text='False', variable=Criterion4,  value=False)
C7.place(x=570,y=160)
C8.place(x=630,y=160)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=570,y=190)

########################################

var3= IntVar()
CB2=Checkbutton(tab1, text = "Support Vector Classification",  variable=var3, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB2.place(x=250,y=100)

L6T1= tk.Label(tab1, text="C",font=("Helvetica", 10))
L6T1.place(x=250,y=130)
E3T1 = tk.Entry(tab1, width=10)
E3T1.place(x=273,y=133)

L7T1= tk.Label(tab1, text="Gamma",font=("Helvetica", 10))
L7T1.place(x=350,y=130)
E4T1 = tk.Entry(tab1, width=10)
E4T1.place(x=403,y=133)

Criterion5 = StringVar()

L8T1= tk.Label(tab1, text="Kernel",font=("Helvetica", 10))
L8T1.place(x=250,y=160)
C9= ttk.Radiobutton(tab1, text='Linear', variable=Criterion5, value='linear')
C10= ttk.Radiobutton(tab1, text='RBF', variable=Criterion5, value='rbf')
C9_x= ttk.Radiobutton(tab1, text='Poly', variable=Criterion5, value='poly')
C10_x= ttk.Radiobutton(tab1, text='Sigmoid', variable=Criterion5, value='sigmoid')
C9.place(x=300,y=160)
C10.place(x=370,y=160)
C9_x.place(x=300,y=180)
C10_x.place(x=370,y=180)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=325,y=205)

#################

var4= IntVar()
CB3=Checkbutton(tab1, text = "Multilayer Perception",  variable=var4, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=30,y=300)

L9T1= tk.Label(tab1, text="alpha",font=("Helvetica", 10))
L9T1.place(x=80,y=330)
E5T1 = tk.Entry(tab1, width=10)
E5T1.place(x=120,y=333)

thirdLabelTabThreer2c1_h=Label(tab1, text='Hidden Layers',font=("Helvetica", 10))
v1=IntVar(tab1,value=100)
v2=IntVar(tab1,value=0)
v3=IntVar(tab1,value=0)
thirdLabelTabThreer2c1_h.place(x=10,y=360)
thirdEntryTabThreer3c1_h=Entry(tab1,textvariable=v1,width=5)
thirdEntryTabThreer3c1_h.place(x=110,y=363)
thirdEntryTabThreer3c1_h1=Entry(tab1,textvariable=v2,width=5)
thirdEntryTabThreer3c1_h1.place(x=150,y=363)
thirdEntryTabThreer3c1_h2=Entry(tab1,textvariable=v3,width=5)
thirdEntryTabThreer3c1_h2.place(x=190, y=363)

Criterion6 = StringVar()
Criterion7 = StringVar()
Criterion8 = StringVar()


L8T1= tk.Label(tab1, text="Activation",font=("Helvetica", 10))
L8T1.place(x=30,y=390)
C11= ttk.Radiobutton(tab1, text='Identity', variable=Criterion6, value='identity')
C12= ttk.Radiobutton(tab1, text='Logistic', variable=Criterion6, value='logistic')
C13= ttk.Radiobutton(tab1, text='Tanh', variable=Criterion6, value='tanh')
C14= ttk.Radiobutton(tab1, text='Relu', variable=Criterion6, value='relu')

C11.place(x=90,y=390)
C12.place(x=160,y=390)
C13.place(x=90,y=410)
C14.place(x=160,y=410)

L9T1= tk.Label(tab1, text="Solver",font=("Helvetica", 10))
L9T1.place(x=30,y=435)
C15= ttk.Radiobutton(tab1, text='SGD', variable=Criterion7, value='sgd')
C16= ttk.Radiobutton(tab1, text='Adam', variable=Criterion7, value='adam')
C15.place(x=90,y=435)
C16.place(x=160,y=435)

L10T1= tk.Label(tab1, text="Learning rate",font=("Helvetica", 10))
L10T1.place(x=10,y=460)
C17= ttk.Radiobutton(tab1, text='Constant', variable=Criterion8, value='constant')
C18= ttk.Radiobutton(tab1, text='Adaptive', variable=Criterion8, value='adaptive')
C19= ttk.Radiobutton(tab1, text='Invscaling', variable=Criterion8, value='invscaling')

C17.place(x=90,y=460)
C18.place(x=160,y=460)
C19.place(x=90,y=480)

XL_r= tk.Label(tab1, text="Random state",font=("Helvetica", 10))
XL_r.place(x=10,y=505)
Criterion26_mr=StringVar()
Criterion26_mr.set('none')
XL_None_r = ttk.Radiobutton(tab1, text='None', variable=Criterion26_mr, value='none')
XL_None_r.place(x=100,y=505)
XL_integer_r = ttk.Radiobutton(tab1, text='Number', variable=Criterion26_mr, value='integer')
XL_integer_r.place(x=155,y=505)
XML_r = tk.Entry(tab1, textvariable=XL_integer_r, width=7)
XML_r.place(x=225,y=505)

b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=120,y=530)


###################
var5= IntVar()
CB3=Checkbutton(tab1, text = "Random Forest",  variable=var5, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=280,y=240)

L11T1= tk.Label(tab1, text="Max depth",font=("Helvetica", 10))
L11T1.place(x=240,y=270)
Criterion25=StringVar()
MaxFeatures_None = ttk.Radiobutton(tab1, text='None', variable=Criterion25, value='none')
MaxFeatures_None.place(x=310,y=273)
MaxFeatures_integer = ttk.Radiobutton(tab1, text='Number', variable=Criterion25, value='integer')
MaxFeatures_integer.place(x=370,y=273)
E6T1 = tk.Entry(tab1,textvariable=MaxFeatures_integer, width=7)
E6T1.place(x=440,y=273)
Criterion25.set('none')


L11T1_r= tk.Label(tab1, text="Random state",font=("Helvetica", 10))
L11T1_r.place(x=220,y=295)
Criterion25_r=StringVar()
MaxFeatures_None_r = ttk.Radiobutton(tab1, text='None', variable=Criterion25_r, value='none')
MaxFeatures_None_r.place(x=310,y=298)
MaxFeatures_integer_r = ttk.Radiobutton(tab1, text='Number', variable=Criterion25_r, value='integer')
MaxFeatures_integer_r.place(x=370,y=298)
E6T1_r = tk.Entry(tab1,textvariable=MaxFeatures_integer_r, width=7)
E6T1_r.place(x=440,y=298)
Criterion25_r.set('none')

L12T1= tk.Label(tab1, text="Minimum samples leaf",font=("Helvetica", 10))
L12T1.place(x=260,y=325)
E7T1 = tk.Entry(tab1, width=10)
E7T1.place(x=400,y=328)


L13T1= tk.Label(tab1, text="Minimum samples split",font=("Helvetica", 10))
L13T1.place(x=260,y=350)
E8T1 = tk.Entry(tab1, width=10)
E8T1.place(x=400,y=353)

L14T1= tk.Label(tab1, text="Number of Estimators",font=("Helvetica", 10))
L14T1.place(x=260,y=375)
E9T1 = tk.Entry(tab1, width=10)
E9T1.place(x=400,y=378)


Criterion9 = StringVar()
Criterion10 = StringVar()
Criterion11 = BooleanVar()


L15T1= tk.Label(tab1, text="Criterion",font=("Helvetica", 10))
L15T1.place(x=260,y=400)
C20= ttk.Radiobutton(tab1, text='Gini', variable=Criterion9, value='gini')
C21= ttk.Radiobutton(tab1, text='Entropy', variable=Criterion9, value='entropy')

C20.place(x=320,y=400)
C21.place(x=370,y=400)


L9T1= tk.Label(tab1, text="Max features",font=("Helvetica", 10))
L9T1.place(x=240,y=425)
C15= ttk.Radiobutton(tab1, text='Auto', variable=Criterion10, value='auto')
C16= ttk.Radiobutton(tab1, text='SQRT', variable=Criterion10, value='sqrt')
C16_1= ttk.Radiobutton(tab1, text='Log2', variable=Criterion10, value='log2')
C15.place(x=320,y=425)
C16.place(x=370,y=425)
C16_1.place(x=420,y=425)

L10T1= tk.Label(tab1, text="Bootstrap",font=("Helvetica", 10))
L10T1.place(x=280,y=450)
C17= ttk.Radiobutton(tab1, text='True', variable=Criterion11, value=True)
C18= ttk.Radiobutton(tab1, text='False', variable=Criterion11, value=False)

C17.place(x=350,y=450)
C18.place(x=400,y=450)


b6=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b6.place(x=330,y=475)

###################

var6= IntVar()
CB3=Checkbutton(tab1, text = "Gradient Boosting",  variable=var6, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
CB3.place(x=530,y=240)


L12T1= tk.Label(tab1, text="Minimum samples leaf",font=("Helvetica", 10))
L12T1.place(x=510,y=270)
E11T1 = tk.Entry(tab1, width=10)
E11T1.place(x=650,y=273)


L13T1= tk.Label(tab1, text="Minimum samples split",font=("Helvetica", 10))
L13T1.place(x=510,y=295)
E12T1 = tk.Entry(tab1, width=10)
E12T1.place(x=650,y=298)

L14T1= tk.Label(tab1, text="Number of Estimators",font=("Helvetica", 10))
L14T1.place(x=510,y=320)
E13T1 = tk.Entry(tab1, width=10)
E13T1.place(x=650,y=323)

L14T1= tk.Label(tab1, text="Learning rate",font=("Helvetica", 10))
L14T1.place(x=560,y=350)
E14T1 = tk.Entry(tab1, width=10)
E14T1.place(x=650,y=353)

L14T1= tk.Label(tab1, text="Subsample",font=("Helvetica", 10))
L14T1.place(x=570,y=380)
E15T1 = tk.Entry(tab1, width=10)
E15T1.place(x=650,y=383)

Criterion22 = StringVar()
Criterion23 = StringVar()
Criterion24 = StringVar()


L15T1= tk.Label(tab1, text="Criterion",font=("Helvetica", 10))
L15T1.place(x=500,y=410)
C20= ttk.Radiobutton(tab1, text='Friedman MSE', variable=Criterion22, value='friedman_mse')
C21= ttk.Radiobutton(tab1, text='MAE', variable=Criterion22, value='mae')

C20.place(x=560,y=410)
C21.place(x=660,y=410)


L16T1= tk.Label(tab1, text="Loss",font=("Helvetica", 10))
L16T1.place(x=505,y=435)
C22= ttk.Radiobutton(tab1, text='Deviance', variable=Criterion23, value='deviance')
C23= ttk.Radiobutton(tab1, text='Exponential', variable=Criterion23, value='exponential')
C22.place(x=540,y=435)
C23.place(x=620,y=435)

L17T1= tk.Label(tab1, text="Max features",font=("Helvetica", 10))
L17T1.place(x=480,y=460)
C24= ttk.Radiobutton(tab1, text='Log2', variable=Criterion24, value='log2')
C25= ttk.Radiobutton(tab1, text='SQRT', variable=Criterion24, value='sqrt')
C26= ttk.Radiobutton(tab1, text='Auto', variable=Criterion24, value='auto')

C24.place(x=560,y=460)
C25.place(x=610,y=460)
C26.place(x=660,y=460)

L11T1= tk.Label(tab1, text="Max depth",font=("Helvetica", 10))
L11T1.place(x=480,y=490)
Criterion26=StringVar()
Criterion26.set('none')
MaxFeatures1_None = ttk.Radiobutton(tab1, text='None', variable=Criterion26, value='none')
MaxFeatures1_None.place(x=555,y=490)
MaxFeatures1_integer = ttk.Radiobutton(tab1, text='Number', variable=Criterion26, value='integer')
MaxFeatures1_integer.place(x=615,y=490)
E10T1 = tk.Entry(tab1, textvariable=MaxFeatures1_integer, width=7)
E10T1.place(x=685,y=490)

L11T1_r= tk.Label(tab1, text="Random state",font=("Helvetica", 10))
L11T1_r.place(x=470,y=515)
Criterion26_r=StringVar()
Criterion26_r.set('none')
MaxFeatures1_None_r = ttk.Radiobutton(tab1, text='None', variable=Criterion26_r, value='none')
MaxFeatures1_None_r.place(x=555,y=515)
MaxFeatures1_integer_r = ttk.Radiobutton(tab1, text='Number', variable=Criterion26_r, value='integer')
MaxFeatures1_integer_r.place(x=615,y=515)
E10T1_r = tk.Entry(tab1, textvariable=MaxFeatures1_integer_r, width=7)
E10T1_r.place(x=685,y=515)

b7=Button(tab1, text='Generate', command=sol,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b7.place(x=565,y=545)


fifthLabelTabOne = tk.Label(tab1, text="Select validation set",font=("Helvetica", 12))
fifthLabelTabOne.place(x=60,y=580)
fifthEntryTabOne = tk.Entry(tab1, width=40)
fifthEntryTabOne.place(x=230,y=583)
b3=tk.Button(tab1,text='Browse', command=data3,font=("Helvetica", 10))
b3.place(x=480,y=580)

b4=Button(tab1, text='Submit', command=sol2,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b4.place(x=320,y=610)

tab_parent.pack(expand=1, fill='both')

form.mainloop()
