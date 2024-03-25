import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
#import pymysql
import os
import shutil
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
import itertools


warnings.simplefilter(action='ignore')

#xls = pd.ExcelFile('Best_three_models.xlsx')



#import db_config
initialdir=os.getcwd()

#initialdir='C:\\Users\\Amit\\Downloads\\pyqsar_tutorial-master\\Best_model'
#initialdir='C:\\Users\\USER\\Downloads\\twitter_classification_project'
def data():
    filename = askopenfilename(initialdir=initialdir,title = "Select file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename)
    global a_
    a_,b_=os.path.splitext(filename)
    global file
    file = pd.ExcelFile(filename)

def iter(stuff):
    ils=[]
    for L in range(0, len(stuff)+1):
        for subset in itertools.combinations(stuff, L):
            if len(list(subset))>1:
               ils.append(list(subset))
    return ils


def process(xls,ls):
    ls1=[]
    for i in ls:
        tri=pd.read_excel(xls,sheet_name=i)
        tri=tri.sort_values(list(tri)[0], ascending=True).reset_index().drop('index',axis=1)
        nd=tri.columns.get_loc('Pred')
        nd=tri.iloc[:,nd-1:nd]
        tri=tri.rename(columns={nd.columns[0]:nd.columns[0]+'_'+str(i)})
        tri=tri.rename(columns={'Set':'Set_'+str(i)})
        ls1.append(tri)
        md=pd.concat(ls1,axis=1)
    return md,i


def selection(df):
    dx=df[['Pred','%Prob(+1)','%Prob(-1)']]
    dx['mark']=dx[['%Prob(+1)','%Prob(-1)']].idxmax(axis=1)
    dx['check']=dx.apply(lambda x: 1 if x['mark']=='%Prob(+1)' else -1, axis=1)
    dx['Consensus_Pred']=dx.apply(lambda x: x['Pred'].mode() if abs(x['Pred'].sum())>0 else x['check'], axis=1)
    fd=pd.concat([df,dx[['mark','check','Consensus_Pred']]],axis=1)
    cp=list(fd.columns).index('Pred')
    acc=accuracy_score(fd[fd.iloc[:,cp-1:cp].columns[0]],fd['Consensus_Pred'])
    mcc=matthews_corrcoef(fd[fd.iloc[:,cp-1:cp].columns[0]],fd['Consensus_Pred'])
    return acc,mcc,fd

def final( ):
    ls=file.sheet_names
    ils=iter(ls)
    m1,m2,m3,m4=[],[],[],[]
    m5,m6,m7=[],[],[]
    for i in ils:
        md,x=process(file,i)
        acc,mcc,fd=selection(md)
        #print(acc)
        #fd.to_csv(str(a_)+"_consensus2.csv", index=False)
        #fd_str=fd[fd['Set_'+str(x)]=='Sub_train']
        fd_ts=fd[fd['Set_'+str(x)]=='Test']
        fd_vd=fd[fd['Set_'+str(x)]=='Validation']
        cp=list(fd.columns).index('Pred')
        #mcc_tr=matthews_corrcoef(fd_str[fd.iloc[:,cp-1:cp].columns[0]],fd_str['Consensus_Pred'])
        mcc_ts=matthews_corrcoef(fd_ts[fd.iloc[:,cp-1:cp].columns[0]],fd_ts['Consensus_Pred'])
        mcc_vd=matthews_corrcoef(fd_vd[fd.iloc[:,cp-1:cp].columns[0]],fd_vd['Consensus_Pred'])
        #acc_tr=accuracy_score(fd_str[fd.iloc[:,cp-1:cp].columns[0]],fd_str['Consensus_Pred'])
        acc_ts=accuracy_score(fd_ts[fd.iloc[:,cp-1:cp].columns[0]],fd_ts['Consensus_Pred'])
        acc_vd=accuracy_score(fd_vd[fd.iloc[:,cp-1:cp].columns[0]],fd_vd['Consensus_Pred'])
        m1.append(i)
        #m2.append(round(acc_tr,3))
        #m3.append(round(mcc_tr,3))
        m4.append(round(acc_ts,3))
        m5.append(round(mcc_ts,3))
        m6.append(round(acc_vd,3))
        m7.append(round(mcc_vd,3))
    #print(m1)
        Dict=dict([('Combination', m1),('Acc_ts', m4),
              ('Mcc_ts',m5),('Acc_vd', m6),('Mcc_vd', m7)])
        dd=pd.DataFrame(Dict)
    dd.to_csv(str(a_)+'_consensus_Table.csv', index=False)

form = tk.Tk()

form.title("QSAR-Co-X (Module-4)")

form.geometry("670x100")

tab_parent = ttk.Notebook(form)

tab1 = tk.Frame(tab_parent) #background='#ffffff')

tab_parent.add(tab1, text="Consensus modeling")


###Tab1#####

firstLabelTabThree = tk.Label(tab1, text="Select excel file",font=("Helvetica", 12))
firstLabelTabThree.place(x=60,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab1,text='Browse', command=data,font=("Helvetica", 10))
b3.place(x=480,y=10)

b4=Button(tab1, text='Submit', command=final,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b4.place(x=320,y=45)


tab_parent.pack(expand=1, fill='both')

form.mainloop()
