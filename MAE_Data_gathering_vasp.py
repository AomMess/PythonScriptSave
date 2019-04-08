import numpy as np
import os
import pandas as pd
import re
command="find ../ -iname 'OSZICAR' -print0 | xargs -0 grep  E0= > Data_grep.dat"
os.system(command)
f=open("Data_grep.dat",'r')
Data=f.read()
f.close()
for char in ":=":
    Data=Data.replace(char,',')
Data=Data.split('\n')
print(len(Data))
Src_string=["RELAX","Relax","relax","BAND","Bulk","/SOC-SCF/","/SCF/","/WS2/","DOS"]
indices=[]
for strg in Src_string:
    for line in Data:
        if line.find(strg)!=-1:
            indices.append(Data.index(line))
indices=set(indices)
clean_list= [i for j, i in enumerate(Data) if j not in indices]
f=open('Data_file.dat','w')
for line in clean_list:
    f.write(line+'\n')
f.close()

Data=pd.read_csv('Data_file.dat', header=None)


Data = Data.drop([1,2,4,5], axis=1)

Data[3] = Data[3].map(lambda x: str(x)[:-4])

Data.columns=['Path', 'Total Energy']
Data.to_csv('Recap_Data.dat',sep='|')
