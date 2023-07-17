import numpy as np
import random
def DataGetter(filepath):
    DataFile=open(filepath,'r')
    Label=[]#Values
    Features=[]#Features/Data
    DataLines=DataFile.readlines()
    random.shuffle(DataLines)
    for line in DataLines:
        Data=line.split(',')
        Label.append(Data[0])
        Data.remove(Data[0])
        Features.append(Data)
    DataFile.close()
    return Label,Features

def DataManager(filepath):
    Label,DataFeatures=DataGetter(filepath)
    Label.remove(Label[0])
    DataFeatures.remove(DataFeatures[0])
    Label=list(map(float,Label))
    Features=[]
    for i in DataFeatures:
        Features.append(list(map(float,i)))
    #print(Features)
    Label=np.asarray(Label)
    Features=np.asarray(Features)  
    return Label,Features

print(DataManager('data/AQI_Pakistan.csv'))