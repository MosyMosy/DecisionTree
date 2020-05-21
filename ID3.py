import pandas as pd
import math
import DTree


def RunID3(attributes: dict, data: pd.DataFrame, ParentTree=None):    
    # for the Root tree, Parent is none

    if len(data.index) == 0:
        return None 
    
    rootAttrCandidateName = GetBestAttByInfoGain(attributes,data)[0]
    NewRoot = DTree.DTree(rootAttrCandidateName,attributes[rootAttrCandidateName],ParentTree); 

    for attValue in attributes[rootAttrCandidateName]:
        
        data_i = data.loc[data[rootAttrCandidateName] == attValue]
        if Entropy(data_i) == 0:            
            if len(data_i.index) == 0:
                # for reserving generality, we hold all attribute
                NewRoot.Branches[attValue] = data.Target.mode()[0] #reversed(list(GetTargetValuesCount(data).values()).sort())[1]
            else:
                # If all examples are negative or positive, Return the single-node tree Branch, with label = thisClass.
                NewRoot.Branches[attValue] = data_i["Target"].iloc[0]
        else:
            newAttributes = RemoveDictItembyKey(attributes,rootAttrCandidateName)
            # If number of predicting attributes is empty, then Return the single node tree Root,
            # with label = most common value of the target attribute in the examples.
            if len(newAttributes) == 1:
                valueCounts = list(GetTargetValuesCount(data).values())
                valueCounts.sort()
                NewRoot.Branches[attValue] = valueCounts[-2]
            else:
                # curent NewRoot will be it's childs parent
                NewRoot.Branches[attValue] = RunID3(newAttributes,data_i,NewRoot)            
    
    return NewRoot
            
    

def Entropy(data: pd.DataFrame):
    entropy = 0
    targetValues = GetTargetValuesCount(data)
    for i in range(len(targetValues) -1):
        portion = list(targetValues.values())[i]/targetValues.get("Total")
        entropy -=  portion* math.log2(portion)
    return entropy

def InformationGain(data: pd.DataFrame,attrName,attrValues):
    mainEntropy = Entropy(data) 
    dataSize = len(data.index)
    _sum = 0
    for value in attrValues:
        data_i = data.loc[data[attrName] == value]        
        _sum += (len(data_i.index)/dataSize) * Entropy(data_i)
    
    return (mainEntropy - _sum)

def GetBestAttByInfoGain(attributes: dict, data: pd.DataFrame):
    biggestInfoGain = 0
    bestAttName = ""
    for att in attributes.items():
        if att[0] != "Target":
            attGain = InformationGain(data,att[0],att[1])
            if attGain > biggestInfoGain:
                biggestInfoGain = attGain
                bestAttName = att[0]
    # if info gains are equal select the first attribute
    if (biggestInfoGain == 0) and (list(attributes)[0] != "Target"):
        biggestInfoGain = attGain
        bestAttName = list(attributes)[0]
    return bestAttName,biggestInfoGain


def GetTargetValuesCount(data: pd.DataFrame):
    return GetAttValuesCount(data,'Target')

def GetAttValuesCount(data: pd.DataFrame,attName):
    dataTargetCount = data.groupby([attName]).count()
    targetLables = list((dataTargetCount).index)
    targetValues = list((dataTargetCount).values[:, 0])
    
    TargetValuesDict = dict(zip(targetLables,targetValues))
    TargetValuesDict.update({'Total': len(data.index)})

    return TargetValuesDict

def RemoveDictItembyKey(d, key):
    r = dict(d)
    del r[key]
    return r
