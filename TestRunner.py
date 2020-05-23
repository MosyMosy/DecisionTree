import numpy as np
import pandas as pd
import os
import ID3
import DTTest
import DTPrune


def RunPrunedTreeTest(data,attDict,validationFrac,testFrac):
    dataRowsCount = len(data.index)
    validSize = int(dataRowsCount*validationFrac)
    testSize = int(dataRowsCount*testFrac)    
    
    validationData = data[0:validSize]
    testData = data[validSize:(validSize + testSize)]
    trainData=data[(validSize + testSize):]
    
    # Train the tree
    tree = ID3.RunID3(attDict,trainData)
    # Important!!! Required for pruning
    tree.InitMostCommonValues()

    prunedTree = DTPrune.RunePruninig(tree,validationData,attDict)

    # Test with test data
    result = DTTest.RunTest(testData,tree)
    firstMetrics = DTTest.PercissionAndRecallAndFMeasure(result,attDict)

    result = DTTest.RunTest(validationData,prunedTree)
    prundMetricsValidation = DTTest.PercissionAndRecallAndFMeasure(result,attDict)

    result = DTTest.RunTest(testData,prunedTree)
    prundMetricsTest = DTTest.PercissionAndRecallAndFMeasure(result,attDict)

    return [firstMetrics,prundMetricsValidation,prundMetricsTest]


def RunTrainTest(data,attDict,fraction):
    # Splite Train and Test by 66% ratio
    trainData=data.sample(frac=fraction,random_state=200) #random state is a seed value
    testData=data.drop(trainData.index)

    # Train the tree
    tree = ID3.RunID3(attDict,trainData)
    # Test with test data
    result = DTTest.RunTest(testData,tree)

    return DTTest.PercissionAndRecallAndFMeasure(result,attDict)

def RunKfold(data,attDict,k):
    # K-Fold average metrics
    finalPercission = 0
    finalRecall = 0
    finalFMeasure = 0
    kfoldData = DataToKFoldList(data,k)

    for i in range(k):
        # Splite Train and Test by k ratio
        testData=kfoldData[i]
        remainingFolds = [x for j,x in enumerate(kfoldData) if j!=1]
        trainData=pd.concat(remainingFolds,ignore_index=True)

        # Train the tree
        tree = ID3.RunID3(attDict,trainData)

        # Test with test data
        result = DTTest.RunTest(testData,tree)

        metrics = DTTest.PercissionAndRecallAndFMeasure(result,attDict)
        finalPercission += metrics[0]
        finalRecall += metrics[1]
        finalFMeasure += metrics[2]
    
    return round((finalPercission/k),3),round((finalRecall/k),3),round((finalFMeasure/k),3)


def DataToKFoldList(data,k):
    kfoldData = []
    dataRowsCount = len(data.index)
    foldSize = int(dataRowsCount/k)
    lastIndex =0
    for i in range(k-1):
        kfoldData.append(data[(i*foldSize):((i+1)*foldSize)])
        lastIndex = (i+1)*foldSize
    kfoldData.append(data[lastIndex:])

    return kfoldData




