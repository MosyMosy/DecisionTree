import numpy as np
import pandas as pd
import os
import ID3
import DTTest
import TestRunner

datasetFolderNames = ["1-Balance Scale Database", "2-Balloons Database",
                      "3-Wisconsin Breast Cancer Databases", "4-Credit Screening Databases",
                      "5-Cylinder Bands Database", "6-Dermatology Database", "8-Adult Database",
                      "9-Abalone Database", "10-Satimage"]

# datasetFolderNames = ["1-Balance Scale Database"]
# Global Dataset folder, curent data folder name, Constants description and data pathes
_DataSetPath = os.path.join(os.path.dirname(os.path.abspath(
    __file__)), 'Data-Sets')   # Get the directory for StringFunctions

# run for all datasets
for datasetName in datasetFolderNames:
    print(datasetName)
    _curentdatapath = os.path.join(_DataSetPath, datasetName)
    _descriptionPath = os.path.join(_curentdatapath, "description.csv")
    _dataPath = os.path.join(_curentdatapath, "data.csv")

    # read description, it contain fiels named and field values. include clas column
    description = pd.read_csv(_descriptionPath, dtype=str)
    # create dictionary of fields and its values as a list
    attDict = dict(zip(list(description), [_vlues.split(
        ",") for _vlues in (description.values.tolist()[0])]))
    # load data and assign columns from description (nodelist)
    data = pd.read_csv(_dataPath, names=list(attDict.keys()), dtype=str)


    data = data.sample(frac=1).reset_index(drop=True)

    PrunedMetrics = TestRunner.RunPrunedTreeTest(data, attDict, 0.2, 0.2)
      
    trainTestMetrics = TestRunner.RunTrainTest(data, attDict, 0.66)
    Fold3Metrics = TestRunner.RunKfold(data, attDict, 3)
    Fold5Metrics = TestRunner.RunKfold(data, attDict, 5)

    with open(os.path.join(_curentdatapath, "Results.txt"), "w") as text_file:
        print("trainTestMetrics: {0},{1},{2}".format(
            *trainTestMetrics), file=text_file)
        print("Fold3Metrics: {0},{1},{2}".format(*Fold3Metrics), file=text_file)
        print("Fold5Metrics: {0},{1},{2}".format(*Fold5Metrics), file=text_file)
        
        print("BeforPruneMetrics: {0},{1},{2}, ".format(*PrunedMetrics[0]) +
         "PruneValidationMetrics: {0},{1},{2}, ".format(*PrunedMetrics[1]) +
         "PruneTestMetrics: {0},{1},{2}".format(*PrunedMetrics[2]), file=text_file)


    print("trainTestMetrics: {0},{1},{2}".format(*trainTestMetrics))
    print("Fold3Metrics: {0},{1},{2}".format(*Fold3Metrics))
    print("Fold5Metrics: {0},{1},{2}".format(*Fold5Metrics))

    print("BeforPruneMetrics: {0},{1},{2}, ".format(*PrunedMetrics[0]) +
         "PruneValidationMetrics: {0},{1},{2}, ".format(*PrunedMetrics[1]) +
         "PruneTestMetrics: {0},{1},{2}".format(*PrunedMetrics[2]))



"""
# Handle missing vales
data = data.replace('?', np.NaN)
data = data.apply(lambda x:x.fillna(x.value_counts().index[0]))

# for decritize the data
for i in range(len(attDict)):
    addName = list(attDict.keys())[i]
    if(list(attDict.values())[i][0] == 'continuous'):
        data[addName] = pd.cut(data[addName], bins=10, labels=np.arange(10), right=False)
data.to_csv(os.path.join(_curentdatapath, "data-d.csv"), index = False,header=False)"""

