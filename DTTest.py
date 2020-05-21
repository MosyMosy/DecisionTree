import DTree


def TreeTraverse(tree: DTree.DTree, sample):
    # get coresponding value of the provided sample based on current tree root attribute name
    sampleAttValue = sample[tree.Name]

    if type(tree.Branches[sampleAttValue]) == DTree.DTree:
        return TreeTraverse(tree.Branches[sampleAttValue], sample)
    else:
        return tree.Branches[sampleAttValue]


def RunTest(testData, tree):
    predictions = []
    for i in range(len(testData.index)):
        predictions.append(TreeTraverse(tree, testData.iloc[i]))

    resultDF = testData.filter(["Target"], axis=1)
    resultDF["Prediction"] = predictions
    resultDF.eval("Match = Prediction == Target", inplace=True)
    return resultDF


def PercissionAndRecallAndFMeasure(testResult, attDict):
    percisiion = 0
    recall = 0
    FMeasure = 0
    classCount = len(attDict["Target"])

    for className in attDict["Target"]:        
        # All Positive
        classPredicted = testResult.loc[testResult["Prediction"] == className]
        classTotalPredictedCount = len(classPredicted.index)
        # True Positive
        classTruePredictedCount = len(
            classPredicted.loc[classPredicted["Match"] == True].index)

        # All Negative
        classNegativePredicted = testResult.loc[testResult["Prediction"] != className]
        classTotalFalsePredictedCount = len(classNegativePredicted.index)
        # False Negative
        classFalseNegativePredictedCount = len(
            classNegativePredicted.loc[classNegativePredicted["Match"] == False].index)

        if (classTotalPredictedCount) != 0:
            percisiion += (classTruePredictedCount/classTotalPredictedCount)/classCount

        if (classTruePredictedCount+classFalseNegativePredictedCount) != 0:
            recall += (classTruePredictedCount /
                       (classTruePredictedCount+classFalseNegativePredictedCount))/classCount

    if (percisiion != None) and (recall != None):
        if (percisiion + recall) != 0:
            FMeasure += (2*percisiion*recall)/(percisiion + recall)
    
    return round(percisiion, 3), round(recall, 3), round(FMeasure, 3)

