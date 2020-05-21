import DTree
import DTTest
import pandas as pd
import copy
import random


def RunePruninig(tree:DTree.DTree, validationData:pd.DataFrame,attDict:dict):
    chekedIdList = []
    testResult = DTTest.RunTest(validationData,tree)
    
    bestMetric = DTTest.PercissionAndRecallAndFMeasure(testResult,attDict) 
    bestTree = copy.deepcopy(tree)
    while True:        
        treeNodIds = GetTreeNodesIDList(bestTree)
        if treeNodIds is None:
            treeNodIds = []
        # remove cheked items
        treeNodIds = [x for x in treeNodIds if x not in chekedIdList]
        if len(treeNodIds) != 0:
            # get last item because its near to leaf
            toPruneUid = treeNodIds[random.randint(0, len(treeNodIds)-1)]
            PrunedTree = copy.deepcopy(bestTree)
            # Prune it
            PruneSubTreeById(PrunedTree,toPruneUid)

            prunetestResult = DTTest.RunTest(validationData,PrunedTree)
            prunedMetric = DTTest.PercissionAndRecallAndFMeasure(prunetestResult,attDict)

            if prunedMetric[2] > bestMetric[2]:
                bestMetric = prunedMetric
                bestTree = PrunedTree

            chekedIdList.append(toPruneUid)
        else:
            break;

            
    return bestTree

def Prune(tree:DTree.DTree):
    for branchKey in tree.Branches:
            if type(tree.Branches[branchKey]) == DTree.DTree:
                mostCommonValue = (tree.Branches[branchKey]).MostCommonValue
                tree.Branches[branchKey] = mostCommonValue
    
    return tree

def PruneSubTreeById(tree:DTree.DTree,uid:int):
  
    if tree.UID == uid:
        tree = Prune(tree)
    else:
        for branchKey in tree.Branches:
            if type(tree.Branches[branchKey]) == DTree.DTree:
                tree.Branches[branchKey] = PruneSubTreeById(tree.Branches[branchKey],uid) 
    return tree
                

# this return a list of id of nodes that has tree child
def GetTreeNodesIDList(tree:DTree.DTree):
    hasTreeChild = False
    idList = [tree.UID]

    for branchKey in tree.Branches:
            if type(tree.Branches[branchKey]) == DTree.DTree:
                hasTreeChild = True
                thisTreeIds = GetTreeNodesIDList(tree.Branches[branchKey])
                if thisTreeIds is not None:
                    idList = idList + thisTreeIds
    
    if hasTreeChild:
        return idList
    else:
        return None
            
