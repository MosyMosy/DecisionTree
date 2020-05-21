import uuid

# Main Decisiontree Node. This represent an attribute in data
# Name: Name of the attribute
# values: values of this attribute 
# Branches: dictionary of value (index) and childs 
# !!!After training, Childs could get either a subtree or a target lable!!!
class DTree(object):
    def __init__(self,name,values,parentTree):
        # a unic identifier
        self.UID = uuid.uuid4().hex
        
        # Name of the attribute
        self.Name = name

        # Dictionary of (value,child) pairs. childs are none at the begining
        self.Branches = dict(zip(values, [None] * len(values)))

        # common value through its chilsd. this would be initialized latter
        self.MostCommonValue = None
    

    def InitMostCommonValues(self):
        bramchValues = []
        for branchKey in self.Branches:
            if type(self.Branches[branchKey]) == DTree:
                bramchValues.append(self.Branches[branchKey].InitMostCommonValues())
            else:
                bramchValues.append(self.Branches[branchKey])
        
        self.MostCommonValue = max(set(bramchValues), key = bramchValues.count) 
        return self.MostCommonValue
