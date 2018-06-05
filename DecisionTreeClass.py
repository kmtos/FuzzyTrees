import string
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab
import sys
import pandas as pd
import math
import random
from operator import itemgetter

class DecisionTree(object):
  '''
  A class that creates a decision Tree that uses the Gini Index (as Opposed to the Entropy information gain) as the decider
  for where to split on which branches. This is the base class for future boosted versions, and can be used with the 
  Fuzzy Tree later. 
  Terminology:
    -nodes: each grouping of dataframe entries is considered a node. 'Leaves' are considered the nodes that have no daughter nodes.
    -BlankNodes: nodes that have no entries or come from Nodes that have entries, but do not pass Gini Index improvement requirements.
    -EndNodes: are nodes where every entrie at that node have the same class. The decision is the unanimous.
  '''

  def __init__(self, idColumn, className, nodeDFIDsFileName, nodeValuesFileName, nodeDecisionsFileName, outputFileName, 
               maxDepth=4, nGiniSplits=10, giniEndVal=0.01, nClasses=2, minSamplesSplit=2, printOutput=True):
    self.idColumn   = idColumn	  # Column that indexes dataframe
    self.className  = className   # Column that will be classified
    self.nodeDFIDsFileName     = nodeDFIDsFileName     # Name of the file that the Datafram ID's for each node will be written to
    self.nodeValuesFileName    = nodeValuesFileName    # Name of the file describing the cuts at every node
    self.nodeDecisionsFileName = nodeDecisionsFileName # Name of the file describing the everything about the end nodes (Nodes with no further decisions)
    self.outputFileName = outputFileName + ".csv" # Name of the output classification on a provided test set
    self.maxDepth    = maxDepth    # The maximum depth of the decision tree
    self.nGiniSplits = nGiniSplits # Number of splits along the range of values to test the optimum splitting of a column for the next branch
    self.giniEndVal  = giniEndVal  # The minimum improvement needed to cut on a column for further branching
    self.nClasses    = nClasses    # The number of possibilities in classification
    self.minSamplesSplit = minSamplesSplit # Minimum number of samples required to be present at a leaf in the tree
    self.printOutput = printOutput # Whether to print all the output of the Decision tree
    self.maxNodes    = 0           # The maximum number of nodes based upon the max number of maximum depth
    for i in range(1,self.maxDepth+1): self.maxNodes += 2**i
    self.nodeDFIDs     = []  # List of tuples that keeps track of the IDs of each entry in the df at each node
    self.nodeValues    = []  # List of tuples that keeps track of the cut values, the column cut on, and the gini improvement at each node
    self.nodeDecisions = []  # list the decision for the GT and LT group at each leaf with no further daughters. Also gives the correct amounts in each decision
    self.testDFIDs     = []  # List of the test df ID's at each node
   
  def FindingBestSplit(self, df, df_weights, nodeCount):
    '''
    This finds which column at which value moves the 'Gini Index' closest to 0 (perfect classification).
    It first looks to see if all the data points in current node have the same class. If so, it is now an end node, because the Gini index is 0.
    'columns' are all the columns that are used for identification and not for decision making (ID column, Class column, etc.)
    If there are less unique values in the column than the number of splits desired to be used (nGiniSplits), then just use the unique values for the class.
    'CalcBestGiniSplit' finds the best split for the current column in the for loop. I then see if it is better than the best split on the run over columns so far.
    At the end of looping through the columns, check that it is greater than 'self.giniEndVal'. If not, return a BlankNode.
    The output is formatted like so: (node number, the amount that the Gini index decreased, column name, the value of column split at, length of the splits)
    '''
    for classVal, rows in df[self.className].value_counts().to_dict().items(): 
      if len(df) == rows:
        if self.printOutput: print ("*****************END NODE")
        return ( nodeCount, 1.0, 'ThisIsAnEndNode', np.NaN, np.NaN)
    columns = [col for col in df if col != self.className and col != self.idColumn] 
    bestGiniSplit = (-1, -1, '', -100, -1)
    for col in columns:
      unique = df[col].unique() 
      high = df[col].max() 
      low = df[col].min() 
      splitLength = (high - low) / (nGiniSplits+1)
      splits = []
      if len(unique) == 1: continue
      elif len(unique) <= nGiniSplits: 
	unique = sorted(unique) 
        for i in range(len(unique)-1):
          splits.append( (unique[i] + unique[i+1])/2.0)
      else: 
        for i in range(1,nGiniSplits+1):
          splits.append(low + (i * splitLength) )
      bestSplitForCol = CalcBestGiniSplit(df, col, splits, df_weights) 
      if bestSplitForCol[0] > bestGiniSplit[1]: 
        bestGiniSplit = (nodeCount, bestSplitForCol[0], col, bestSplitForCol[1], splitLength )
    if bestGiniSplit[1] < giniEndVal:
      iself.printOutputprint ("Less that Min gini improvement: bestGiniSplit[1]=", bestGiniSplit[1], "\tginiEndVal=", giniEndVal)
      bestGiniSplit = (nodeCount, np.NaN, '', np.NaN, np.NaN) 
    return bestGiniSplit
 
  def CalcBestGiniSplit(self, df, colName, splits, df_weights):
    '''
    Given a column, the dataframe entries at the current node, and the possible splits, find the best value to split on.
    Before calculating split, make sure that each leaf after the split, has more than the 'minSamplesSplit'. 
    If not, then sum the weights of the entries with values greater than (GT) the cut and less than (LT) the cut for each split.
    Gini index calculationo explained here: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
    Of the splits, find the one with the lowest 'newGiniIndex'. Return the split and the difference between the old and new gini index.
    '''
    uniqueClasses = df[self.className].unique()
    bestGiniDecrease = (-100, -1)
    counts = []
    IDs = df[self.idColumn].tolist()
    weightSum = df_weights.loc[ df_weights[self.idColumn].isin(IDs)]["Weights"].sum(axis=0)
    giniSum = 0
    for cls in uniqueClasses:
      sumWeights = df_weights.loc[(df_weights[self.idColumn].isin(IDs)) & (df_weights[self.className]==cls)]["Weights"].sum(axis=0)
      giniSum += (sumWeights * sumWeights / weightSum / weightSum)
    currentGiniIndex = 1 - giniSum
    for split in splits: 
      if len(df[df[colName]>split]) < self.minSamplesSplit or len(df[df[colName]<=split]) < self.minSamplesSplit: 
        continue
      GTIDs=df[ df[colName]>split][self.idColumn].tolist()
      LTIDs=df[ df[colName]<=split][self.idColumn].tolist()
      weightSumGT = df_weights.loc[ df_weights[self.idColumn].isin(GTIDs)]["Weights"].sum(axis=0)
      weightSumLT = df_weights.loc[ df_weights[self.idColumn].isin(LTIDs)]["Weights"].sum(axis=0)
      giniSumGT = 0
      giniSumLT = 0
      for cls in uniqueClasses:
        sumWeightsGT = df_weights.loc[(df_weights[self.idColumn].isin(GTIDs)) & (df_weights[self.className]==cls)]["Weights"].sum(axis=0)
        sumWeightsLT = df_weights.loc[(df_weights[self.idColumn].isin(LTIDs)) & (df_weights[self.className]==cls)]["Weights"].sum(axis=0)
        giniSumGT += (sumWeightsGT * sumWeightsGT / weightSumGT / weightSumGT )
        giniSumLT += (sumWeightsLT * sumWeightsLT / weightSumLT / weightSumLT )
      counts.append( ( giniSumLT, weightSumLT, split, giniSumGT, weightSumGT) )
    for tup in counts: 
      newGiniIndex = (1 - tup[0])*tup[1]/(tup[1]+tup[4]) + (1 - tup[3])*tup[4]/(tup[1]+tup[4])
      if newGiniIndex < bestGiniDecrease[0] and newGiniIndex < currentGiniIndex:
        bestGiniDecrease = (currentGiniIndex-newGiniIndex, tup[2])
    return bestGiniDecrease
  
  def BuildTree(self, df, df_weights):
    '''
    This makes the tree and decides on the cuts to use at each split. 'self.nodeValues' and 'self.nodeDFIDs' keep track of the nodes.
    'parentTup' is a tuple of the parent node, grabbed from the current node number. Nodes are numbered left to right starting at 0. 
      - 0->1,2 | 1->3,4 | 2->5,6 | 3->7,8 | 4->9,10, with the lower number (i.e. 9) having the LT group and the larger value (i.e 10) having the GT group
    If parent node is a BlankNode or an EndNode, then make two BlankNode children.
    Depending on the nodeCount value, 'dfCurr' grabs the entries that are LT or GT the cut values specified in 'parentTup'.
    If there are not enough entries to split, based on the self.minSamplesSplit, it's children are BlankNodes. Otherwise, get the best split.
    '''
    if self.printOutput: print ("\n\n###################################\n Making a Decision Tree\n###################################")
    nodeCount = 0
    while nodeCount <= maxNodes: 
      if nodeCount == 0: 
        self.nodeValues.append( FindingBestSplit(df=df, df_weights=df_weights, nodeCount=nodeCount) )
        self.nodeDFIDs.append( (nodeCount, df[self.idColumn].tolist()) )
      else:
        parentTup = self.nodeValues[(nodeCount-1) // 2] 
        parentDFIDs = self.nodeDFIDs[(nodeCount-1) // 2][1] 
        if self.printOutput: print ("\nnode=", nodeCount, "parentNode=", (nodeCount-1) // 2, "\tparentTup=", parentTup, "\tlen(parentDFIDs)=", len(parentDFIDs))
        if pd.isnull(parentTup[3]) and pd.isnull(parentTup[4]): 
          self.nodeValues.append( (nodeCount, np.NaN, '' , np.NaN, np.NaN) )
          self.nodeDFIDs.append( (nodeCount, [] ) )
        else: 
          if nodeCount % 2  == 1: dfCurr = df.loc[(df[self.idColumn].isin(parentDFIDs)) & (df[parentTup[2]] <= parentTup[3]) ] 
          else: dfCurr = df.loc[(df[self.idColumn].isin(parentDFIDs)) & (df[parentTup[2]] > parentTup[3]) ] 
          if self.printOutput: print ("len(dfCurr)=", len(dfCurr) )
          if len(dfCurr) < self.minSamplesSplit:
            if self.printOutput: print ("Too Few to be able to split")
            nodeValues.append( (nodeCount, np.NaN, '', np.NaN, np.NaN) )
            nodeDFIds.append( (nodeCount, [] ) )
          else:
            self.nodeValues.append(FindingBestSplit(df=dfCurr, df_weights=df_weights, nodeCount=nodeCount) ) 
            self.nodeDFIDs.append( (nodeCount, dfCurr[self.idColumn].tolist()) ) 
            if self.printOutput: print ("######## NEW ########:", "nodeValues[nodeCount]=", self.nodeValues[nodeCount], "\tlen(nodeDFIds[", nodeCount, "][1])=", len(self.nodeDFIDs[nodeCount][1]))
            if not pd.isnull(self.nodeValues[nodeCount][3]) and self.printOutput: 
              print ("len(lessThan)=", len(dfCurr.loc[dfCurr[self.nodeValues[nodeCount][2]] <= self.nodeValues[nodeCount][3]]), 
                     "\tlen(greaterThan)=", len(dfCurr.loc[dfCurr[self.nodeValues[nodeCount][2]] > self.nodeValues[nodeCount][3]]) )
      nodeCount += 1
  
       
  def WritingNodeValues(self):
    '''
    This writes out to a .csv file the tuples that contain the nodes with their cut information and the decrease in Gini Index.
    '''
    if self.printOutput: print ("\n\n###########################\nWriting Out the Nodes Values\n###########################")
    nodeValuesFile = open(self.nodeValuesFileName + "csv", 'w')
    nodeValuesFileCSV=csv.writer(nodeValuesFile)
    nodeValuesFileCSV.writerow(["NodeNumber,GiniIncrease,ColumnName,ValueOfSplit,RangeBetweenSplits"])
    for tup in self.nodeValues:
      nodeValuesFileCSV.writerow(tup)

  def WritingNodeDFIDs(self):
    '''
    This writes out to a .csv file the node numbers and the datafram IDs that are present in each node.
    '''
    if self.printOutput: print ("\n\n###########################\nWriting Out the Nodes DF IDs\n###########################")
    nodeDFIdsFile = open(self.nodeDFIDsFileName + ".csv", 'w')
    nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
    nodeDFIdsFileCSV.writerow(["NodeNumber,ListOfID'sAtNode"])
    for tup in self.nodeDFIDs:
      nodeDFIdsFileCSV.writerow(tup)
 
  def GetWeightedNodeDecisions(df, leaf, index, soloNodeDecision, GTorLT, df_weights):
    '''
    This gets the decision for a give node. It first checks if the node is an EndNode (Only happens when all the entries in the node have the same class.
    Then it checks if the node is a solo node, which means it has only 1 daughter. This means only the LT or GT was split further. A decision is needed
    for the other non-split group. Note that the 'soloNodeDecision' uses 'ltMaxWeight' and 'ltMaxClassVal' variables, whether the node is a GT or a LT.
    To get a decision, it loops through the unique class values and finds the class that has the most weight present in the current node. 
    '''
    ltMaxWeight = -100000000000000000000000000000000
    ltMaxClassVal = -100000000000000000000000000000000
    gtMaxWeight = -100000000000000000000000000000000
    gtMaxClassVal = -100000000000000000000000000000000
    ltTotalWeight = -1
    gtTotalWeight = -1
    if leaf[1] == 1.0 and leaf[2] == 'ThisIsAnEndNode' and pd.isnull(leaf[3]) and pd.isnull(leaf[4]): # See if this is a node where every element is the same class
      if self.printOutput: print("This is An End Node")
      return (index, df[self.className].unique()[0], np.NaN, df_weights['Weights'].sum(axis=0), np.NaN, df_weights['Weights'].sum(axis=0), np.NaN)
  
    if (soloNodeDecision):
      df_IDs = df[df[leaf[2]]>leaf[3]][self.idColumn].tolist() if GTorLT > 0 else df[df[leaf[2]]<=leaf[3]][self.idColumn].tolist()
      totalWeight = df_weights[ df_weights[self.idColumn].isin(df_IDs) ]['Weights'].sum(axis=0)
      for classVal, row in  df[df[self.idColumn].isin(df_IDs)][self.className].value_counts().to_dict().items():
        currWeight = df_weights[ (df_weights[self.idColumn].isin(df_IDs)) & (df_weights[self.className] == classVal) ]['Weights'].sum(axis=0)
        if self.printOutput: print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
        if currWeight > ltMaxWeight:
          ltMaxWeight = currWeight
          ltMaxClassVal = classVal
      if GTorLT > 0:
        if self.printOutput: print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. GT Node Decision.")
        return (index, np.NaN, ltMaxClassVal, np.NaN, ltMaxWeight, np.NaN, totalWeight)
      else:
        if self.printOutput: print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. LT Node Decision.")
        return (index, ltMaxClassVal, np.NaN, ltMaxWeight, np.NaN, totalWeight, np.NaN)
  
    if self.printOutput: print ("\tlen(ltDF)=", len(df[ df[leaf[2]]<=leaf[3] ]), "\tlen(gtDF)=", len(df[ df[leaf[2]]>leaf[3] ]) )
    if self.printOutput: print ("\tLESS THAN") 
    df_ltIDs = df[ df[leaf[2]]<=leaf[3] ][self.idColumn].tolist()
    df_gtIDs = df[ df[leaf[2]]> leaf[3] ][self.idColumn].tolist()
    ltTotalWeight = df_weights[ df_weights[self.idColumn].isin(df_ltIDs) ]['Weights'].sum(axis=0)
    gtTotalWeight = df_weights[ df_weights[self.idColumn].isin(df_gtIDs) ]['Weights'].sum(axis=0)
    for classVal, row in  df[ df[self.idColumn].isin(df_ltIDs)  ][self.className].value_counts().to_dict().items():
      currWeight = df_weights[ (df_weights[self.idColumn].isin(df_ltIDs)) & (df_weights[self.className] == classVal) ]['Weights'].sum(axis=0)
      if self.printOutput: print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
      if currWeight > ltMaxWeight:
        ltMaxWeight = currWeight
        ltMaxClassVal = classVal
    if self.printOutput: print("\tGREATER THAN") 
    for classVal, row in  df[ df[self.idColumn].isin(df_gtIDs) ][self.className].value_counts().to_dict().items(): 
      currWeight = df_weights[ (df_weights[self.idColumn].isin(df_gtIDs)) & (df_weights[self.className] == classVal) ]['Weights'].sum(axis=0)
      if self.printOutput: print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
      if currWeight > gtMaxWeight:
        gtMaxWeight = currWeight
        gtMaxClassVal = classVal
    if self.printOutput: print ("\t", (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight) )
    return (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight)
 
 
  def GetDecisions(self, df, df_weights):
    '''
    This gets the decisions at each node, after the tree is fully explored using the stop requirements in the tree's initialization.
    It looks for the first non-BlankNode going from the bottom of the tree upwards.
    Three scenarios: 
      1) Node is an End node and it does a vote weighted by 'df_weights' and democratically decides the class of the entries at the node.
      2) Node and it's sister are BlankNodes, so grab the nearest parent that is an EndNode and add decision to tuple of decisions, if that node isn't already added.
      3) Node is a Blank Node and Sister is Not. That means one parent is a needs a decision on only the GT or the LT split. Call this a half-EndNode
    '''
    self.nodeDecisions = []
    minNodeNotLeaf = self.maxNodes
    if self.printOutput: print ("df_weights['Weights'].sum(axis=0)=", df_weights['Weights'].sum(axis=0) )
    for ite in range(self.maxNodes, self.maxNodes - 2**self.maxDepth, -1):
      index = ite
      currentLeaf = self.nodeValues[index]
      currentDF = df.loc[df[self.idColumn].isin(self.nodeDFIDs[index][1])]
      currentDF_weights = df_weights.loc[ df[self.idColumn].isin(self.nodeDFIDs[index][1])]
      soloNode = False
      GTorLT = 0
      while pd.isnull(currentLeaf[1] ) and  currentLeaf[2] == '' and pd.isnull(currentLeaf[3] ) and pd.isnull(currentLeaf[4] ): 
        sisterNode = index-1 if index % 2 == 0 else index+1
        sisterLeaf = self.nodeValues[sisterNode]
        GTorLT = 1 if index % 2 == 0 else -1
        soloNode = False
        if not pd.isnull(sisterLeaf[1] ) and sisterLeaf[2] != '' and not pd.isnull(sisterLeaf[3] ) and not pd.isnull(sisterLeaf[4] ): soloNode = True
        index = (index-1) // 2
        currentLeaf = self.nodeValues[index]
        currentDF = df.loc[df[self.idColumn].isin(self.nodeDFIDs[index][1])]
        currentDF_weights = df_weights.loc[ df[self.idColumn].isin(self.nodeDFIDs[index][1])]
      if self.printOutput: print ("\n\nindex=", index, "\tcurrentLeaf=", currentLeaf, "\tlen(currentDF)=", len(currentDF) )
      currentNodeDecision = (GetWeightedNodeDecisions(df=currentDF, leaf=currentLeaf, index=index, soloNodeDecision=soloNode, GTorLT=GTorLT, df_weights=currentDF_weights) )
      try: 
        next (tup for tup in self.nodeDecisions if tup[0] == index)
        print ("Decision already included from other daugther node")
      except StopIteration:  
        self.nodeDecisions.append(currentNodeDecision )
  
  def WriteDecisions(self, df, df_weights):
    '''
    Write the decisions gathered in "GetDecisions".
    '''
    #Write out the self.nodeDecisions
    nodeDecisionsFile = open(self.nodeDecisionsFileName + ".csv", 'w')
    nodeDecisionsFileCSV=csv.writer(nodeDecisionsFile)
    nodeDecisionsFileCSV.writerow(["NodeNumber,LT_className_decision,GT_className_decision,LT_WeightCorrect,GT_WeightCorrect,LT_TotalWeight,GT_TotalWeight"])
    for tup in self.nodeDecisions:
      nodeDecisionsFileCSV.writerow(tup)
  
  
 
  def ClassifyWithTree(self, df_test):
    '''
    This classifies test points, whether validation or a test set, and gives the predicted answer from the tree. First it iterates throught the nodes from the tree and stores
    each dataframe entry along the tree in the proper node. 'self.testDFIDs' stores what dataframe IDs are at each node. 'dfCurr' has the dataframe entries at the current node.
    First it checks if the node is a BlankNode, and if it is, then it creates empty node placeholders in 'self.testDFIDs', if it isn't at the maxdepth yet. Then it checks if it is 
    an EndNode, and if it is, then it sets all test entry answers accordingly and creates empty node placeholders for deeper nodes from it. Then it checks if the node is at 
    the max tree depth. If it is, it then splits the entries at the current node into the GT and LT nodes, and checks if the node has a decision in any of its nodes. Finally,
    if it is at max tree depth, then it does have a decision in the decision file, and it is applied to all test set entries in at each node.
    '''
    if self.printOutput: print ("\n\n####################################\n Classifying test points with Tree from Build Tree\n#############################################")
    df_Answers = df_test.filter([self.idColumn], axis=1)
    df_Answers[self.className] = np.nan 
  
    self.testDFIDs = [ (0, df_test[self.idColumn].tolist()) ]
    for ite in self.nodeValues: 
      nodeValueTup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4]))
      if self.printOutput: print ("\n\tnodeValueTup=", nodeValueTup, "\tnodeValueTup[0]=", nodeValueTup[0], "\tself.testDFIDs[nodeValueTup[0]][0])=", self.testDFIDs[nodeValueTup[0]][0])
      dfCurr = df_test.loc[df_test[self.idColumn].isin(self.testDFIDs[nodeValueTup[0]][1])] 
  
      if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]): 
        self.Classify_BlankNode(nodeValueTup)

      elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: 
        self.Classify_EndNode(df=dfCurr, df_Answers=df_Answers, nodeValueTup=nodeValueTup)

      elif nodeValueTup[0] < self.maxNodes / 2: 
        self.Classify_NotMaxDepth(df=dfCurr, df_Answers=df_Answers, nodeValueTup=nodeValueTup)

      else: 
        self.Classify_MaxDepth(df=dfCurr, df_Answers=df_Answers, nodeValueTup=nodeValueTup)

      del dfCurr 

    if self.printOutput: print ("df_Answers=", df_Answers.head(10) )
    df_Answers.to_csv(outputFileName + ".csv", sep=',', index=False) 

  def Classify_BlankNode(self, nodeValueTup=nodeValueTup):
    '''
    If node is a BlankNode, then add BlankNodes for daughters.
    '''
    if self.printOutput: print ("\tdf=", self.testDFIDs[nodeValueTup[0]][1])
    if nodeValueTup[0] < self.maxNodes / 2: 
      self.testDFIDs.append( (nodeValueTup[0]*2 + 1, [] ) )
      self.testDFIDs.append( (nodeValueTup[0]*2 + 2, [] ) )

  def Classify_EndNode(self, df, df_Answers, nodeValueTup):
    '''
    If node is an EndNode, then classify points there and add BlankNode for daughters, if current node is not at the max tree depth
    '''
    decision = next(iteTup for iteTup in self.nodeDecisions if int(iteTup[0]) == nodeValueTup[0])
    IDs = dfCurr[self.idColumn].tolist() 
    df_Answers.loc[ df_Answers[self.idColumn].isin(IDs) , self.className] = decision[1] 
    if self.printOutput: print ("Class for End-Node=",  decision[1], "\tlen(IDs)=", IDs)
    if nodeValueTup[0] < self.maxNodes / 2: 
      self.testDFIDs.append( (nodeValueTup[0]*2 + 1, [] ) )
      self.testDFIDs.append( (nodeValueTup[0]*2 + 2, [] ) )

  def Classify_NotMaxDepth(self, df, df_Answers, nodeValueTup):
    '''
    If node is not at the max tree depth, then add the split df ID's to self.testDFIDs. Then, check if there
    is a decision for the LT, GT, or both groups.
    '''
    if self.printOutput: print ("\tlen(lt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ]) )
    self.testDFIDs.append( (nodeValueTup[0]*2 + 1, dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist() ) )
    self.testDFIDs.append( (nodeValueTup[0]*2 + 2, dfCurr[ dfCurr[nodeValueTup[2]] >  nodeValueTup[3] ][self.idColumn].tolist() ) )
    try:  
      decision = next (itetup for itetup in self.nodeDecisions if int(itetup[0]) == nodeValueTup[0])
      if self.printOutput: print ("\tOne of this Node's Daughters is a BlankNode.")

      if pd.isnull(float(decision[2]) ) and not pd.isnull(float(decision[1]) ):
        ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist()  
        df_Answers.loc[ df_Answers[self.idColumn].isin(ltIDs) , self.className] = decision[1] 

      elif not pd.isnull(float(decision[2]) ) and pd.isnull(float(decision[1]) ):
        gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][self.idColumn].tolist() 
        df_Answers.loc[ df_Answers[self.idColumn].isin(gtIDs) , self.className] = decision[2] 

      else:
        ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist()
        gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][self.idColumn].tolist() 
        df_Answers.loc[ df_Answers[self.idColumn].isin(ltIDs) , self.className] = decision[1]
        df_Answers.loc[ df_Answers[self.idColumn].isin(gtIDs) , self.className] = decision[2]
        if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
    except StopIteration:  
      if self.printOutput: print ("Non of this node's daughters are Blank Nodes")

  def Classify_MaxDepth(self, df, df_Answers, nodeValueTup):
    '''
     Given the values at the node at the max depth of the tree, Classify the entries there of the df test set. 
    '''
    decision = next(iteTup for iteTup in self.nodeDecisions if int(iteTup[0]) == nodeValueTup[0])
    ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist() 
    gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] >  nodeValueTup[3] ][self.idColumn].tolist() 
    df_Answers.loc[ df_Answers[self.idColumn].isin(ltIDs) , self.className] = decision[1] 
    df_Answers.loc[ df_Answers[self.idColumn].isin(gtIDs) , self.className] = decision[2] 
    if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )


  def CleanTree(self):
    '''
    If you would like to keep the class, but reset the lists of node decisions, ID's, and values.
    '''
    self.nodeDFIDs     = []  
    self.nodeValues    = []  
    self.nodeDecisions = []  
     
