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

#######################################################
# Given a df, this finds the column and value of
# The best split for the next two leaves based on gini
# tup = (node #, amount of gini increase, the column name, 
#        the value split at, range between splits)
#######################################################
def FindingBestSplit(df, className, idColumn, nGiniSplits, nodeCount, giniEndVal, minSamplesSplit, df_weights):
  for classVal, rows in df[className].value_counts().to_dict().items(): # If the number of data points in the leaf are all of the same class, then the node ends
    if len(df) == rows: 
      print ("*****************END NODE")
      return ( nodeCount, 1.0, 'ThisIsAnEndNode', np.NaN, np.NaN)
  columns = [col for col in df if col != className and col != idColumn and col != "Memberships" and col != "MembershipNodeList"] # Get all columns but the class and ID column or Memb col for fuzzy
  bestGiniSplit = (-1, -1, '', -100, -1)
  for col in columns:
    useUnique = False
    unique = sorted(df[col].unique() )
    high = unique[len(unique)-1]
    low =  unique[0]
    splitLength = (high - low) / (nGiniSplits+1)
    splits = []
    if len(unique) == 1: continue
    elif len(unique) <= nGiniSplits: #if the number of unique values is less than the desired number of splits, then just make the unique values the splits
      for i in range(len(unique)-1):
        splits.append( (unique[i] + unique[i+1])/2.0)
    else: #Find the number of splits for the current column
      for i in range(1,nGiniSplits+1):
        splits.append(low + (i * splitLength) )
    bestSplitForCol = CalcBestGiniSplit(df, className, col, splits, minSamplesSplit, idColumn, df_weights) #Find the best split for this column
    if bestSplitForCol[0] > bestGiniSplit[1]: #See if this column provides the best Gini increase so far
      bestGiniSplit = (nodeCount, bestSplitForCol[0], col, bestSplitForCol[1], splitLength )
  if bestGiniSplit[1] < giniEndVal: 
    print ("Less that Min gini improvement: bestGiniSplit[1]=", bestGiniSplit[1], "\tginiEndVal=", giniEndVal)
    bestGiniSplit = (nodeCount, np.NaN, '', np.NaN, np.NaN) # Returns BlankNode if the best possible gini increase is less than the minimum for ending
  return bestGiniSplit
    
#############################################################
# given a column an it's splits, find the best gini increase
# tup = (amount of gini increase,the value split at)
#############################################################
def CalcBestGiniSplit(df, className, colName, splits, minSamplesSplit, idColumn, df_weights):
  uniqueClasses = df[className].unique()
  bestGiniIncrease = (-100, -1)
  counts = []
  for split in splits: # Make list of the groups of elements for each split to later calculate the best gini increase
    if len(df[df[colName]>split]) < minSamplesSplit or len(df[df[colName]<=split]) < minSamplesSplit: # Checks that leaf has at least the minimum # of data points
      continue
    GTIDs=df[ df[colName]>split][idColumn].tolist()
    LTIDs=df[ df[colName]<=split][idColumn].tolist()
    counts.append( ( df_weights.loc[(df_weights[idColumn].isin(GTIDs)) & (df_weights[className]==uniqueClasses[0])]["Weights"].sum(axis=0), # Now done with Weights
                     df_weights.loc[ df_weights[idColumn].isin(GTIDs)]["Weights"].sum(axis=0),  split,
                     df_weights.loc[(df_weights[idColumn].isin(LTIDs)) & (df_weights[className]==uniqueClasses[0])]["Weights"].sum(axis=0),  
                     df_weights.loc[ df_weights[idColumn].isin(LTIDs)]["Weights"].sum(axis=0) ) )
    # Below is the gini split without weights. Recently added with weights
    #counts.append( (len(df[ (df[className]==uniqueClasses[0]) & (df[colName]>split) ]), len(df[df[colName]>split]), split,
    #                len(df[ (df[className]==uniqueClasses[0]) & (df[colName]<=split) ]), len(df[df[colName]<=split])) )
  for tup in counts:  #Finds the best gini increase for the splits
    giniIncreaseGT = tup[0]/tup[1] * tup[0]/tup[1] +  (1 - tup[0]/tup[1]) * (1 - tup[0]/tup[1])
    giniIncreaseLT = tup[3]/tup[4] * tup[3]/tup[4] +  (1 - tup[3]/tup[4]) * (1 - tup[3]/tup[4])
    giniIncrease = giniIncreaseGT*tup[1]/(tup[1]+tup[4]) + giniIncreaseLT*tup[4]/(tup[1]+tup[4])
    if giniIncrease > bestGiniIncrease[0]: # If the current increase is bigger than the previously best increase, then reassign to new best gini increase
      bestGiniIncrease = (giniIncrease, tup[2])
  return bestGiniIncrease
  
##################################################################
# Makes a Decision Tree that saves output in three forms:
#    1) The tuple found with FindingBestSplit for each node
#    2) The node number and the DF ID's at that leaf
#    3) The Decision that was made at each leaf
##################################################################
def MakeTreeOld(df, className, nGiniSplits, giniEndVal, maxDepth, idColumn, minSamplesSplit, df_weights, nodeDFIDsFileName, nodeValuesFileName, nodeDecisionsFileName):
  print ("\n\n###################################\n Making a Decision Tree\n###################################")
  maxNodes = 0
  for i in range(1,maxDepth+1): maxNodes += 2**i
  nodeValues = [] #Node number,  amount of gini increase, the column name, the value split at, range between splits)
  nodeDFIds = [] #The point ID's that are in each leaf
  nodeCount = 0
  while nodeCount <= maxNodes: #checking that I haven't met the max node numb set by maxDepth
    if nodeCount == 0: #for the trunk or first tree node
      nodeValues.append(FindingBestSplit(df=df, className=className, idColumn=idColumn, nGiniSplits=nGiniSplits, nodeCount=nodeCount, giniEndVal=giniEndVal, 
                                         minSamplesSplit=minSamplesSplit, df_weights=df_weights) ) 
      nodeDFIds.append( (nodeCount, df[idColumn].tolist()) )
    else:
      parentTup = nodeValues[(nodeCount-1) // 2] #getting parent nodeValues tuple
      parentDFIDs = nodeDFIds[(nodeCount-1) // 2][1] #getting parent dataframe row ID's
      print ("\nnode=", nodeCount, "parentNode=", (nodeCount-1) // 2, "\tparentTup=", parentTup, "\tlen(parentDFIDs)=", len(parentDFIDs))
      if pd.isnull(parentTup[3]) and pd.isnull(parentTup[4]): # Make BlankNodes for leaves whose parents are End nodes or other BlankNodes
        nodeValues.append( (nodeCount, np.NaN, '' , np.NaN, np.NaN) )
        nodeDFIds.append( (nodeCount, [] ) )
      else: # Create new node with the best gini increase and the df ID's and other important  information
        if nodeCount % 2  == 1: dfCurr = df.loc[(df[idColumn].isin(parentDFIDs)) & (df[parentTup[2]] <= parentTup[3]) ] #Getting dataframe elements that are lower than the parent split
        else: dfCurr = df.loc[(df[idColumn].isin(parentDFIDs)) & (df[parentTup[2]] > parentTup[3]) ] #getting dataframe elements that are greater than or equal to the parent split
        print ("len(dfCurr)=", len(dfCurr) )
        if len(dfCurr) < minSamplesSplit:
          print ("Too Few to be able to split")
          nodeValues.append( (nodeCount, np.NaN, '', np.NaN, np.NaN) )
          nodeDFIds.append( (nodeCount, [] ) )
        else: 
          nodeValues.append(FindingBestSplit(df=dfCurr, className=className, idColumn=idColumn, nGiniSplits=nGiniSplits, nodeCount=nodeCount, giniEndVal=giniEndVal, 
                                           minSamplesSplit=minSamplesSplit, df_weights=df_weights) ) # get next best split node Values
          nodeDFIds.append( (nodeCount, dfCurr[idColumn].tolist()) ) # Get next best split df ID's
          print ("######## NEW ########:", "nodeValues[nodeCount]=", nodeValues[nodeCount], "\tlen(nodeDFIds[", nodeCount, "][1])=", len(nodeDFIds[nodeCount][1]))
          if not pd.isnull(nodeValues[nodeCount][3]): print ("len(lessThan)=", len(dfCurr.loc[dfCurr[nodeValues[nodeCount][2]] <= nodeValues[nodeCount][3]]), "\tlen(greaterThan)=", len(dfCurr.loc[dfCurr[nodeValues[nodeCount][2]] > nodeValues[nodeCount][3]]) ) 
    nodeCount += 1

  #Writing out the tuples for the nodes and cuts on which columns and dataframe ID's in each of the leaves
  print ("\n\n###########################\nWriting Out the Nodes, Values, and df ID's of the nodes\n###########################")
  nodeValuesFileName = nodeValuesFileName + ".csv"
  nodeValuesFile = open(nodeValuesFileName, 'w')
  nodeValuesFileCSV=csv.writer(nodeValuesFile)
  nodeValuesFileCSV.writerow(["NodeNumber,GiniIncrease,ColumnName,ValueOfSplit,RangeBetweenSplits"])
  for tup in nodeValues:
    nodeValuesFileCSV.writerow(tup)
  nodeDFIDsFileName = nodeDFIDsFileName + ".csv"
  nodeDFIdsFile = open(nodeDFIDsFileName, 'w')
  nodeDFIdsFileCSV=csv.writer(nodeDFIdsFile)
  nodeDFIdsFileCSV.writerow(["NodeNumber,ListOfID'sAtNode"])
  for tup in nodeDFIds:
    nodeDFIdsFileCSV.writerow(tup)
  
  #Getting the first non-dead leaf, i.e. leaf who's parent has a gini increase greater than the minimum for a leaf to end
  nodeDecisions = []
  minNodeNotLeaf = maxNodes
  print ("df_weights['Weights'].sum(axis=0)=", df_weights['Weights'].sum(axis=0) )
  for ite in range(maxNodes, maxNodes - 2**maxDepth, -1):
    index = ite 
    currentLeaf = nodeValues[index]
    currentDF = df.loc[df[idColumn].isin(nodeDFIds[index][1])]
    currentDF_weights = df_weights.loc[ df[idColumn].isin(nodeDFIds[index][1])]
    soloNode = False
    gt_or_lt = 0
    while pd.isnull(currentLeaf[1]) and  currentLeaf[2] == '' and pd.isnull(currentLeaf[3]) and pd.isnull(currentLeaf[4]): # Check if node is a BlankNode and if so get it's parent until non-blank
      sisterNode = index-1 if index % 2 == 0 else index+1
      gt_or_lt = 1 if index % 2 == 0 else -1
      soloNode = False
      if not pd.isnull(nodeValues[sisterNode][1]) and nodeValues[sisterNode][2] != '' and not pd.isnull(nodeValues[sisterNode][3]) and not pd.isnull(nodeValues[sisterNode][4]): soloNode = True
      index = (index-1) // 2
      currentLeaf = nodeValues[index]
      currentDF = df.loc[df[idColumn].isin(nodeDFIds[index][1])]
      currentDF_weights = df_weights.loc[ df[idColumn].isin(nodeDFIds[index][1])]
    print ("\n\nindex=", index, "\tcurrentLeaf=", currentLeaf, "\tlen(currentDF)=", len(currentDF) )
    currentNodeDecision = (GetWeightedNodeDecisions(df=currentDF, leaf=currentLeaf, index=index, className=className, soloNodeDecision=soloNode, gt_or_lt=gt_or_lt,
                                                    df_weights=currentDF_weights, idColumn=idColumn) ) # Get the decision of the node
    try:  # This sees if the decision of a node is already added. From a sister BlankNode
      next (tup for tup in nodeDecisions if tup[0] == index)
      print ("Decision already included from other daugther node")
    except StopIteration:  #If node is not found in nodeDecisions, then add it
      nodeDecisions.append(currentNodeDecision )

  #Write out the nodeDecisions
  nodeDecisionsFileName =  nodeDecisionsFileName + ".csv"
  nodeDecisionsFile = open(nodeDecisionsFileName, 'w')
  nodeDecisionsFileCSV=csv.writer(nodeDecisionsFile)
  nodeDecisionsFileCSV.writerow(["NodeNumber,LT_className_decision,GT_className_decision,LT_WeightCorrect,GT_WeightCorrect,LT_TotalWeight,GT_TotalWeight"])
  for tup in nodeDecisions: 
    nodeDecisionsFileCSV.writerow(tup)

########################################################################
# With a finished  tree, get the decisions at each final
# node. Ouput will be a tuple = (Node number, the decision for the
# group that is less than the final cut, the decision for the 
# group greater than the last cut of the final node, number of 
# correctly labeled in lt group, number of correctly labeled in gt group
#########################################################################
def GetNodeDecisions(df, leaf, index, className): 
  ltMaxCount = -100000000000000000000000000000000
  ltMaxClassVal = -100000000000000000000000000000000
  gtMaxCount = -100000000000000000000000000000000
  gtMaxClassVal = -100000000000000000000000000000000
  if leaf[1] == 1.0 and leaf[2] == 'ThisIsAnEndNode' and pd.isnull(leaf[3]) and pd.isnull(leaf[4]): # See if this is a node where every element is the same class
    print ("END NODE")
    for classVal, row in  df[className].value_counts().to_dict().items():
      print ("\tclassVal=", classVal, "\trow=", int(row) )
      if int(row) > ltMaxCount:
        ltMaxCount = int(row)
        ltMaxClassVal = classVal
    print ("\t", (index, ltMaxClassVal, np.NaN, ltMaxCount, np.NaN) )
    return (index, ltMaxClassVal, np.NaN, ltMaxCount, np.NaN)
  
  print ("\tlen(ltDF)=", len(df[ df[leaf[2]]<=leaf[3] ]), "\tlen(gtDF)=", len(df[ df[leaf[2]]>leaf[3] ]) )
  print ("\tLESS THAN") #Getting the <= decision at node
  for classVal, row in  df[ df[leaf[2]]<=leaf[3] ][className].value_counts().to_dict().items(): # Get the democratic decision from the elements in the LT group of end node
    print ("\tclassVal=", classVal, "\trow=", int(row) )
    if int(row) > ltMaxCount:
      ltMaxCount = int(row)
      ltMaxClassVal = classVal
  print("\tGREATER THAN") #Getting the <= decision at node
  for classVal, row in  df[ df[leaf[2]]>leaf[3] ][className].value_counts().to_dict().items(): # Get the democratic decision from the elements in the GT group of end node
    print ("\tclassVal=", classVal, "\trow=", int(row) )
    if int(row) > gtMaxCount:
      gtMaxCount = int(row)
      gtMaxClassVal = classVal
  print ("\t", (index, ltMaxClassVal, gtMaxClassVal, ltMaxCount, gtMaxCount))
  return (index, ltMaxClassVal, gtMaxClassVal, ltMaxCount, gtMaxCount)


#########################################################################
# With a Finished Tree, get the decision for the LT and GT group at each 
# node bsed upon a weighted democracy. Tuple=(node #, LT decision, GT 
# decision, LT Weight correct, GT weight correct, LT total weight, GT
# total Weight.)
#########################################################################
def GetWeightedNodeDecisions(df, leaf, index, className, soloNodeDecision, gt_or_lt, df_weights, idColumn):
  ltMaxWeight = -100000000000000000000000000000000
  ltMaxClassVal = -100000000000000000000000000000000
  gtMaxWeight = -100000000000000000000000000000000
  gtMaxClassVal = -100000000000000000000000000000000
  ltTotalWeight = -1
  gtTotalWeight = -1
  if leaf[1] == 1.0 and leaf[2] == 'ThisIsAnEndNode' and pd.isnull(leaf[3]) and pd.isnull(leaf[4]): # See if this is a node where every element is the same class
    print("This is An End Node")
    return (index, df[className].unique()[0], np.NaN, df_weights['Weights'].sum(axis=0), np.NaN, 
            df_weights['Weights'].sum(axis=0), np.NaN)
  
  if (soloNodeDecision):
    df_IDs = df[ df[leaf[2]]> leaf[3] ][idColumn].tolist() if gt_or_lt > 0 else df[ df[leaf[2]]<=leaf[3] ][idColumn].tolist()      
    totalWeight = df_weights[ df_weights[idColumn].isin(df_IDs) ]['Weights'].sum(axis=0)
    for classVal, row in  df[ df[idColumn].isin(df_IDs)  ][className].value_counts().to_dict().items():
      currWeight = df_weights[ (df_weights[idColumn].isin(df_IDs)) & (df_weights[className] == classVal) ]['Weights'].sum(axis=0)
      print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
      if currWeight > ltMaxWeight:
        ltMaxWeight = currWeight
        ltMaxClassVal = classVal
    if gt_or_lt > 0: 
      print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. GT Node Decision.")
      return (index, np.NaN, ltMaxClassVal, np.NaN, ltMaxWeight, np.NaN, totalWeight)
    else:            
      print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. LT Node Decision.")
      return (index, ltMaxClassVal, np.NaN, ltMaxWeight, np.NaN, totalWeight, np.NaN)

  print ("\tlen(ltDF)=", len(df[ df[leaf[2]]<=leaf[3] ]), "\tlen(gtDF)=", len(df[ df[leaf[2]]>leaf[3] ]) )
  print ("\tLESS THAN") #Getting the <= decision at node
  df_ltIDs = df[ df[leaf[2]]<=leaf[3] ][idColumn].tolist()
  df_gtIDs = df[ df[leaf[2]]> leaf[3] ][idColumn].tolist()
  ltTotalWeight = df_weights[ df_weights[idColumn].isin(df_ltIDs) ]['Weights'].sum(axis=0)
  gtTotalWeight = df_weights[ df_weights[idColumn].isin(df_gtIDs) ]['Weights'].sum(axis=0)
  for classVal, row in  df[ df[idColumn].isin(df_ltIDs)  ][className].value_counts().to_dict().items():
    currWeight = df_weights[ (df_weights[idColumn].isin(df_ltIDs)) & (df_weights[className] == classVal) ]['Weights'].sum(axis=0)
    print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
    if currWeight > ltMaxWeight:
      ltMaxWeight = currWeight
      ltMaxClassVal = classVal
  print("\tGREATER THAN") #Getting the <= decision at node
  for classVal, row in  df[ df[idColumn].isin(df_gtIDs) ][className].value_counts().to_dict().items(): # Get the democratic decision from the elements in the GT group of end node
    currWeight = df_weights[ (df_weights[idColumn].isin(df_gtIDs)) & (df_weights[className] == classVal) ]['Weights'].sum(axis=0)
    print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
    if currWeight > gtMaxWeight:
      gtMaxWeight = currWeight
      gtMaxClassVal = classVal
  print ("\t", (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight) )
  return (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight)



####################################################################
# Given a final Tree described by the nodes and their tple values
# described above and the decisions of those nodes,  make decisions
# of a set of points in a DF
####################################################################
def ClassifyWithTree(df_test, className, idColumn, maxDepth, outputFileName, nodeDecisionsFileName, nodeValuesFileName):
  print ("\n\n########################################################################\n Classifying test points with Tree from Make Tree\n########################################################################")
  df_Answers = df_test.filter([idColumn], axis=1)
  df_Answers[className] = np.nan # Answer storage df
  with open(nodeDecisionsFileName) as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeValuesFileName) as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
    nodeValues = [tuple(line) for line in nodeValuesFileReader]

  dfIDList = [ (0, df_test[idColumn].tolist()) ]
  maxNodeCount = 0
  for i in range(1,maxDepth+1): maxNodeCount += 2**i # Get the max number of nodes from maxDepth
  for ite in nodeValues: #Iterate through the nodes
    nodeValueTup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4])) # The File stores all info as strings. Cleaner to reassing type now, not every instance.
    print ("\n\tnodeValueTup=", nodeValueTup, "\tnodeValueTup[0]=", nodeValueTup[0], "\tdfIDList[nodeValueTup[0]][0])=", dfIDList[nodeValueTup[0]][0])
#    for tup in dfIDList: print (tup[0])
    dfCurr = df_test.loc[df_test[idColumn].isin(dfIDList[nodeValueTup[0]][1])] # Get the elements of df_test at node

    if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]): # If decision of node from MakeTree is blank, then skip
      print ("\tdf=", dfIDList[nodeValueTup[0]][1])
      if nodeValueTup[0] < maxNodeCount / 2: # If this EndNode isn't at the furthest depth, then add empty placeholders for future BlankNodes
        dfIDList.append( (nodeValueTup[0]*2 + 1, [] ) )
        dfIDList.append( (nodeValueTup[0]*2 + 2, [] ) )
      continue
    elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: # If decision of node from MakeTree is an EndNode, then proceed
      decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get the decision of the End Node
      IDs = dfCurr[idColumn].tolist() # Get df_test elements that made it to this node following the tree structure
      df_Answers.loc[ df_Answers[idColumn].isin(IDs) , className] = decision[1] # Give the elements the appropriate class value from decision
      print ("Class for End-Node=",  decision[1], "\tlen(IDs)=", IDs)
      if nodeValueTup[0] < maxNodeCount / 2: # If this EndNode isn't at the furthest depth, then add empty placeholders for future BlankNodes
        dfIDList.append( (nodeValueTup[0]*2 + 1, [] ) )
        dfIDList.append( (nodeValueTup[0]*2 + 2, [] ) )
    elif nodeValueTup[0] < maxNodeCount / 2: # If element isn't an EndNode and also not at the furthest depth, then proceed
      print ("\tlen(lt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ]) )
      dfIDList.append( (nodeValueTup[0]*2 + 1, dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() ) ) # Give the df_test ID's in the daughter LT leaf of current Node
      dfIDList.append( (nodeValueTup[0]*2 + 2, dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() ) )  # Give the df_test ID"s in the daughter Gt leaf of current Node
      try:  # This sees if the decision of a node is already added. From a sister BlankNode
        decision = next (itetup for itetup in nodeDecisions if int(itetup[0]) == nodeValueTup[0])
        print ("\tOne of this Node's Daughters is a BlankNode.")
        if pd.isnull(float(decision[2]) ) and not pd.isnull(float(decision[1]) ):
          ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          df_Answers.loc[ df_Answers[idColumn].isin(ltIDs) , className] = decision[1] # Apply decision to LT group
          print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs) )
        elif not pd.isnull(float(decision[2]) ) and pd.isnull(float(decision[1]) ):
          gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          df_Answers.loc[ df_Answers[idColumn].isin(gtIDs) , className] = decision[2] # Apply decision to LT group
          print ("\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
        else:
          ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          df_Answers.loc[ df_Answers[idColumn].isin(ltIDs) , className] = decision[1] # Apply decision to LT group
          df_Answers.loc[ df_Answers[idColumn].isin(gtIDs) , className] = decision[2] # Apply decision to LT group
          print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
      except StopIteration:  
        print ("Non of this node's daughters are Blank Nodes")
        continue #If node is not found in nodeDecisions, then add it
    else: # If not an EndNode, BlankNode, or a node NOT at the max depth, then get decisions there
      decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get decision of Make Tree at node
      ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
      gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] >  nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter GT leaf of current Node
      df_Answers.loc[ df_Answers[idColumn].isin(ltIDs) , className] = decision[1] # Apply decision to LT group
      df_Answers.loc[ df_Answers[idColumn].isin(gtIDs) , className] = decision[2] # Apply decision to GT group
      print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
      del ltIDs, gtIDs, decision # Delete containers to preserve memory, in case they don't already get deleted
    del dfCurr 
  #Writing the answers out
  print ("df_Answers=", df_Answers.head(10) )
  df_Answers.to_csv(outputFileName + ".csv", sep=',', index=False) #Write out the answers

