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
import os.path
sys.path.insert(0, '/home/kyletos/Projects/FuzzyTrees/')
from operator import itemgetter
import scipy.stats as st
from DecisionTreeClass import DecisionTree
from BoostedTreeClass import BDT
from FuzzyTreeFunctions import *

class FuzzyDT(BDT):
  '''
  Definition of a Fuzzy Decision Tree. A Fuzzy decision tree excpet that data points can have membership in multiple leaves/branches/nodes. If the entry's calue
  is close to the cut off value for splitting a node, then it will have partial membership. The partial memberships of the entries are kept in the columns called
  "Memberships" and "MembershipNodeList". This tree can also be boosted, which means that each data point will have a boosted weight, but also have a partial 
  membership weight. These two weights are separate and must both be used."
  '''
  def __init__(self, idColumn, className, nodeDFIDsFileName, nodeValuesFileName, nodeDecisionsFileName, outputFileName, treeErrorFileName, maxDepth=4, nGiniSplits=10, giniEndVal=0.01,
               nClasses=2, minSamplesSplit=2, printOutput=True, nEstimators=10, rateOfChange=0.1, colRandomness=0.0, rowRandomness=0.0, isContinuing=False, continuedTreeErrorFile=''
               fuzziness=.9, fuzzyFunction='linear'):
    '''
    Initialize the FuzzyDT, first with the properties of the DecisionTree, then with the properties of the BDT, then with the rest of the elements.
    '''
    DecisionTree.__init__(self, idColumn=idColumn, className=className, nodeDFIDsFileName=nodeDFIDsFileName, nodeValuesFileName=nodeValuesFileName,
                          nodeDecisionsFileName=nodeDecisionsFileName, outputFileName=outputFileName, maxDepth=maxDepth, nGiniSplits=nGiniSplits, giniEndVal=giniEndVal,
                          nClasses=nClasses, minSamplesSplit=minSamplesSplit, printOutput=printOutput)
    BDT.__init__(self, treeErrorFileName=treeErrorFileName, nEstimators=nEstimators, rateOfChange=rateOfChange, colRandomness=colRandomness, rowRandomness=rowRandomness, 
                 isContinuing=isContinuing, continuedTreeErrorFile=continuedTreeErrorFile)
    self.fuzziness     = fuzziness     # This dictates how close to the cutoff a value needs to be to have membership in more than 1 node
    self.fuzzyFunction = fuzzyFunction # This determines the function used to determine how to split the membership percentage. 'linear' available now

  def BuildFuzzyTree(df, df_weights):
    '''
    This builds the fuzzy tree with properties dictated in initialization. Fuzzy Trees allow entries to have a percentage of it's weight in any number of nodes base upon
    if it's value is close to a node's cut value. The "MembershipNodeList" column keeps track of the number of nodes each entry is in, and the "Memberships" column contains
    tuples of the (node number, percent of membership in node) for each entry. After adding the first node, it then checks if the parent node was an EndNode or Blank node.
    If not at the max tree depth or an EndNode or BlankNode, it adds nodes based upon the best split from "FindingBestSplit" and checks if the entries are close enough to the 
    cutoff to have "fuzzy" membership in both daughters. NOTE: when making the tree, the current df_weight of an entry defines it's initial 'Membership' weight at first node. 
    '''
    if self.printOutput: print ("\n\n###################################\n Making a Decision Tree with unique Weights=", df_weights['Weights'].unique(), "\n###################################")
    df['MembershipNodeList'] = [ [0] for _ in range( len(df)) ] 
    df['Memberships'] = df_weights.apply(lambda row: [ (0, row['Weights']) ], axis=1)
    while nodeCount <= maxNodes: 
      if nodeCount == 0: 
        df_currNodeMembWeights = df[self.idColumn].copy()
        df_currNodeMembWeights["Weights"] = df.apply( lambda row: GetCurrentNodeFuzziedWeight( memberships=row['Memberships'], nodeNumber=nodeCount), axis=1) 
        self.nodeValues.append( self.FindingBestSplit(df=df, df_weights=df_currNodeMembWeights) ) 
        df['Memberships'] = df.apply(lambda row:  FuzzyMembershipLinear(value=row[self.nodeValues[0][2]], split=self.nodeValues[0][3], fuzzinessRange=self.nodeValues[0][4]*fuzziness,
                                                                        previousList=row['Memberships'], nodeNumber=0, daughterEndNode=None ), axis=1 )
        df['MembershipNodeList'] = df.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1)
        del df_currNodeMembWeights
      else:
        parentTup = self.nodeValues[(nodeCount-1) // 2] 
        if self.printOutput: print ("\nnode=", nodeCount, "parentNode=", (nodeCount-1) // 2, "\tparentTup=", parentTup)
        if pd.isnull(parentTup[3]) and pd.isnull(parentTup[4]): 
          self.nodeValues.append( (nodeCount, np.NaN, '' , np.NaN, np.NaN) )
        else:
          if self.printOutput: print ("nodeCount=", nodeCount)
          df_Curr = df[ df['MembershipNodeList'].apply(lambda x: True if nodeCount in x else False) ].copy() 
          df_currNodeMembWeights = dfCurr[self.idColumn].copy()
          df_currNodeMembWeights["Weights"] = dfCurr.apply( lambda row: GetCurrentNodeFuzziedWeight( memberships=row['Memberships'], nodeNumber=nodeCount), axis=1)     
          if self.printOutput: print ("len(df_Curr.index)=", len(df_Curr.index))
          if len(df_Curr.index) < minSamplesSplit:
            if self.printOutput: print ("Too Few to be able to split")
            self.nodeValues.append( (nodeCount, np.NaN, '', np.NaN, np.NaN) )
          else: 
            self.nodeValues.append(FindingBestSplit(df=df_Curr, df_weights=df_currNodeMembWeights) ) 
            if not pd.isnull(self.nodeValues[nodeCount][4]) and not pd.isnull(self.nodeValues[nodeCount][3]) and nodeCount < maxNodes/2: 
              df_Curr['Memberships'] = df_Curr.apply(lambda row: FuzzyMembershipLinear( value=row[self.nodeValues[nodeCount][2]], split=self.nodeValues[nodeCount][3],
                                                     fuzzinessRange=self.nodeValues[nodeCount][4]*fuzziness, previousList=row['Memberships'], nodeNumber=nodeCount, daughterEndNode=None ), axis=1 ) 
              df_Curr['MembershipNodeList'] = df_Curr.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1) 
              IDs = df_Curr[idColumn].tolist() 
              df.loc[ df[idColumn].isin(IDs), 'Memberships'] = df_Curr['Memberships']
              df.loc[ df[idColumn].isin(IDs), 'MembershipNodeList'] = df_Curr['MembershipNodeList'] 
            elif self.nodeValues[nodeCount][2] == '':
              df_Curr['Memberships'] = df_Curr.apply(lambda row: ReturnNodePointsToParent(memberships=row['Memberships'], nodeNumber=nodeCount), axis=1)
              df_Curr['MembershipNodeList'] = df_Curr.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1)
              IDs = df_Curr[idColumn].tolist() 
              df.loc[ df[idColumn].isin(IDs), 'Memberships'] = df_Curr['Memberships'] 
              df.loc[ df[idColumn].isin(IDs), 'MembershipNodeList'] = df_Curr['MembershipNodeList'] 
               
            if self.printOutput:  print ("######## NEW ########:", "self.nodeValues[nodeCount]=", self.nodeValues[nodeCount], "\tlen(nodeDFIds[", nodeCount, "][1])=", 
                                         len( df[ df['MembershipNodeList'].apply(lambda x: True if nodeCount in x else False)].index ) )
      nodeCount += 1
  

  def GatherAndWriteFuzzyDecisions(self, df, df_weights):
    '''
    This function gets the decisions the entries at each node and calls and organizes the decisions calculated in the function "GetFuzzyWeightedNodeDecisions". It loops through the 
    nodes at the max depth. First, the entries with some membership at the current node are grouped. If the node is a BlankNode, then it looks at the parent node to see if there is
    a decision there, and if not, looks at the parent's parent, and so on. It also checks if a node has a sister that is either a BlankNode or EndNode. Then it gets the current node's
    weight of Membership and uses that to get the decision at the node, based upon weighted democracy via "GetFuzzyWeightedNodeDecisions", and adds the decision, if not already added.
    '''
    if self.printOutput: print ("df_weights['Weights'].sum(axis=0)=", df_weights['Weights'].sum(axis=0) )
    for ite in range(self.maxNodes, self.maxNodes - 2**self.maxDepth, -1):
      index = ite 
      if self.printOutput: print ("\n\nindex=", index)
      currentLeaf = self.nodeValues[index]
      currentDF = df[ df['MembershipNodeList'].apply(lambda x: True if index in x else False) ].copy() 
      currentDF_weights = df_weights.loc[ df[self.idColumn].isin(currentDF[self.idColumn].tolist() )]
      gt_or_lt = 0 
      soloNode = False 
      while pd.isnull(currentLeaf[1]) and  currentLeaf[2] == '' and pd.isnull(currentLeaf[3]) and pd.isnull(currentLeaf[4]): 
        sisterNode = index-1 if index % 2 == 0 else index+1
        gt_or_lt = 1 if index % 2 == 0 else -1
        soloNode = False 
        if  pd.isnull(self.nodeValues[sisterNode][3]) or pd.isnull(self.nodeValues[sisterNode][4]): soloNode = True
        index = (index-1) // 2
        currentLeaf = self.nodeValues[index]
        currentDF = df[ df['MembershipNodeList'].apply(lambda x: True if index in x else False) ].copy()
        currentDF_weights = df_weights.loc[ df[self.idColumn].isin(currentDF[self.idColumn].tolist() )]
      if self.printOutput: print ("index=", index, "\tgt_or_lt=", gt_or_lt, "\tsoloNode=", soloNode, "\tcurrentLeaf=", currentLeaf, "\tlen(currentDF)=", len(currentDF) )
      currentDF['CurrNodeFuzziedWeight'] = currentDF.apply( lambda row: GetCurrentNodeFuzziedWeight( memberships=row['Memberships'], nodeNumber=index), axis=1)
      if self.printOutput: print ("currentDF[[className, 'CurrNodeFuzziedWeight', 'Memberships']]=\n", currentDF[[self.className, 'CurrNodeFuzziedWeight', 'Memberships']] )
      currentNodeDecision = (GetFuzzyWeightedNodeDecisions(df=currentDF, leaf=currentLeaf, index=index, soloNodeDecision=soloNode, gt_or_lt=gt_or_lt) ) 
      try:  
        next (tup for tup in self.nodeDecisions if tup[0] == index)
        print ("Decision already included from other daugther node")
      except StopIteration:  
        self.nodeDecisions.append(currentNodeDecision )
  
  
  def GetFuzzyWeightedNodeDecisions(df, leaf, index, soloNodeDecision, gt_or_lt):
    '''
    This function uses the fuzzy weight to get a decision at a node. First, it checks for an EndNode. Then it checks for a soloNode. And then it finds the decision for the LT
    and GT group at the node.
    '''
    if leaf[1] == 1.0 and leaf[2] == 'ThisIsAnEndNode' and pd.isnull(leaf[3]) and pd.isnull(leaf[4]): 
      if self.printOutput: print("This is An End Node")
      return (index, df[self.className].unique()[0], np.NaN, df['CurrNodeFuzziedWeight'].sum(axis=0), np.NaN, df['CurrNodeFuzziedWeight'].sum(axis=0), np.NaN)
    if (soloNodeDecision):
      df_IDs = df[ df[leaf[2]]> leaf[3] ][idColumn].tolist() if gt_or_lt > 0 else df[ df[leaf[2]]<=leaf[3] ][idColumn].tolist()
      dfCurr = df[ df[idColumn].isin(df_IDs)  ]
      GetFuzzyWeightedNodeDecisions_SoloNodeDecision(df=dfCurr, leaf=leaf, index=index, gt_or_lt=gt_or_lt)
    else: GetFuzzyWeightedNodeDecisions_BothNodeDecision(df=df, leaf=leaf, index=index)


  def GetFuzzyWeightedNodeDecisions_BothNodeDecision(df, leaf, index):
    '''
    This gets the decision for both the LT and GT group of a node. It gets the total weight in each, and finds the class with the highest percentage of weight. Works with many classes.
    ''' 
    if self.printOutput: print ("\tlen(ltDF)=", len(df[ df[leaf[2]]<=leaf[3] ]), "\tlen(gtDF)=", len(df[ df[leaf[2]]>leaf[3] ]) )
    ltMaxWeight = -100000000000000000000000000000000
    ltMaxClassVal = -100000000000000000000000000000000
    gtMaxWeight = -100000000000000000000000000000000
    gtMaxClassVal = -100000000000000000000000000000000
    ltTotalWeight = df[ df[leaf[2]]<=leaf[3] ]['CurrNodeFuzziedWeight'].sum(axis=0)
    gtTotalWeight = df[ df[leaf[2]]> leaf[3] ]['CurrNodeFuzziedWeight'].sum(axis=0)

    if self.printOutput: print ("\tLESS THAN") 
    for classVal, row in  df[ df[leaf[2]]<=leaf[3]  ][self.className].value_counts().to_dict().items(): 
      currWeight = df[ (df[leaf[2]]<=leaf[3]) & (df[self.className] == classVal) ]['CurrNodeFuzziedWeight'].sum(axis=0)
      if self.printOutput: print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
      if currWeight > ltMaxWeight:
        ltMaxWeight = currWeight
        ltMaxClassVal = classVal

    if self.printOutput: print("\tGREATER THAN") 
    for classVal, row in  df[ df[leaf[2]]> leaf[3] ][self.className].value_counts().to_dict().items(): 
      currWeight = df[ (df[leaf[2]]> leaf[3]) & (df[self.className] == classVal) ]['CurrNodeFuzziedWeight'].sum(axis=0)
      if self.printOutput: print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
      if currWeight > gtMaxWeight:
        gtMaxWeight = currWeight
        gtMaxClassVal = classVal

    if self.printOutput: print ("\t", (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight) )
    return (index, ltMaxClassVal, gtMaxClassVal, ltMaxWeight, gtMaxWeight, ltTotalWeight, gtTotalWeight)
   

  def GetFuzzyWeightedNodeDecisions_SoloNodeDecision(df, leaf, index, gt_or_lt):
    '''
    This gets the decision for a solo node. It gets the total weight, and finds the class with the highest percentage of weigh. It works with many classes.
    '''
    maxWeight = -100000000000000000000000000000000
    maxClassVal = -100000000000000000000000000000000
    totalWeight = df['CurrNodeFuzziedWeight'].sum(axis=0)
    for classVal, row in  df[self.className].value_counts().to_dict().items():
      currWeight = df[df[self.className] == classVal ]['CurrNodeFuzziedWeight'].sum(axis=0)
      if self.printOutput: print ("\tclassVal=", classVal, "\tcurrWeight=", currWeight)
      if currWeight > maxWeight:
        maxWeight = currWeight
        maxClassVal = classVal
    if gt_or_lt > 0:
      if self.printOutput: print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. GT Node Decision.")
      return (index, np.NaN, maxClassVal, np.NaN, maxWeight, np.NaN, totalWeight)
    else:
      if self.printOutput: print ("Blank Node has a non-Blank Sister, so only make decision for one part of parent node. LT Node Decision.")
      return (index, maxClassVal, np.NaN, maxWeight, np.NaN, totalWeight, np.NaN)





  def GetFuzzyBoostingTreesErrorsAndWeights(df, df_weights):
    '''
    This function creates "nEstimators" number of Fuzzy Trees to then boost with an amount of randomness dictates by "colRandomness" and "rowRandomness". This is very similar to
    The boosting function in "BoostedTreeClass.py", except it alters the boosting weights in "df_weights" w.r.t. the membership present in each node. Each entry could be altered
    several times in each tree, based upon it's node membership.
    '''
    if self.isContinuing == True and self.lastCompletedEstimator != -1 and lastCompletedEstimator < self.nEstimators:
      df_weights = pd.read_csv("Answers/DF_WEIGHTS.csv")
    try:
      while self.currEst > 0: 
        columnsList = [ ite for ite in df.columns.tolist() if ite != self.className and ite != self.idColumn ]
        dfCurr = df[ random.sample(columnsList, math.ceil(len(columnsList) * (1-colRandomness) ) ) ].copy() 
        dfCurr[self.className] = df[self.className]
        dfCurr[self.idColumn ] = df[self.idColumn ]
        dfCurr = dfCurr.sample( math.ceil(len(dfCurr.index) * (1-rowRandomness) ) ) 
        dfCurr_weights = df_weights[ df_weights[self.idColumn].isin(dfCurr[self.idColumn].tolist() )].copy()

        self.nodeDecisionsFileName = self.nodeDecisionsFileName.rstrip('1234567890') + str(self.currEst) 
        self.nodeValuesFileName    = self.nodeValuesFileName.rstrip('1234567890')    + str(self.currEst)
        if self.printOutput: print ("###########################\n###########################\n  STARTING ESTIMATOR", self.currEst, "\n###########################\n###########################")
        TEMP = self.BuildFuzzyTree(df=dfCurr, df_weights=dfCurr_weights) 
        if TEMP == "ERROR": raise UnboundLocalError("\n\nRunning Ended Early")
        self.GatherAndWriteFuzzyDecisions(df=dfCurr, df_weights=dfCurr_weights)
        self.treeError.append( (self.currEst, self.GetTreeError(df=dfCurr, df_weights=dfCurr_weights) ) )




        dfCurr_weights = self.AlterFuzzyWeights(df=dfCurr, df_weights=dfCurr_weights, error=next(i[1] for i in self.treeError if i[0] == self.currEst) )

        if self.printOutput: print ("BEFORE: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )
        IDs = dfCurr_weights[self.idColumn].tolist()  
        df_weights.loc[ df_weights[self.idColumn].isin( IDs ), 'Weights'] = dfCurr_weights['Weights'] 
        if self.printOutput: print ("After: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )

        if self.printOutput:
          for ite in df_weights['Weights'].unique():
            print ("\tlen('Weights' ==", ite, ")=", len(df_weights[ df_weights['Weights'] == ite].index) )
        df_weights['Weights'] = df_weights['Weights'] / df_weights['Weights'].sum(axis=0) 
        if self.printOutput: print ("\n\n####################\nFINAL SCORE FOR TREE #", self.currEst, "is 1-error=", 1-next(i[1] for i in treeError if i[0] == self.currEst),"\n##############")
        self.currEst -= 1
        self.CleanTree() 

 
      # Writing the TreeErrors
      currTreeErrorFileName =  self.treeErrorFileName + ".csv"
      if os.path.isfile(currTreeErrorFileName): treeErrorFile = open(currTreeErrorFileName, 'a')
      else: treeErrorFile = open(currTreeErrorFileName, 'w')
      treeErrorFileCSV=csv.writer(treeErrorFile)
      treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
      for tup in self.treeError:
        treeErrorFileCSV.writerow(tup)

    except (KeyboardInterrupt,UnboundLocalError): #If user stops runnig, still save 
        currTreeErrorFileName =  self.treeErrorFileName + "_Incomplete.csv"
        if os.path.isfile(currTreeErrorFileName):
          treeErrorFile = open(currTreeErrorFileName, 'a')
          treeErrorFileCSV=csv.writer(treeErrorFile)
        else:
          treeErrorFile = open(currTreeErrorFileName, 'w')
          treeErrorFileCSV=csv.writer(treeErrorFile)
          treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
        for tup in self.treeError:
          treeErrorFileCSV.writerow(tup)
        df_weights.to_csv("Answers/DF_WEIGHTS.csv", sep=',', index=False)
 

  def AlterFuzzyWeights(self, df, df_weights, error):
    '''
    This alters the weights of the entries in a dataframe based upon their correctness. This also takes into account whether part of their membership was correctly identified and some
    wasn't. First, the function gets teh weights for the elements that have some membership at each node in the list of node decisions. Then it checks if the node is an EndNode, a node
    where all the elements have the same class. Then, it checks if the decision has a GT decision and then a LT decision. Each time, weights are increased if incorrectly identified, and
    vice versa. The function "ChangeWeightWithRow" changes the weights. "moreOrLess" indicates whether the entry weight needs to be increased or decreased.
    '''
    alpha = .5 * math.log1p((1 - error) / error) * self.rateOfChange 
    if self.printOutput: print ("error=", error, "\talpha*rateOfChange=", alpha, "\tCorrectFactor=", math.exp(-1*alpha), "\tIncorrectFactor=", math.exp(1*alpha)  )
    for decisionTup in self.nodeDecisions:
      df_weights['MembershipNodeList'] = df['MembershipNodeList']
      df_weights['Memberships'] = df['Memberships']
      dfIDs = df[ df['MembershipNodeList'].apply(lambda x: True if int(decisionTup[0]) in x else False) ][self.idColumn].tolist() 
      nodeValuesTup = next(iteTup for iteTup in self.nodeValues if int(iteTup[0]) == int(decisionTup[0]) )
      if nodeValuesTup[2] == "ThisIsAnEndNode": 
        df_weights.loc[ df_weights[self.idColumn].isin(dfIDs), 'Weights'] = df_weights.loc[ df_weights[self.idColumn].isin(dfIDs)].apply(lambda row: 
                                                            ChangeWeightWithRow(weight=row['Weights'], moreOrLess=-1, memberships=row['Memberships'], 
                                                            alpha=alpha, nodeNumber=nodeValuesTup[0]), axis=1 )
      elif not pd.isnull(float(decisionTup[2]) ): 
        gtCorrIDs  = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[self.className] == int(decisionTup[2]) )][self.idColumn].tolist() 
        gtWrongIDs = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[self.className] != int(decisionTup[2]) )][self.idColumn].tolist() 
        if len(gtCorrIDs) > 0:
          df_weights.loc[ df_weights[self.idColumn].isin(gtCorrIDs),  'Weights'] = df_weights.loc[ df_weights[self.idColumn].isin(gtCorrIDs)].apply(lambda row: 
                                                                                 ChangeWeightWithRow(weight=row['Weights'], moreOrLess=-1, memberships=row['Memberships'], 
                                                                                 alpha=alpha, nodeNumber=nodeValuesTup[0]), axis=1 )
        if len(gtWrongIDs) > 0: 
          df_weights.loc[ df_weights[self.idColumn].isin(gtWrongIDs), 'Weights'] = df_weights.loc[df_weights[ self.idColumn].isin(gtWrongIDs)].apply(lambda row: 
                                                                                 ChangeWeightWithRow(weight=row['Weights'], moreOrLess=1,  memberships=row['Memberships'], 
                                                                                 alpha=alpha, nodeNumber=nodeValuesTup[0]), axis=1 )
      elif not pd.isnull(float(decisionTup[1]) ): 
        ltCorrIDs  = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[self.className] == int(decisionTup[1]) )][self.idColumn].tolist() 
        ltWrongIDs = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[self.className] != int(decisionTup[1]) )][self.idColumn].tolist() 
        if len(ltCorrIDs) > 0:
          df_weights.loc[ df_weights[self.idColumn].isin(ltCorrIDs),  'Weights'] = df_weights.loc[ df_weights[self.idColumn].isin(ltCorrIDs)].apply(lambda row: 
                                                                                 ChangeWeightWithRow(weight=row['Weights'], moreOrLess=-1, memberships=row['Memberships'], 
                                                                                 alpha=alpha, nodeNumber=nodeValuesTup[0]), axis=1 )
        if len(ltWrongIDs) > 0: 
          df_weights.loc[ df_weights[self.idColumn].isin(ltWrongIDs), 'Weights'] = df_weights.loc[ df_weights[self.idColumn].isin(ltWrongIDs)].apply(lambda row: 
                                                                                 ChangeWeightWithRow(weight=row['Weights'], moreOrLess=1,  memberships=row['Memberships'], 
                                                                                 alpha=alpha, nodeNumber=nodeValuesTup[0]), axis=1 )
    return df_weights








  def ClassifyWithFuzzyTree(self, df_test):
    '''
    Classifies a test set after have tree built. It iterates through the nodes in the "self.nodeValues" element and propogates the test set entries to their final node location.
    It first checks for EndNodes or BlankNodes. Then, if the node isn't at the maximum depth of the tree, it checks to see if the node has a decision and adds the membership weight
    and that node to the decision at that node. If no decision, split the groups at that node, with fuzziness and go to next node. If at max tree depth, add weight to class at node 
    from built tree.  
    '''
    print ("\n\n###########################\n Classifying test points with Tree from Make Tree\n###########################")
    df_Answers = df_test.filter([self.idColumn], axis=1)
    for classVal in uniqueClasses:
      df_test[self.className + "_" + str(classVal)] = 0.0 
    df_test['Memberships'] = [ [(0, 1.0)] for _ in range( len(df_test)) ] 
    df_test['MembershipNodeList'] = [ [0] for _ in range( len(df_test)) ] 

    for nodeValueTup in self.nodeValues: 
      if self.printOutput: print ("\n\tnodeValueTup=", nodeValueTup )
      df_Curr = df_test[ df_test['MembershipNodeList'].apply(lambda x: True if nodeValueTup[0] in x else False) ].copy() 
  
      if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]):
        if self.printOutput: print ("\tlen(df)=", len(df_Curr.index) )
        continue

      elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: 
        decision = next (ite for ite in self.nodeDecisions if int(itetup[0]) == nodeValueTup[0])
        IDs = df_Curr[self.idColumn].tolist()
        if len(IDs) > 0: self.ClassifyWithFuzzyTre_AddDecisionWeights(df=df_test, decision=str(decision[1]), nodeNumber=nodeValueTup[0], IDs=IDs)

      elif nodeValueTup[0] < maxNodeCount / 2: 
        daughterEndNodeCheck = None 
        try:  
          decision = next (itetup for itetup in nodeDecisions if int(itetup[0]) == nodeValueTup[0]) 
          if self.printOutput: print ("\tOne of this Node's Daughters is a BlankNode.")

          if pd.isnull(float(decision[2]) ) and not pd.isnull(float(decision[1]) ): 
            daughterEndNodeCheck = 'LT'
            ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist() 
            if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs) )
            if len(ltIDs) > 0: self.ClassifyWithFuzzyTre_AddDecisionWeights(df=df_test, decision=str(decision[1]), nodeNumber=nodeValueTup[0], IDs=ltIDs)

          elif not pd.isnull(float(decision[2]) ) and pd.isnull(float(decision[1]) ): 
            daughterEndNodeCheck = 'GT'
            gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][self.idColumn].tolist() 
            if self.printOutput: print ("\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
            if len(gtIDs) > 0: 
              self.ClassifyWithFuzzyTre_AddDecisionWeights(df=df_test, decision=str(decision[1]), nodeNumber=nodeValueTup[0], IDs=gtIDs)
                   
          else:   
            daughterEndNodeCheck = 'BOTH'
            ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist() 
            gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][self.idColumn].tolist() 
            if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
            if len(ltIDs) > 0: self.ClassifyWithFuzzyTre_AddDecisionWeights(df=df_test, decision=str(decision[1]), nodeNumber=nodeValueTup[0], IDs=ltIDs)
            if len(gtIDs) > 0: self.ClassifyWithFuzzyTre_AddDecisionWeights(df=df_test, decision=str(decision[1]), nodeNumber=nodeValueTup[0], IDs=gtIDs) 
        except StopIteration:
          print ("Non of this node's daughters are Blank Nodes")
        df_Curr['Memberships'] = df_Curr.apply(lambda row: 
                                        FuzzyMembershipLinear( value=row[nodeValueTup[2]], split=nodeValueTup[3], splitLength=nodeValueTup[4], duality=duality,
                                        previousList=row['Memberships'], nodeNumber=nodeValueTup[0], daughterEndNode=daughterEndNodeCheck ), axis=1 ) 
        df_Curr['MembershipNodeList'] = df_Curr.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1) 
        IDs = df_Curr[self.idColumn].tolist() 
        df_test.loc[ df_test[self.idColumn].isin(IDs), 'Memberships'] = df_Curr['Memberships'] 
        df_test.loc[ df_test[self.idColumn].isin(IDs), 'MembershipNodeList'] = df_Curr['MembershipNodeList'] 

      else: 
        decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) 
        ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist() 
        gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][self.idColumn].tolist() 
        if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
        if len(ltIDs) > 0: self.ClassifyWithFuzzyTre_AddDecisionWeights(df=df_test, decision=str(decision[1]), nodeNumber=nodeValueTup[0], IDs=ltIDs)
        if len(gtIDs) > 0: self.ClassifyWithFuzzyTre_AddDecisionWeights(df=df_test, decision=str(decision[1]), nodeNumber=nodeValueTup[0], IDs=gtIDs)

    #Writing the answers out
    df_Answers[self.className] = -1
    df_Answers[self.className + "_probability"] = -1
    for classVal in self.uniqueClasses:
      if self.printOutput: print ("classVal=", classVal, "\ndf_test[className + '_' + str(classVal):\n", df_test[self.className + "_" + str(classVal)] )
      df_Answers[self.className + "_" + str(classVal)] = df_test[self.className + "_" + str(classVal)]
      df_Answers.loc[ df_Answers[self.className + "_" + str(classVal)] > df_Answers[self.className], self.className] = classVal 
      df_Answers.loc[ df_Answers[self.className] == classVal, self.className + "_probability"] = df_Answers[self.className + "_" + str(classVal)] 
    df_Answers.to_csv(outputFileName + "_Prob_Frac_ExtraInfo.csv", sep=',', index=False) 
    df_Answers[[self.idColumn, self.className]].to_csv(outputFileName + ".csv", sep=',', index=False) 


  def ClassifyWithFuzzyTre_AddDecisionWeights(self, df, decision, nodeNumber, IDs):
    '''
    Adds the membership weight of the test set entries at an EndNode to the decision already found for that node. 
    '''
    df_Changed = df[ df[self.idColumn].isin(IDs)].copy() 
    df_Changed[self.className + "_" + decision] = df_Changed.apply(lambda row: 
                    FuzzyDecisionScoreUpdate( previous=row[self.className + "_" + decision], membershipList=row['Memberships'], nodeNumber=nodeNumber), axis=1)
    df.loc[ df[self.idColumn].isin(IDs), self.className + "_" + decision] = df_Changed[self.className + "_" + decision]






  
  ####################################################################
  # Given a final Tree described by the nodes and their tple values
  # described above and the decisions of those nodes,  make decisions
  # of a set of points in a DF
  ####################################################################
  def ClassifyWithBoostedFuzzyTree(df_test, nEstimators, className, idColumn, maxDepth, duality, uniqueClasses, treeErrorFileName, outputFileName, nodeDecisionsFileName, nodeValuesFileName):
    print ("\n\n########################################################################\n Classifying test points with Tree from Make Tree\n########################################################################")
    df_Answers = df_test.filter([idColumn], axis=1)
    df_test[className + "_total"] = 0.0 # Total avaliable decision weight based upon the alphas
    for classVal in uniqueClasses:
      df_test[className + "_" + str(classVal)] = 0.0 # Create new column for sum of alpha's for each unique className value
    with open(treeErrorFileName + ".csv") as treeErrorFile:
      treeErrorFileReader = csv.reader(treeErrorFile)
      next(treeErrorFileReader)
      treeError = [tuple(line) for line in treeErrorFileReader]
    currEst = 1
    while currEst <= nEstimators: # Loop over the nEstimators number of trees in boost
      df_test['Memberships'] = [ [(0, 1.0)] for _ in range( len(df_test)) ] # Initialize all points to be a part of node 0 wiht complete percent membership there
      df_test['MembershipNodeList'] = [ [0] for _ in range( len(df_test)) ] # Initialize all points to have their list of nodes they are a part of be only node 0
      nodeValuesFileName =  nodeValuesFileName.rstrip('1234567890') + str(currEst)
      nodeDecisionsFileName =  nodeDecisionsFileName.rstrip('1234567890') + str(currEst)
      with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
        nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
        next(nodeDecisionsFileReader)
        nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
      with open(nodeValuesFileName + ".csv") as nodeValuesFile:
        nodeValuesFileReader = csv.reader(nodeValuesFile)
        next(nodeValuesFileReader)
        nodeValues = [tuple(line) for line in nodeValuesFileReader]
   
      currErrorTup = next(iteTup for iteTup in treeError if int(iteTup[0]) == currEst)
      alpha = .5 * math.log1p((1 - float(currErrorTup[1]) ) / float(currErrorTup[1]) ) # exponent factor for weight of decision 
      df_test[className + "_total"] += alpha
      print ("currEst=", currEst, "\talpha=", alpha, "\tcurrErrorTup=", currErrorTup, "\ndf_test[classNames]=", df_test[[className + "_0", className + "_1", className + "_total"]])
      maxNodeCount = 0
      for i in range(1,maxDepth+1): maxNodeCount += 2**i # Get the max number of nodes based upon maxDepth  
      for ite in nodeValues: #Iterate through the nodes
        nodeValueTup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4])) # The File stores all info as strings. Cleaner to reassing type now, not every instance.
        print ("\n\tnodeValueTup=", nodeValueTup )
        df_Curr = df_test[ df_test['MembershipNodeList'].apply(lambda x: True if nodeValueTup[0] in x else False) ].copy() # Get the nodes that have a membership in current node number
    
        if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]): # If decision of node from MakeTree is blank, then skip
          print ("\tlen(df)=", len(df_Curr.index) )
          continue
        elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: #If decision of node from MakeTree is an EndNode, then proceed
          decision = next (itetup for itetup in nodeDecisions if int(itetup[0]) == nodeValueTup[0])  # Try to find decision. If no decision, node's daugthers are not blank nodes. Proceed to excpetion
          IDs = df_Curr[idColumn].tolist() # Get the IDs of the points at this node
          if len(IDs) > 0: # Check if there are some test points  here
            df_test_Changed = df_test[ df_test[idColumn].isin(IDs)].copy() # Create dummy df. Below put the precent membership, weighted by alpha, towards the decision of that node. 
            df_test_Changed[className + "_" + str(decision[1])] = df_test_Changed.apply(lambda row: 
                                BoostedFuzzyDecisionScoreUpdate(  previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0], alpha=alpha), axis=1)
            df_test.loc[ df_test[idColumn].isin(IDs), className + "_" + str(decision[1])] = df_test_Changed[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
        elif nodeValueTup[0] < maxNodeCount / 2: # If element isn't an EndNode and also not at the furthest depth, then proceed
          daughterEndNodeCheck = None
          try:  # This sees if the decision of a node is already added. From a sister BlankNode
            decision = next (itetup for itetup in nodeDecisions if int(itetup[0]) == nodeValueTup[0])
            print ("\tOne of this Node's Daughters is a BlankNode.")
            if pd.isnull(float(decision[2]) ) and not pd.isnull(float(decision[1]) ):
              daughterEndNodeCheck = 'LT'
              ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
              if len(ltIDs) > 0: # Check if there are some test points  here
                df_test_Changed = df_test[ df_test[idColumn].isin(ltIDs)].copy() # Create dummy df. Below put the precent membership, weighted by alpha, towards the decision of that node
                df_test_Changed[className + "_" + str(decision[1])] = df_test_Changed.apply(lambda row: 
                                BoostedFuzzyDecisionScoreUpdate(  previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0], alpha=alpha), axis=1)
                df_test.loc[ df_test[idColumn].isin(ltIDs), className + "_" + str(decision[1])] = df_test_Changed[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
              print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs) )
            elif not pd.isnull(float(decision[2]) ) and pd.isnull(float(decision[1]) ):
              daughterEndNodeCheck = 'GT'
              gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
              if len(gtIDs) > 0: # Check if there are some test points  here
                df_test_Changed = df_test[ df_test[idColumn].isin(gtIDs)].copy() # Create dummy df. Below put the precent membership, weighted by alpha, towards the decision of that node
                df_test_Changed[className + "_" + str(decision[2])] = df_test_Changed.apply(lambda row: 
                                BoostedFuzzyDecisionScoreUpdate(  previous=row[className + "_" + str(decision[2])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0], alpha=alpha), axis=1)
                df_test.loc[ df_test[idColumn].isin(gtIDs), className + "_" + str(decision[2])] = df_test_Changed[className + "_" + str(decision[2])] # Set updated decisions to non-copy df_test
              print ("\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
            else:
              daughterEndNodeCheck = 'BOTH'
              ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
              gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
              if len(ltIDs) > 0: # Check if there are some test points  here
                df_test_Changed_lt = df_test[ df_test[idColumn].isin(ltIDs)].copy() # Create dummy df. Below put the precent membership, weighted by alpha, towards the decision of that node
                df_test_Changed_lt[className+"_"+str(decision[1])] = df_test_Changed_lt.apply(lambda row: 
                                   BoostedFuzzyDecisionScoreUpdate( previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0], alpha=alpha), axis=1)
                df_test.loc[ df_test[idColumn].isin(ltIDs), className + "_" + str(decision[1])] = df_test_Changed_lt[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
              if len(gtIDs) > 0: # Check if there are some test points  here 
                df_test_Changed_gt = df_test[ df_test[idColumn].isin(gtIDs)].copy() # Create dummy df. Below put the precent membership, weighted by alpha, towards the decision of that node
                df_test_Changed_gt[className+"_"+str(decision[2])] = df_test_Changed_gt.apply(lambda row: 
                                   BoostedFuzzyDecisionScoreUpdate( previous=row[className + "_" + str(decision[2])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0], alpha=alpha), axis=1)
                df_test.loc[ df_test[idColumn].isin(gtIDs), className + "_" + str(decision[2])] = df_test_Changed_gt[className + "_" + str(decision[2])] # Set updated decisions to non-copy df_test
              print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
          except StopIteration:
            print ("Non of this node's daughters are Blank Nodes")
          print ("BEFORE: df_Curr[['MembershipNodeList', 'Memberships']].head(100)=", df_Curr[['MembershipNodeList', 'Memberships']].head(100) )
          df_Curr['Memberships'] = df_Curr.apply(lambda row: 
                                           FuzzyMembershipLinear( value=row[nodeValueTup[2]], split=nodeValueTup[3], splitLength=nodeValueTup[4], duality=duality,
                                           previousList=row['Memberships'], nodeNumber=nodeValueTup[0], daughterEndNode=daughterEndNodeCheck ), axis=1 ) # Sets points membership via linear range
          df_Curr['MembershipNodeList'] = df_Curr.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1) # Updates Membership's Node list depending on the points membership
          IDs = df_Curr[idColumn].tolist() # Get the df_test ID's in the
          df_test.loc[ df_test[idColumn].isin(IDs), 'Memberships'] = df_Curr['Memberships']  # Update df_test 'Membership' with df_Curr's values
          df_test.loc[ df_test[idColumn].isin(IDs), 'MembershipNodeList'] = df_Curr['MembershipNodeList'] #Update df_test 'MembershipNodeList' with df_Curr's value
          print ("AFTER: df_Curr[['MembershipNodeList','Memberships']].head(100)=", df_Curr[['MembershipNodeList', 'Memberships']].head(100) )
        else: # If not an EndNode, BlankNode, or a node NOT at the max depth, then get decisions there
          decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get decision of Make Tree at node
          ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          if len(ltIDs) > 0: # Check if there are some test points  here
            df_test_Changed_lt = df_test[ df_test[idColumn].isin(ltIDs)].copy() # Create dummy df. Below put the precent membership, weighted by alpha, towards the decision of that node
            df_test_Changed_lt[className+ "_" + str(decision[1])] = df_test_Changed_lt.apply(lambda row: 
                               BoostedFuzzyDecisionScoreUpdate(  previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0], alpha=alpha), axis=1)
            df_test.loc[ df_test[idColumn].isin(ltIDs), className + "_" + str(decision[1])] = df_test_Changed_lt[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
          if len(gtIDs) > 0: # Check if there are some test points  here
            df_test_Changed_gt = df_test[ df_test[idColumn].isin(gtIDs)].copy() # Create dummy df. Below put the precent membership, weighted by alpha, towards the decision of that node
            df_test_Changed_gt[className+ "_" + str(decision[2])] = df_test_Changed_gt.apply(lambda row: 
                               BoostedFuzzyDecisionScoreUpdate(  previous=row[className + "_" + str(decision[2])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0], alpha=alpha), axis=1)
            df_test.loc[ df_test[idColumn].isin(gtIDs), className + "_" + str(decision[2])] = df_test_Changed_gt[className + "_" + str(decision[2])] # Set updated decisions to non-copy df_test
          print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
  
      currEst += 1
      nodeDecisionsFile.close()
      del nodeDecisionsFileReader
      nodeValuesFile.close()
      del nodeValuesFileReader
  
  
    #Writing the answers out
    df_Answers[className] = -1
    df_Answers[className + "_probability"] = -1
    for classVal in uniqueClasses:
      print ("classVal=", classVal, "\ndf_test[className + '_' + str(classVal):\n", df_test[className + "_" + str(classVal)] )
      df_test[className + "_" + str(classVal)] = df_test[className + "_" + str(classVal)] / df_test[className + "_total"] # Normalizing sums to total, to make a probabilit
      df_Answers[className + "_" + str(classVal)] = df_test[className + "_" + str(classVal)]
      df_Answers.loc[ df_Answers[className + "_" + str(classVal)] > df_Answers[className], className] = classVal # if current classVal prob is greater, reassign answer to the classVal
      df_Answers.loc[ df_Answers[className] == classVal, className + "_probability"] = df_Answers[className + "_" + str(classVal)] # If current classVal prob got changed, reassign probab
    df_Answers.to_csv(outputFileName + "_Prob_Frac_ExtraInfo.csv", sep=',', index=False) #Write out the answers with all answer information
    df_Answers[[idColumn, className]].to_csv(outputFileName + ".csv", sep=',', index=False) #Write out the answers
  
