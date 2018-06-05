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
from DecisionTreeClass import DecisionTree 

class BDT(DecisionTree):
  '''
  This implements a Boosted version of the 'DecisionTree' and inheirets from it, but also has it's own methods.
  '''

  def __init__(self, idColumn, className, nodeDFIDsFileName, nodeValuesFileName, nodeDecisionsFileName, outputFileName, treeErrorFileName, maxDepth=4, nGiniSplits=10, giniEndVal=0.01, 
               minSamplesSplit=2, printOutput=True, nEstimators=10, rateOfChange=0.1, colRandomness=0.0, rowRandomness=0.0, writeTree=False, isContinuing=False, continuedTreeErrorFile=''):
    '''
    Initialize the BDT, first with the properties of the DecisionTree, then with the rest of the elements.
    '''
    DecisionTree.__init__(self, idColumn=idColumn, className=className, nodeDFIDsFileName=nodeDFIDsFileName, nodeValuesFileName=nodeValuesFileName, 
                          nodeDecisionsFileName=nodeDecisionsFileName, outputFileName=outputFileName, maxDepth=maxDepth, nGiniSplits=nGiniSplits, giniEndVal=giniEndVal, 
                          minSamplesSplit=minSamplesSplit, printOutput=printOutput)
    self.nEstimators  = nEstimators   # The number of estimators used in the boost
    self.rateOfChange = rateOfChange  # The speed at which the weights change
    if colRandomness >= 0.0 and colRandomness < 1.0: self.colRandomness = colRandomness # The fraction of columns to be left out to induce randomness
    else: self.colRandomness = 0
    if rowRandomness >= 0.0 and rowRandomness < 1.0: self.rowRandomness = rowRandomness # The fraction of datafram entries to be left out to induce randomness
    else: self.rowRandomness = 0
    self.treeErrorFileName = treeErrorFileName # The output file name
    self.writeTr3e         = writeTre3         # Bool for if you want to write out all of the nodeValues and nodeDecisions for the nEstimators
    self.isContinuing      = isContinuing      # Boolean stating if this is a continuation of a previous premature ending of running a BDT
    self.treeError         = []                # List containing the tree estimator and it's error
    self.allNodeValues     = []                # List of tuples where tuple = (nEstimator, nodeValues for nEstimator)
    self.allNodeDecisions  = []                # List of tuples where tuple = (nEstimator, nodeDecisions for nEstimator)
    if self.isContinuing == True and self.lastCompletedEstimator != '' and lastCompletedEstimator < self.nEstimators: 
      with open(continuedTreeErrorFile= + ".csv") as treeErrorFile:
        treeErrorFileReader = csv.reader(treeErrorFile)
        next(treeErrorFileReader)
        self.treeError = [tuple(line) for line in treeErrorFileReader]
      self.currEst = self.treeError[-1][0] # Setting the current estimator to the last completed estimator if continuing
    else:
      self.currEst = nEstimators           # Otherwise, the current Estimator is the nEstimators          

  def GetBoostingTreesErrorsAndWeights(self, df, df_weights):
    '''
    Gets the boosted Tree's errors and weights. First, it imports current weights, if continuing a prematurely ended run. 
    Then it selects a random portion of columns and rows. Then it makes a tree. Then, it gets the tree error and stores it
    as a tuple with the current esimator. Next, it alters the weights according to if classified correctly.
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

        if self.printOutput: print ("###########################\n###########################\n  STARTING ESTIMATOR", self.currEst, "\n###########################\n###########################")
        TEMP = self.BuildTree(df=dfCurr, df_weights=dfCurr_weights) 
        if TEMP == "ERROR": raise UnboundLocalError("\n\nRunning Ended Early")
        self.GatherDecisions(df=dfCurr, df_weightsdfCurr_weights)
        self.treeError.append( (self.currEst, self.GetTreeError(df=dfCurr, df_weights=dfCurr_weights ) ) )
        dfCurr_weights = self.AlterWeights(df=dfCurr, df_weights=dfCurr_weights, error=next(i[1] for i in treeError if i[0] == self.currEst) ) 

        if self.printOutput: print ("BEFORE: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )
        IDs = dfCurr_weights[self.idColumn].tolist() 
        df_weights.loc[ df_weights[self.idColumn].isin( IDs ), 'Weights'] = dfCurr_weights['Weights'] 
        if self.printOutput: print ("After: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )

        if self.printOutput:
          for ite in df_weights['Weights'].unique():
            print ("\tlen('Weights' ==", ite, ")=", len(df_weights[ df_weights['Weights'] == ite].index) )
        df_weights['Weights'] = df_weights['Weights'] / df_weights['Weights'].sum(axis=0) 
        if self.printOutput: print ("\n\n####################\nFINAL SCORE FOR TREE #", self.currEst, "is 1-error=", 1-next(i[1] for i in treeError if i[0] == self.currEst),"\n##############")
        self.allNodeValues.append( (self.currEst, self.nodeValues) )
        self.allNodeDecisions.append( (selfcurrEst, self.nodeDecisions) )
        if self.writeTree:
          self.nodeValuesFileName    =  self.nodeValuesFileName.rstrip('1234567890')    + str(self.currEst)
          self.nodeDecisionsFileName =  self.nodeDecisionsFileName.rstrip('1234567890') + str(self.currEst)
          self.WritingNodeValues()
          self.WriteDecisions(df=df, df_weights=df_weights)
        self.currEst -= 1 
        self.CleanTree()

    
      if self.writeTree:
        # Writing the TreeErrors
        currTreeErrorFileName =  self.treeErrorFileName + ".csv"
        if os.path.isfile(currTreeErrorFileName): treeErrorFile = open(currTreeErrorFileName, 'a')
        else: treeErrorFile = open(currTreeErrorFileName, 'w')
        treeErrorFileCSV=csv.writer(treeErrorFile)
        treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
        for tup in self.treeError:
          treeErrorFileCSV.writerow(tup)
  
    except (KeyboardInterrupt,UnboundLocalError): #If user stops runnig, still save 
      if self.writeTree:
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


  def GetTreeError(self, df, df_weights):
    '''
    Gets the error of the tree, using each row's weight. It checks to see if each decision in the node decisions file
    applies to the LT group, the GT group, or both based upon the tuple in the nodeDecision File. It's tuples look like this:
    (Node number, LT class decision, GT class decision, LT Correct weight, GT Correct weight, LT total weight, GT total weight)
    (    0      ,         1        ,         2        ,         3        ,         4        ,        5       ,        6       )
    Returns the percentage of correct weight, unless the correct is greater than the total. This means a massive error and returns
    the value -1000000.
    '''
    if self.printOutput: print ("\n##############\n Getting Tree Error\n ###################")
    totalCorrWeight = 0
    sumWeights = 0
    for decisionTup in self.nodeDecisions: 
      if self.printOutput: print ("\ndecisionTup=", decisionTup)

      if pd.isnull(float(decisionTup[4]) ): 
        totalCorrWeight += float(decisionTup[3])
        sumWeights += float(decisionTup[5])

      elif pd.isnull(float(decisionTup[3]) ): 
        totalCorrWeight += float(decisionTup[4]) 
        sumWeights += float(decisionTup[6])

      else: 
        totalCorrWeight += float(decisionTup[3]) + float(decisionTup[4])
        sumWeights += float(decisionTup[5]) + float(decisionTup[6])
      if self.printOutput: print("\ttotalCorrWeight=", totalCorrWeight)
    if self.printOutput: print ("totalCorrWeight=", totalCorrWeight , "\tsumWeights=", sumWeights, "\terror=", 1 - (totalCorrWeight/sumWeights) )
    return float(1 - (totalCorrWeight / sumWeights)) if sumWeights >= totalCorrWeight else -100000 
 

  def AlterWeights(self, df, df_weights, error):
    '''
    Given the df, the error of the current tree, and the rate of change, find and alter the weights of the dataframe entries
    based upon their correctness. "alpha" is the exponent facto for weight adjustment. It checks if each node is an EndNode,
    which it then alters uniformly. It then checks if a LT and GT decision exists, and if so, alters their weights. 
    '''
    alpha = .5 * math.log1p((1 - error) / error) * self.rateOfChange 
    if self.printOutput: print ("error=", error, "\talpha*rateOfChange=", alpha, "\tCorrectFactor=", math.exp(-1*alpha), "\tIncorrectFactor=", math.exp(1*alpha)  )
    for decisionTup in self.nodeDecisions:
      dfIDs      = next(iteTup[1] for iteTup in self.nodeDFIDs  if int(iteTup[0]) == int(decisionTup[0]) )
      nodeValuesTup = next(iteTup for iteTup in self.nodeValues if int(iteTup[0]) == int(decisionTup[0]) )
      #nodeValuesTup = (int(nodeValuesTup[0]), float(nodeValuesTup[1]), str(nodeValuesTup[2]), float(nodeValuesTup[3]), float(nodeValuesTup[4]) )
      if nodeValuesTup[2] == "ThisIsAnEndNode": 
        df_weights.loc[ df_weights[self.idColumn].isin(dfIDs), 'Weights'] = df_weights[ df_weights[self.idColumn].isin(dfIDs)]['Weights'] * math.exp(-1*alpha)
        continue
      if not pd.isnull(float(decisionTup[2]) ):
        gtCorrIDs  = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] == int(decisionTup[2]) )][self.idColumn].tolist() 
        gtWrongIDs = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] != int(decisionTup[2]) )][self.idColumn].tolist() 
        df_weights.loc[df_weights[self.idColumn].isin(gtCorrIDs ), 'Weights'] = df_weights[df_weights[self.idColumn].isin(gtCorrIDs )]['Weights'] * math.exp(-1*alpha) 
        df_weights.loc[df_weights[self.idColumn].isin(gtWrongIDs), 'Weights'] = df_weights[df_weights[self.idColumn].isin(gtWrongIDs)]['Weights'] * math.exp( 1*alpha) 
      if not pd.isnull(float(decisionTup[1]) ): 
        ltCorrIDs  = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] == int(decisionTup[1]) )][self.idColumn].tolist()
        ltWrongIDs = df[ (df[self.idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] != int(decisionTup[1]) )][self.idColumn].tolist()
        df_weights.loc[df_weights[self.idColumn].isin(ltCorrIDs ), 'Weights'] = df_weights[df_weights[self.idColumn].isin(ltCorrIDs )]['Weights'] * math.exp(-1*alpha) 
        df_weights.loc[df_weights[self.idColumn].isin(ltWrongIDs), 'Weights'] = df_weights[df_weights[self.idColumn].isin(ltWrongIDs)]['Weights'] * math.exp( 1*alpha)
    return df_weights
  
  def CalssifyWithBoost(df_test, boostAnswersFileName):
    '''
    Classify a test set with the estimators in the BDT. It iterates through the nEstimators and reads in the file 
    with the node decisions and values. Then it uses the tree error to calculate the weight of the decision for 
    each estimator. To classify with each estimator, it goes through all the nodes, reading in the decision and cut
    value, and splitting the test set at each branch. It checks the node for being a BlankNode or EndNode. If it's not
    at the max tree depth, it checks if a decision exists for the LT, GT, or both groups. If at max tree depth, then 
    get decisions. To classify, the weight (alpha) of each tree is added to the class it was labeled as.
    '''
    if self.printOutput: print ("\n\n######################################\n Classifying Boosted Tree\n################################################")
    df_Answers = df_test.filter([self.idColumn], axis=1)
    df_Answers[self.className + "_total"] = 0.0 
    for classVal in self.uniqueClasses: 
      df_Answers[self.className + "_" + str(classVal)] = 0.0
    self.currEst = 1

    while self.currEst <= self.nEstimators: 
      if self.printOutput: print ("currEst=", self.currEst)
      self.nodeDecisions = next(decTup[1] for decTup in self.allNodeDecisions if decTup[0] == self.currEst)
      self.nodeValues    = next(valtup[1] for valTup in self.allNodeValues    if valTup[0] == self.currEst)
      currErrorTup = next(iteTup for iteTup in self.treeError if int(iteTup[0]) == self.currEst)
      alpha = .5 * math.log1p((1 - float(currErrorTup[1]) ) / float(currErrorTup[1]) ) 
      df_Answers[self.className + "_total"] += alpha
      self.testDFIDs = [ (0, df_test[self.idColumn].tolist()) ]

      for ite in self.nodeValues:
        if self.printOutput: print ("\n\tnodeValueTup=", nodeValueTup )
        dfCurr = df_test.loc[df_test[self.idColumn].isin(self.testDFIDs[nodeValueTup[0]][1])]
  
        if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]): 
          self.ClassifyWithBoost_BlankNode(nodeValueTup=nodeValueTup, alpha=alpha)

        elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: 
          self.ClassifyWithBoost_EndNode(df=dfCurr, df_Answers=df_Answers, nodeValueTup=nodeValueTup, alpha=alpha)

        elif nodeValueTup[0] < self.maxNodes / 2:
          self.ClassifyWithBoost_NotMaxDepth(df=dfCurr, df_Answers=df_Answers, nodeValueTup=nodeValueTup, alpha=alpha)

        else:
          self.ClassifyWithBoost_MaxDepthNode(nodeValueTup=nodeValueTup, df=dfCurr, df_Answers=df_Answers) 
      self.currEst += 1
      self.CleanTree()     
    self.ClassifyWithBoost_WriteAnswers(df_Answers=df_Answers)

  def ClassifyWithBoost_WriteAnswers(self, df_Answers):
    ''' 
    Given a df with columns for each classname, classify each element based upon the use of all estimators.
    Finds a probability for each class based upon the fraction of weight in each class. Works for >= 2 classes.
    It iterates through possible classes and finds the class with the highest weight fraction.
    ''' 
    df_Answers[self.className] = -1
    df_Answers[self.className + "_probability"] = -1
    totalWeight = df_Answers[self.className + "_total"]
    if self.printOutput: print (df_Answers )
    for classVal in self.uniqueClasses:
      df_Answers[self.className + "_" + str(classVal)] = df_Answers[self.className + "_" + str(classVal)] / totalWeight
      if self.printOutput: print ("\n\n\n\n", df_Answers.head(10) )
      df_Answers.loc[ df_Answers[self.className + "_" + str(classVal)] > df_Answers[self.className], self.className] = classVal 
      df_Answers.loc[ df_Answers[self.className] == classVal, self.className + "_probability"] = df_Answers[self.className + "_" + str(classVal)] 
    df_Answers.to_csv(boostAnswersFileName + "_Prob_Frac_ExtraInfo.csv", sep=',', index=False) 
    df_Answers[[self.idColumn, self.className]].to_csv(boostAnswersFileName + ".csv", sep=',', index=False) 
  
  def ClassifyWithBoost_BlankNode(self, nodeValueTup, alpha):
    '''
    If node is a BlankNode, then add BlankNodes for daughters.
    '''
    if self.printOutput: print ("\tdf=", self.testDFIDs[nodeValueTup[0]][1])
    if nodeValueTup[0] < self.maxNodes / 2:
      self.testDFIDs.append( (nodeValueTup[0]*2 + 1, [] ) )
      self.testDFIDs.append( (nodeValueTup[0]*2 + 2, [] ) )
  
  def ClassifyWithBoost_EndNode(self, df, df_Answers, nodeValueTup, alpha):
    '''
    If node is an EndNode, then classify points there and add BlankNode for daughters, if current node is not at the max tree depth
    '''
    decision = next(iteTup for iteTup in self.nodeDecisions if int(iteTup[0]) == nodeValueTup[0])
    IDs = dfCurr[self.idColumn].tolist()
    df_Answers.loc[ df_Answers[self.idColumn].isin(IDs), self.className + "_" + str(decision[1])] += alpha
    if self.printOutput: print ("\tdecision=", decision[1], "\talpha=", alpha )
    if nodeValueTup[0] < self.maxNodes / 2:
      self.testDFIDs.append( (nodeValueTup[0]*2 + 1, [] ) )
      self.testDFIDs.append( (nodeValueTup[0]*2 + 2, [] ) )
  

  def ClassifyWithBoost_NotMaxDepth(self, df, df_Answers, nodeValueTup, alpha):
    '''
    If node is not at the max tree depth, then add the split df ID's to self.testDFIDs. Then, check if there
    is a decision for the LT, GT, or both groups.
    ''' 
    if self.printOutput: print ("\tlen(lt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ]) )
    self.testDFIDs.append( (nodeValueTup[0]*2 + 1, dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist() ) )
    self.testDFIDs.append( (nodeValueTup[0]*2 + 2, dfCurr[ dfCurr[nodeValueTup[2]] >  nodeValueTup[3] ][self.idColumn].tolist() ) ) 
    if self.printOutput: print ("\tnodeValueTup[0]*2 + 1=", nodeValueTup[0]*2 + 1, "\n\tltDFIiDs=", len(dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist()) )

    try:  
      decision = next (itetup for itetup in self.nodeDecisions if int(itetup[0]) == nodeValueTup[0])
      if self.printOutput: print ("\tOne of this Node's Daughters is a BlankNode.")

      if pd.isnull(float(decision[2]) ) and not pd.isnull(float(decision[1]) ):
        ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist() 
        df_Answers.loc[ df_Answers[self.idColumn].isin(ltIDs) , self.className + "_" + str(decision[1])] += alpha
        if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs) )

      elif not pd.isnull(float(decision[2]) ) and pd.isnull(float(decision[1]) ):
        gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][self.idColumn].tolist() 
        df_Answers.loc[ df_Answers[self.idColumn].isin(gtIDs) , self.className + "_" + str(decision[2])] += alpha
        if self.printOutput:print ("\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )

      else:
        ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist()
        gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][self.idColumn].tolist() 
        df_Answers.loc[ df_Answers[self.idColumn].isin(ltIDs) , self.className + "_" + str(decision[1])] += alpha
        df_Answers.loc[ df_Answers[self.idColumn].isin(gtIDs) , self.className + "_" + str(decision[2])] += alpha 
        if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
    except StopIteration:  
     if self.printOutput: print ("Non of this node's daughters are Blank Nodes")


  def ClassifyWithBoost_MaxDepthNode(self, df, df_Answers, nodeValueTup, alpha):
    '''
    Given a node value tuple and knowing that it is a max tree depth node, classify the df entries
    '''
    decision = next(iteTup for iteTup in self.nodeDecisions if int(iteTup[0]) == nodeValueTup[0])
    ltIDs = df[ df[nodeValueTup[2]] <= nodeValueTup[3] ][self.idColumn].tolist()
    gtIDs = df[ df[nodeValueTup[2]] >  nodeValueTup[3] ][self.idColumn].tolist()
    df_Answers.loc[ df_Answers[self.idColumn].isin(ltIDs), self.className + "_" + decision[1]] += alpha
    df_Answers.loc[ df_Answers[self.idColumn].isin(gtIDs), self.className + "_" + decision[2]] += alpha
    if self.printOutput: print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[1], "\tlen(gtIDs)=",  len(gtIDs) )
 
