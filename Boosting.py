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
import os.path
sys.path.insert(0, '/home/kyletos/Kaggle/Algorithms/FuzzyTreeProject/')
from FuzzyTrees_Building import  *

####################################################################
# This takes dict of parameters for MakeTree, the number of trees
# desired and makes that many of the normally produced files in 
# Make Trees. TreeErrors are also written based on tree correctness
####################################################################
def GetBoostingTreesErrorsAndWeights(df, nEstimators, rateOfChange, df_weights, paramDict, colRandomness, rowRandomness, treeErrorFileName, middleStart=False, middleStartNum=-1):
  if colRandomness > 1 or colRandomness < 0: print ("Give a colRandomness between 0-1 for the fraction of columns to be removed.") 
  if rowRandomness > 1 or rowRandomness < 0: print ("Give a rowRandomness between 0-1 for the fraction of row to be removed.") 
  if middleStart == False or middleStartNum == -1: currEst = nEstimators 
  else: 
    currEst = middleStartNum
    df_weights = pd.read_csv("Answers/DF_WEIGHTS.csv")
  treeError = []
  try:
    while currEst > 0: # Make a Tree for each of the desired estimators. Start at nEstimators and go down, so if run is stopped, then you can readjust the number to get the original # of Trees
      columnsList = [ ite for ite in df.columns.tolist() if ite != paramDict['className'] and ite != paramDict['idColumn'] ]
      columnsList.extend((paramDict['className'], paramDict['idColumn']) )
      dfCurr = df[ random.sample(columnsList, math.ceil(len(columnsList) * (1-colRandomness) ) ) ].copy() #Selecting a random portion of columns like a randomForest
      dfCurr[paramDict['className'] ] = df[paramDict['className']]
      dfCurr[paramDict['idColumn'] ] = df[paramDict['idColumn']]
      dfCurr = dfCurr.sample(math.ceil(len(dfCurr.index) * (1-rowRandomness) ) ) #Selecting random portion of rows for double randomness
      dfCurr_weights = df_weights[ df_weights[paramDict['idColumn']].isin(dfCurr[paramDict['idColumn']].tolist() )].copy()
      #dfCurr_NotIn = df[ ~dfCurr[idColumn].isin(dfCurr[idColumn].tolist()) ].copy()
      print (paramDict['nodeDecisionsFileName'].rstrip('1234567890') )
      paramDict['nodeDecisionsFileName'] = paramDict['nodeDecisionsFileName'].rstrip('1234567890') + str(currEst) # Add currEst to Tree names to differentiate solutions to each estimator
      paramDict['nodeValuesFileName'] = paramDict['nodeValuesFileName'].rstrip('1234567890') + str(currEst)
      print ("#############################################\n#############################################\n  STARTING ESTIMATOR", currEst, "\n#############################################\n#############################################")
      TEMP = MakeFuzzyTree(df=dfCurr, df_weights=dfCurr_weights, **paramDict) # Make the Tree for currEst
      if TEMP == "ERROR": raise UnboundLocalError("\n\nRunning Ended Early")
      treeError.append( (currEst, GetTreeError(df=dfCurr, className=paramDict['className'], df_weights=dfCurr_weights, #Getting error for MakeTree currEst
                                               idColumn=paramDict['idColumn'], nodeDecisionsFileName=paramDict['nodeDecisionsFileName'] ) ) )
      dfCurr_weights = AlterWeights(df=dfCurr, df_weights=dfCurr_weights, error=next(i[1] for i in treeError if i[0] == currEst), rateOfChange=rateOfChange, idColumn=paramDict['idColumn'],  
                                    className=paramDict['className'], nodeDecisionsFileName=paramDict['nodeDecisionsFileName'], nodeValuesFileName=paramDict['nodeValuesFileName']) 
      print ("BEFORE: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )
      IDs = dfCurr_weights[paramDict['idColumn']].tolist()  # Getting list of IDs of data points included from rowRandomness
      df_weights.loc[ df_weights[paramDict['idColumn']].isin( IDs ), 'Weights'] = dfCurr_weights['Weights'] # Changing the official weights of those points included fromt he rowRandomness
      print ("After: df_weights['Weights'].unique()=", df_weights['Weights'].unique() )
      for ite in df_weights['Weights'].unique():
        print ("\tlen('Weights' ==", ite, ")=", len(df_weights[ df_weights['Weights'] == ite].index) )
      df_weights['Weights'] = df_weights['Weights'] / df_weights['Weights'].sum(axis=0) # Normalizing weights to 1 after changing them
      print ("\n\n###############################################################\nFINAL SCORE FOR TREE #", currEst, "is 1-error=", 1-next(i[1] for i in treeError if i[0] == currEst),"\n###############################################################")
      currEst -= 1 
  
    # Writing the TreeErrors
    treeErrorFileName =  treeErrorFileName + ".csv"
    if os.path.isfile(treeErrorFileName): treeErrorFile = open(treeErrorFileName, 'a')
    else: treeErrorFile = open(treeErrorFileName, 'w')
    treeErrorFileCSV=csv.writer(treeErrorFile)
    treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
    for tup in treeError:
      treeErrorFileCSV.writerow(tup)

  except (KeyboardInterrupt,UnboundLocalError): #If user stops runnig, still save 
      treeErrorFileName =  treeErrorFileName + "_Incomplete.csv"
      if os.path.isfile(treeErrorFileName): 
        treeErrorFile = open(treeErrorFileName, 'a')
        treeErrorFileCSV=csv.writer(treeErrorFile)
      else: 
        treeErrorFile = open(treeErrorFileName, 'w')
        treeErrorFileCSV=csv.writer(treeErrorFile)
        treeErrorFileCSV.writerow(["NumberOfEstimator,TreeErrorAssociatedWithCorrectness"])
      for tup in treeError:
        treeErrorFileCSV.writerow(tup)
      df_weights.to_csv("Answers/DF_WEIGHTS.csv", sep=',', index=False) 

   
      

########################################################
# Takes Set of NodeDecisions and uses the nCorr to get
# the weight of the correctness
########################################################
def GetTreeError(df, className, df_weights, idColumn, nodeDecisionsFileName):
  print ("\n##############\n Getting Tree Error\n ###################")
  with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]

  totalCorrWeight = 0
  sumWeights = 0
  for decisionTup in nodeDecisions: # iterate over the decisions to add the correctly identified one to "totalCorrWeight"
    print ("\ndecisionTup=", decisionTup)
    if pd.isnull(float(decisionTup[4]) ): # If the decision was only made for the lt group or if all the elements in the node had the same className (End-Node)
      print ("totalCorrWeight=", totalCorrWeight , "\tfloat(decisionTup[3])=", float(decisionTup[3]))
      totalCorrWeight += float(decisionTup[3]) # Add correct weights to the total
      sumWeights += float(decisionTup[5])
    elif pd.isnull(float(decisionTup[3]) ): # I the decision was only made for the GT group
      print ("totalCorrWeight=", totalCorrWeight , "\tfloat(decisionTup[4])=", float(decisionTup[4]))
      totalCorrWeight += float(decisionTup[4]) # Add correct weights to the total
      sumWeights += float(decisionTup[6])
    else: # Add the weights of the correctly assign to totalCorrweights
      print ("totalCorrWeight=", totalCorrWeight , "\tfloat(decisionTup[3])=", float(decisionTup[3]), "\tfloat(decisionTup[4])=", float(decisionTup[4]))
      totalCorrWeight += float(decisionTup[3]) + float(decisionTup[4]) 
      sumWeights += float(decisionTup[5]) + float(decisionTup[6])
    print("\ttotalCorrWeight=", totalCorrWeight)
  print ("totalCorrWeight=", totalCorrWeight , "\tsumWeights=", sumWeights, "\terror=", 1 - (totalCorrWeight/sumWeights) )
  nodeDecisionsFile.close()
  del nodeDecisionsFileReader
  return float(1 - (totalCorrWeight / sumWeights))

###########################################################
# Given the df, and the nodeDecisions, node DF IDs and 
# the nodeValues, alter the weights based upon correctness
###########################################################
def AlterWeights(df, df_weights, error, idColumn, rateOfChange, className, nodeDecisionsFileName, nodeValuesFileName):
  with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeValuesFileName + ".csv") as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
    nodeValues = [tuple(line) for line in nodeValuesFileReader]
  alpha = .5 * math.log1p((1 - error) / error) * rateOfChange # exponent factor for adjustment of weights
  print ("error=", error, "\talpha*rateOfChange=", alpha, "\tCorrectFactor=", math.exp(-1*alpha), "\tIncorrectFactor=", math.exp(1*alpha)  )
  for decisionTup in nodeDecisions:
    dfIDs = df[ df['MembershipNodeList'].apply(lambda x: True if int(decisionTup[0]) in x else False) ][idColumn].tolist() # Get the nodes that have a membership in current node number
    nodeValuesTup = next(iteTup for iteTup in nodeValues if int(iteTup[0]) == int(decisionTup[0]) )
    nodeValuesTup = (int(nodeValuesTup[0]), float(nodeValuesTup[1]), str(nodeValuesTup[2]), float(nodeValuesTup[3]), float(nodeValuesTup[4]) )
    if nodeValuesTup[2] == "ThisIsAnEndNode": # If the node is all the same class, all the weights get reduced from being all correct
      df_weights.loc[ df_weights[idColumn].isin(dfIDs), 'Weights'] = df_weights[ df_weights[idColumn].isin(dfIDs)]['Weights'] * math.exp(-1*alpha)
      continue
    if not pd.isnull(float(decisionTup[2]) ): # If the GT part of the node has a decision, change those accordingly
      gtCorrIDs  = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] == int(decisionTup[2]) )][idColumn].tolist() # Get correctly Id'd Id's in GT group
      gtWrongIDs = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] >  nodeValuesTup[3]) & (df[className] != int(decisionTup[2]) )][idColumn].tolist() # GEt incorrectly ID'd ID's in GT group
      df_weights.loc[df_weights[idColumn].isin(gtCorrIDs ), 'Weights'] = df_weights[df_weights[idColumn].isin(gtCorrIDs )]['Weights'] * math.exp(-1*alpha) # Make weights of correct ones less
      df_weights.loc[df_weights[idColumn].isin(gtWrongIDs), 'Weights'] = df_weights[df_weights[idColumn].isin(gtWrongIDs)]['Weights'] * math.exp( 1*alpha) # Make weights of incorrect ones more
    if not pd.isnull(float(decisionTup[1]) ): # If the LT part of the node has a decision, change those accordingly
      ltCorrIDs  = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] == int(decisionTup[1]) )][idColumn].tolist() # Get correctly ID'd ID's in LT group
      ltWrongIDs = df[ (df[idColumn].isin(dfIDs)) & (df[nodeValuesTup[2]] <= nodeValuesTup[3]) & (df[className] != int(decisionTup[1]) )][idColumn].tolist() # Get incorrectly ID'd ID's in LT group
      df_weights.loc[df_weights[idColumn].isin(ltCorrIDs ), 'Weights'] = df_weights[df_weights[idColumn].isin(ltCorrIDs )]['Weights'] * math.exp(-1*alpha) # Make weights of correct ones less
      df_weights.loc[df_weights[idColumn].isin(ltWrongIDs), 'Weights'] = df_weights[df_weights[idColumn].isin(ltWrongIDs)]['Weights'] * math.exp( 1*alpha) # Make weights of incorrect ones more
  nodeDecisionsFile.close()
  del nodeDecisionsFileReader
  nodeValuesFile.close()
  del nodeValuesFileReader
  return df_weights

#############################################################
# For nEstimator trees, give the overall class decisions and
# the probability of point based on correctness. Should work 
# With >= 2 different classes, but haven't tested yet.
#############################################################
def CalssifyWithBoost(df_test, nEstimators, maxDepth, idColumn, className, uniqueClasses, treeErrorFileName, nodeValuesFileName, nodeDecisionsFileName, boostAnswersFileName):
  print ("\n\n########################################################################\n Classifying Boosted Tree\n######################################################################")
  df_Answers = df_test.filter([idColumn], axis=1)
  df_Answers[className + "_total"] = 0.0 # Total sum of the alphas over the nEstimator trees
  for classVal in uniqueClasses: 
    df_Answers[className + "_" + str(classVal)] = 0.0 # Create new column for sum of alpha's for each unique className value
  with open(treeErrorFileName + ".csv") as treeErrorFile: 
    treeErrorFileReader = csv.reader(treeErrorFile)
    next(treeErrorFileReader)
    treeError = [tuple(line) for line in treeErrorFileReader]
  currEst = 1
  while currEst <= nEstimators: # Loop over the nEstimators number of trees in boost
    print ("currEst=", currEst)
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
    df_Answers[className + "_total"] += alpha
    dfIDList = [ (0, df_test[idColumn].tolist()) ] # List of node population of df_test based on the MakeTree nodeDecisions
    maxNodeCount = 0
    for i in range(1,maxDepth+1): maxNodeCount += 2**i # Get the max number of nodes based upon maxDepth
    for ite in nodeValues:
      nodeValueTup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4]) ) # The File stores all info as strings. Cleaner to reassing type now instead of every instance.
      print ("\n\tnodeValueTup=", nodeValueTup )
      dfCurr = df_test.loc[df_test[idColumn].isin(dfIDList[nodeValueTup[0]][1])] #Get the population of df_test at current node

      if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]): # Check if node is a BlankNode
        print ("\tdf=", dfIDList[nodeValueTup[0]][1])
        if nodeValueTup[0] < maxNodeCount / 2: # If node is not at the maxDepth, then make blank placeholders for daughters in the iteration
          dfIDList.append( (nodeValueTup[0]*2 + 1, [] ) )
          dfIDList.append( (nodeValueTup[0]*2 + 2, [] ) )
      elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: #Check if node is an EndNode
        decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get decision of the EndNode
        IDs = dfCurr[idColumn].tolist() #Get the elements of df_test at this node
        df_Answers.loc[ df_Answers[idColumn].isin(IDs), className + "_" + str(decision[1])] += alpha #Apply the decision from the MakeTree to all df_test elements at present node
        print ("\tdecision=", decision[1], "\talpha=", alpha ) 
        if nodeValueTup[0] < maxNodeCount / 2: # If node is not at the maxDepth, then make blank placeholders for daughters in the iteration
          dfIDList.append( (nodeValueTup[0]*2 + 1, [] ) )
          dfIDList.append( (nodeValueTup[0]*2 + 2, [] ) )
      elif nodeValueTup[0] < maxNodeCount / 2: # If node is not a BlankNode or an EndNode and it is not at the max depth, then proceed
        print ("\tlen(lt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ]), "\tlen(gt)=", len(dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ]) )
        dfIDList.append( (nodeValueTup[0]*2 + 1, dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() ) ) # Add the LT group of df_test elements as daughter of node
        dfIDList.append( (nodeValueTup[0]*2 + 2, dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() ) )  # ADd the GT group of df_test elements as daughter of node
        print ("\tnodeValueTup[0]*2 + 1=", nodeValueTup[0]*2 + 1, "\n\tltDFIiDs=", len(dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist()) )
        print ("\tgtDFIDs=", len(dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist()) )
        try:  # This sees if the decision of a node is already added. From a sister BlankNode
          decision = next (itetup for itetup in nodeDecisions if int(itetup[0]) == nodeValueTup[0])
          print ("\tOne of this Node's Daughters is a BlankNode.")
          if pd.isnull(float(decision[2]) ) and not pd.isnull(float(decision[1]) ):
            ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
            df_Answers.loc[ df_Answers[idColumn].isin(ltIDs) , className + "_" + str(decision[1])] += alpha # Apply decision to LT group
            print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs) )
          elif not pd.isnull(float(decision[2]) ) and pd.isnull(float(decision[1]) ):
            gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
            df_Answers.loc[ df_Answers[idColumn].isin(gtIDs) , className + "_" + str(decision[2])] += alpha # Apply decision to LT group
            print ("\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
          else:
            ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
            gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
            df_Answers.loc[ df_Answers[idColumn].isin(ltIDs) , className + "_" + str(decision[1])] += alpha # Apply decision to LT group
            df_Answers.loc[ df_Answers[idColumn].isin(gtIDs) , className + "_" + str(decision[2])] += alpha # Apply decision to LT group
            print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
        except StopIteration:  
          print ("Non of this node's daughters are Blank Nodes")
      else: # If node is not a BlankNode, EndNode, and is at the maxDepth of tree, theni proceed
        decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get decision of node for the LT and GT groups
        ltIDs = dfCurr[ dfCurr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's that are LT of cut of final node
        gtIDs = dfCurr[ dfCurr[nodeValueTup[2]] >  nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's that are GT of cur of final node
        df_Answers.loc[ df_Answers[idColumn].isin(ltIDs), className + "_" + decision[1]] += alpha # Apply the decisions of LT group from nodeDecisions to LT group of df_test elements
        df_Answers.loc[ df_Answers[idColumn].isin(gtIDs), className + "_" + decision[2]] += alpha # Apply the decisions of GT group from nodeDecisions to GT group of df_test elements
        print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[1], "\tlen(gtIDs)=",  len(gtIDs) )
        del ltIDs, gtIDs, decision # Delete holders to preserve memory, if not already deleted by python
      del dfCurr 

    currEst += 1     
    nodeDecisionsFile.close()
    del nodeDecisionsFileReader
    nodeValuesFile.close()
    del nodeValuesFileReader 

  df_Answers[className] = -1
  df_Answers[className + "_probability"] = -1
  print (df_Answers )
  for classVal in uniqueClasses:
    df_Answers[className + "_" + str(classVal)] = df_Answers[className + "_" + str(classVal)] / df_Answers[className + "_total"] # Normalizing sums to total, to make a probability
    print ("\n\n\n\n", df_Answers.head(10) )
    df_Answers.loc[ df_Answers[className + "_" + str(classVal)] > df_Answers[className], className] = classVal # if current classVal prob is greater, reassign answer to the classVal
    df_Answers.loc[ df_Answers[className] == classVal, className + "_probability"] = df_Answers[className + "_" + str(classVal)] # If current classVal prob got changed, reassign probab
  df_Answers.to_csv(boostAnswersFileName + "_Prob_Frac_ExtraInfo.csv", sep=',', index=False) #Write out the answers with all answer information
  df_Answers[[idColumn, className]].to_csv(boostAnswersFileName + ".csv", sep=',', index=False) #Write out the answers

