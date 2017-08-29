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
import scipy.stats as st

###################################################################
# Returns a list of tuples describing the test point's membership.
# Duality defines dual membership range. Linearly here
# linearly tup = (node number, membership value at node)
###################################################################
def FuzzyMembershipLinear(value, split, splitLength, duality,  previousList, nodeNumber, daughterEndNode=None):
  if daughterEndNode == 'LT': # Checking if the LT node is an end node. If so, keep the membership % for the current node, not the daughter
    ltNodeNumber = nodeNumber 
    gtNodeNumber = nodeNumber*2 + 2
  elif daughterEndNode == 'GT': # Checking if hte GT node is an end node. If so, keep the membership % for the current node, not the daughter
    gtNodeNumber = nodeNumber 
    ltNodeNumber = nodeNumber*2 + 1
  elif daughterEndNode == 'BOTH': #Checking if both nodes are end nodes. If so, then skip this function and keep the membership all for the mother
    return previousList 
  else:      #Else Neither nodes are end nodes
    gtNodeNumber = nodeNumber*2 + 2
    ltNodeNumber = nodeNumber*2 + 1


  parentTup = [tup for tup in previousList if int(tup[0]) == int(nodeNumber)] # Get mother membership
  previousList.remove( parentTup[0])  # Remove mother membership
  membership = []
  if duality == 0:
    if   value > (split):  membership.append( (gtNodeNumber, 1.0*parentTup[0][1]) ) #If greater than the range of dual memebership, all weight goes to GT
    elif value <= (split):  membership.append( (ltNodeNumber, 1.0*parentTup[0][1]) ) #If less than the range of dual membership, all weight goes to LT
    return previousList + membership 
  if   value > (split + splitLength*duality):  membership.append( (gtNodeNumber, 1.0*parentTup[0][1]) ) #If greater than the range of dual memebership, all weight goes to GT
  elif value < (split - splitLength*duality):  membership.append( (ltNodeNumber, 1.0*parentTup[0][1]) ) #If less than the range of dual membership, all weight goes to LT
  else:  
    percentGT = (value - split + splitLength*duality) / (2 * splitLength*duality) # Find percent of weight that is GT
    membership.append( (ltNodeNumber, (1.0-percentGT)*parentTup[0][1] ) ) # Save the membership for LT
    membership.append( (gtNodeNumber, percentGT*parentTup[0][1] ) )   # Save memberhsip % for GT
  return previousList + membership

#################################################
# Update class decision Score by taking percent
# of memberhship and put it all towards the 
# decision at the node it currently is a part of
#################################################
def FuzzyDecisionScoreUpdate(previous, membershipList, nodeNumber):
  nodeTup = [tup for tup in membershipList if int(tup[0]) == int(nodeNumber)] # Find the percent of membership in this node 
  return previous + nodeTup[0][1]  # Add the membership percent to the decision of this node dictated in calling the function

#############################################################
# Update Boosted class decision Score by taking precent
# of membership and applying it, weighted by the alpha
# of the current tree estimator, to the decision fo the node
#############################################################
def BoostedFuzzyDecisionScoreUpdate(previous, membershipList, nodeNumber, alpha):
  nodeTup = [tup for tup in membershipList if int(tup[0]) == int(nodeNumber)] # Find the percent of membership in this node 
  return previous + nodeTup[0][1] * alpha  # Add the membership percent to the decision of this node dictated in calling the function

################################
# Update MembershipNodeList to
# include all membership nodes 
# in the current membershiplist
################################
def FuzzyUpdateMembershipNodeList(membershipList):
  return [tup[0] for tup in membershipList]

####################################################################
# Given a final Tree described by the nodes and their tple values
# described above and the decisions of those nodes,  make decisions
# of a set of points in a DF
####################################################################
def ClassifyWithFuzzyTree(df_test, className, idColumn, maxDepth, duality, uniqueClasses, outputFileName, nodeDecisionsFileName, nodeValuesFileName):
  print ("\n\n########################################################################\n Classifying test points with Tree from Make Tree\n########################################################################")
  df_Answers = df_test.filter([idColumn], axis=1)
  for classVal in uniqueClasses:
    df_test[className + "_" + str(classVal)] = 0.0 # Create new column for sum of alpha's for each unique className value
  df_test['Memberships'] = [ [(0, 1.0)] for _ in range( len(df_test)) ] # Initialize all points to be a part of node 0 wiht complete percent membership there
  df_test['MembershipNodeList'] = [ [0] for _ in range( len(df_test)) ] # Initialize all points to have their list of nodes they are a part of be only node 0
  with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeValuesFileName + ".csv") as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
    nodeValues = [tuple(line) for line in nodeValuesFileReader]

  maxNodeCount = 0
  for i in range(1,maxDepth+1): maxNodeCount += 2**i # Get the max number of nodes based upon maxDepth
  for ite in nodeValues: #Iterate through the nodes
    nodeValueTup = (int(ite[0]),  float(ite[1]), ite[2], float(ite[3]), float(ite[4])) # The File stores all info as strings. Cleaner to reassing type now, not every instance.
    print ("\n\tnodeValueTup=", nodeValueTup )
    df_Curr = df_test[ df_test['MembershipNodeList'].apply(lambda x: True if nodeValueTup[0] in x else False) ].copy() # Get the nodes that have a membership in current node number

    if pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[2] == '' and pd.isnull(nodeValueTup[1]): # If decision of node from MakeTree is blank, then skip
      print ("\tlen(df)=", len(df_Curr.index) )
      continue
    elif nodeValueTup[2] == 'ThisIsAnEndNode' and pd.isnull(nodeValueTup[3]) and pd.isnull(nodeValueTup[4]) and nodeValueTup[1] == 1.0: # If decision of node from MakeTree is an EndNode, then proceed
      decision = next (itetup for itetup in nodeDecisions if int(itetup[0]) == nodeValueTup[0]) # Get the decision at this node
      IDs = df_Curr[idColumn].tolist() # Get the IDs of the points at this node
      if len(IDs) > 0: # Check if there are some test points  here
        df_test_Changed = df_test[ df_test[idColumn].isin(IDs)].copy() # Create dummy df
        df_test_Changed[className + "_" + str(decision[1])] = df_test_Changed.apply(lambda row: 
                        FuzzyDecisionScoreUpdate( previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0]), axis=1) # Add percent to decision
        df_test.loc[ df_test[idColumn].isin(IDs), className + "_" + str(decision[1])] = df_test_Changed[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
    elif nodeValueTup[0] < maxNodeCount / 2: # If element isn't an EndNode and also not at the furthest depth, then proceed
      daughterEndNodeCheck = None # This is a check if the node is and end node. Possible values are 'LT', 'GT', and 'BOTH'
      try:  # This sees if the decision of a node is already added. From a sister BlankNode
        decision = next (itetup for itetup in nodeDecisions if int(itetup[0]) == nodeValueTup[0]) # Try to find decision. If no decision, node's daugthers are not blank nodes. Proceed to excpetion
        print ("\tOne of this Node's Daughters is a BlankNode.")
        if pd.isnull(float(decision[2]) ) and not pd.isnull(float(decision[1]) ): # If the LT daughter node is the Blank node
          daughterEndNodeCheck = 'LT'
          ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          if len(ltIDs) > 0: # Check if there are some test points  here
            df_test_Changed = df_test[ df_test[idColumn].isin(ltIDs)].copy()  # Create dummy df
            df_test_Changed[className + "_" + str(decision[1])] = df_test_Changed.apply(lambda row: 
                            FuzzyDecisionScoreUpdate( previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0]), axis=1) # Add percent to dec.
            df_test.loc[ df_test[idColumn].isin(ltIDs), className + "_" + str(decision[1])] = df_test_Changed[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
          print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs) )
        elif not pd.isnull(float(decision[2]) ) and pd.isnull(float(decision[1]) ): # If the GT daughter node is the Blank Node
          daughterEndNodeCheck = 'GT'
          gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          if len(gtIDs) > 0: # Check if there are some test points  here
            df_test_Changed = df_test[ df_test[idColumn].isin(gtIDs)].copy()  # Create dummy df
            df_test_Changed[className + "_" + str(decision[2])] = df_test_Changed.apply(lambda row: 
                            FuzzyDecisionScoreUpdate( previous=row[className + "_" + str(decision[2])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0]), axis=1) # Add percent to dec
            df_test.loc[ df_test[idColumn].isin(gtIDs), className + "_" + str(decision[2])] = df_test_Changed[className + "_" + str(decision[2])] # Set updated decisions to non-copy df_test
          print ("\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
        else:   # Only other option for there to be a decision and not LT or GT is for BOTH GT and LT to be blank nodes
          daughterEndNodeCheck = 'BOTH'
          ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
          gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter GT leaf of current Node
          if len(ltIDs) > 0: # Check if there are some test points  here
            df_test_Changed_lt = df_test[ df_test[idColumn].isin(ltIDs)].copy()  # Create dummy df
            df_test_Changed_lt[className+"_" + str(decision[1])] = df_test_Changed_lt.apply(lambda row: 
                               FuzzyDecisionScoreUpdate( previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0]), axis=1) # Add percent to dec
            df_test.loc[ df_test[idColumn].isin(ltIDs), className + "_" + str(decision[1])] = df_test_Changed_lt[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
          if len(gtIDs) > 0: # Check if there are some test points  here
            df_test_Changed_gt = df_test[ df_test[idColumn].isin(gtIDs)].copy()  # Create dummy df
            df_test_Changed_gt[className+"_" + str(decision[2])] = df_test_Changed_gt.apply(lambda row: 
                               FuzzyDecisionScoreUpdate( previous=row[className + "_" + str(decision[2])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0]), axis=1) # Add percent to dec
            df_test.loc[ df_test[idColumn].isin(gtIDs), className + "_" + str(decision[2])] = df_test_Changed_gt[className + "_" + str(decision[2])] # Set updated decisions to non-copy df_test
          print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
      except StopIteration:
        print ("Non of this node's daughters are Blank Nodes")
      df_Curr['Memberships'] = df_Curr.apply(lambda row: 
                                      FuzzyMembershipLinear( value=row[nodeValueTup[2]], split=nodeValueTup[3], splitLength=nodeValueTup[4], duality=duality,
                                      previousList=row['Memberships'], nodeNumber=nodeValueTup[0], daughterEndNode=daughterEndNodeCheck ), axis=1 ) # Sends points membership via linear range
      df_Curr['MembershipNodeList'] = df_Curr.apply(lambda row: FuzzyUpdateMembershipNodeList(row['Memberships']), axis=1) # Updates Membership's Node list depending on the points membership
      IDs = df_Curr[idColumn].tolist() # Get the df_test ID's in the
      df_test.loc[ df_test[idColumn].isin(IDs), 'Memberships'] = df_Curr['Memberships'] # Update df_test 'Membership' with df_Curr's values
      df_test.loc[ df_test[idColumn].isin(IDs), 'MembershipNodeList'] = df_Curr['MembershipNodeList'] #Update df_test 'MembershipNodeList' with df_Curr's values
    else: # If not an EndNode, BlankNode, or a node NOT at the max depth, then get decisions there
      decision = next(iteTup for iteTup in nodeDecisions if int(iteTup[0]) == nodeValueTup[0]) # Get decision of Make Tree at node
      ltIDs = df_Curr[ df_Curr[nodeValueTup[2]] <= nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
      gtIDs = df_Curr[ df_Curr[nodeValueTup[2]] > nodeValueTup[3] ][idColumn].tolist() # Get the df_test ID's in the daughter LT leaf of current Node
      if len(ltIDs) > 0 : # Check if there are some test points  here
        df_test_Changed_lt = df_test[ df_test[idColumn].isin(ltIDs)].copy() # Create dummy df
        df_test_Changed_lt[className + "_" + str(decision[1])] = df_test_Changed_lt.apply(lambda row: 
                           FuzzyDecisionScoreUpdate(  previous=row[className + "_" + str(decision[1])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0]), axis=1) # Add percent to dec
        df_test.loc[ df_test[idColumn].isin(ltIDs), className + "_" + str(decision[1])] = df_test_Changed_lt[className + "_" + str(decision[1])] # Set updated decisions to non-copy df_test
      if len(gtIDs) > 0: # Check if there are some test points  here
        df_test_Changed_gt = df_test[ df_test[idColumn].isin(gtIDs)].copy() # Create dummy df
        df_test_Changed_gt[className + "_" + str(decision[2])] = df_test_Changed_gt.apply(lambda row: 
                           FuzzyDecisionScoreUpdate(  previous=row[className + "_" + str(decision[2])], membershipList=row['Memberships'], nodeNumber=nodeValueTup[0]), axis=1) # Add percent to dec
        df_test.loc[ df_test[idColumn].isin(gtIDs), className + "_" + str(decision[2])] = df_test_Changed_gt[className + "_" + str(decision[2])] # Set updated decisions to non-copy df_test
      print ("\tClass for LT=", decision[1], "\tlen(ltIDs)=",  len(ltIDs), "\tClass for GT=", decision[2], "\tlen(gtIDs)=",  len(gtIDs) )
  #Writing the answers out
  df_Answers[className] = -1
  df_Answers[className + "_probability"] = -1
  for classVal in uniqueClasses:
    print ("classVal=", classVal, "\ndf_test[className + '_' + str(classVal):\n", df_test[className + "_" + str(classVal)] )
    df_Answers[className + "_" + str(classVal)] = df_test[className + "_" + str(classVal)]
    df_Answers.loc[ df_Answers[className + "_" + str(classVal)] > df_Answers[className], className] = classVal # if current classVal prob is greater, reassign answer to the classVal
    df_Answers.loc[ df_Answers[className] == classVal, className + "_probability"] = df_Answers[className + "_" + str(classVal)] # If current classVal prob got changed, reassign probab
  df_Answers.to_csv(outputFileName + "_Prob_Frac_ExtraInfo.csv", sep=',', index=False) #Write out the answers with all answer information
  df_Answers[[idColumn, className]].to_csv(outputFileName + ".csv", sep=',', index=False) #Write out the answers

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
 


 
