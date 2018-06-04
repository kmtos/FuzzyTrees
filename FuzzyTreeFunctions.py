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


def GetCurrentNodeFuzziedWeight(memberships, nodeNumber):
  '''
  Takes the memberships of an entry and a node number, and returns the membership weight at that node.
  '''
  nodeTup = [tup for tup in memberships if int(tup[0]) == int(nodeNumber) ] 
  return nodeTup[0][1]  

def ReturnNodePointsToParent(memberships, nodeNumber):
  '''
  Returns the weight of a node to it's parent node. It gets the membership at a node, then gets the memberhship at the parent node. Then it adds the node weight to the parent weight.
  '''
  nodeTup = [tup for tup in memberships if int(tup[0]) == int(nodeNumber)] 
  memberships.remove( nodeTup[0])
  parentNodeNum = (nodeTup[0][0]-1)//2
  parentTup = [tup for tup in memberships if int(tup[0]) == int(parentNodeNum)]
  new = []
  if parentTup:
    memberships.remove(parentTup[0])
    new.append( (parentNodeNum, nodeTup[0][1] + parentTup[0][1]) )
    return new + memberships
  new.append( (parentNodeNum, nodeTup[0][1]) )
  return new + memberships

def FuzzyMembershipLinear(value, split, fuzzinessRange,  previousList, nodeNumber, daughterEndNode=None):
  '''
  This function uses the linear function and the level of fuzziness to determine how percentages are split between the two nodes after the cut is established.
  It checks these scenarios: If LT or GT are end node, then it keeps the end node's membership as it's parent's and splits the other group's membership weight.
  If both are end nodes, then all weight is kept as parent's weight. Other wise, it splits both the LT and GT group. It then either adds all weight to the LT 
  group, all to the GT group, or it splits between the two depending on the closeness to the cut value.
  '''
  if daughterEndNode == 'LT': 
    gtNodeNumber = nodeNumber*2 + 2
    ltNodeNumber = nodeNumber
  elif daughterEndNode == 'GT': 
    gtNodeNumber = nodeNumber
    ltNodeNumber = nodeNumber*2 + 1
  elif daughterEndNode == 'BOTH': 
    return previousList
  else:      
    gtNodeNumber = nodeNumber*2 + 2
    ltNodeNumber = nodeNumber*2 + 1

  parentTup = [tup for tup in previousList if int(tup[0]) == int(nodeNumber)] 
  previousList.remove( parentTup[0])  
  membership = []
  if fuzziness == 0:
    if   value > (split):  membership.append( (gtNodeNumber, 1.0*parentTup[0][1]) ) 
    elif value <= (split):  membership.append( (ltNodeNumber, 1.0*parentTup[0][1]) ) 
    return previousList + membership
  if   value > (split + fuzzinessRange):  membership.append( (gtNodeNumber, 1.0*parentTup[0][1]) ) 
  elif value < (split - fuzzinessRange):  membership.append( (ltNodeNumber, 1.0*parentTup[0][1]) ) 
  else:
    percentGT = (value - split + fuzzinessRange) / (2 * fuzzinessRange) 
    membership.append( (ltNodeNumber, (1.0-percentGT)*parentTup[0][1] ) ) 
    membership.append( (gtNodeNumber, percentGT*parentTup[0][1] ) )   
  return previousList + membership


def FuzzyUpdateMembershipNodeList(membershipList):
  '''
  Returns all the nodes in "Memberships" to update the "MembershipNodeList".
  '''
  return [tup[0] for tup in membershipList]

def FuzzyDecisionScoreUpdate(previous, membershipList, nodeNumber):
  '''
  This adds the fraction of memebership at a node to the previous amount of a column specified in the calling.
  '''
  nodeTup = [tup for tup in membershipList if int(tup[0]) == int(nodeNumber)] 
  return previous + nodeTup[0][1]  

def ChangeWeightWithRow(weight, moreOrLess, memberships, alpha, nodeNumber):
  '''
  This changes the weight of the entry in the dataframe. "weight" is the current weight. "moreOrLess" is the direction in which the weight is changed. "nodeTup" is a
  1 element list of tuples where the second element in the tuple is the membership weight at that node, which is a percentage of the total weight.
  '''
  nodeTup = [tup for tup in memberships if int(tup[0]) == int(nodeNumber) ]
  return weight * math.exp(alpha * moreOrLess * nodeTup[0][1])


def BoostedFuzzyDecisionScoreUpdate(previous, membershipList, nodeNumber, alpha):
  nodeTup = [tup for tup in membershipList if int(tup[0]) == int(nodeNumber)] 
  return previous + nodeTup[0][1] * alpha  


