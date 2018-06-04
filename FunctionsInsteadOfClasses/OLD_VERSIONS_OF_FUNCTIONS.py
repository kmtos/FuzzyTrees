###########################################################
# Given the df, and the nodeDecisions, node DF IDs and 
# the nodeValues, alter the weights based upon correctness
###########################################################
def AlterWeights(df, df_weights, error, idColumn, rateOfChange, className, nodeDecisionsFileName, nodeDFIDsFileName, nodeValuesFileName):
  with open(nodeDecisionsFileName + ".csv") as nodeDecisionsFile:
    nodeDecisionsFileReader = csv.reader(nodeDecisionsFile)
    next(nodeDecisionsFileReader)
    nodeDecisions = [tuple(line) for line in nodeDecisionsFileReader]
  with open(nodeValuesFileName + ".csv") as nodeValuesFile:
    nodeValuesFileReader = csv.reader(nodeValuesFile)
    next(nodeValuesFileReader)
    nodeValues = [tuple(line) for line in nodeValuesFileReader]
  with open(nodeDFIDsFileName + ".csv") as nodeDFIDsFile:
    nodeDFIDsFileReader = csv.reader(nodeDFIDsFile)
    next(nodeDFIDsFileReader)
    nodeDFIDs = [tuple(line) for line in nodeDFIDsFileReader]
  alpha = .5 * math.log1p((1 - error) / error) * rateOfChange # exponent factor for adjustment of weights
  print ("error=", error, "\talpha*rateOfChange=", alpha, "\tCorrectFactor=", math.exp(-1*alpha), "\tIncorrectFactor=", math.exp(1*alpha)  )
  for decisionTup in nodeDecisions:
    dfIDs = next(iteTup[1] for iteTup in nodeDFIDs if int(iteTup[0]) == int(decisionTup[0]) )
    dfIDs = dfIDs.replace("[", "").replace("]", "").replace(",", "").split()
    dfIDs = [int(i) for i in dfIDs]
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
  nodeDFIDsFile.close()
  del nodeDFIDsFileReader
  nodeDecisionsFile.close()
  del nodeDecisionsFileReader
  nodeValuesFile.close()
  del nodeValuesFileReader
  return df_weights


##################################################################
# Makes a Decision Tree that saves output in three forms:
#    1) The tuple found with FindingBestSplit for each node
#    2) The node number and the DF ID's at that leaf
#    3) The Decision that was made at each leaf
##################################################################
def MakeTree(df, className, nGiniSplits, giniEndVal, maxDepth, idColumn, minSamplesSplit, df_weights, nodeDFIDsFileName, nodeValuesFileName, nodeDecisionsFileName):
  print ("\n\n###################################\n Making a Decision Tree\n###################################")
  maxNodes = 0
  for i in range(1,maxDepth+1): maxNodes += 2**i
  nodeValues = [] #Node number,  amount of gini increase, the column name, the value split at, range between splits)
  nodeDFIds = [] #The point ID's that are in each leaf
  nodeCount = 0
  try:
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


