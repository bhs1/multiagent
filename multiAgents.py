# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, itertools
import search, searchAgents

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPositions = successorGameState.getGhostPositions()

        DEBUG = False
        
        if DEBUG: print "Is win state:", successorGameState.isWin()

        if (successorGameState.isWin()):
            if DEBUG: print "Score: 2000"
            return 2000

        if DEBUG: print "Is lose state:", successorGameState.isLose()

        # Move into helper function for testing death
        if (successorGameState.isLose()):
            if DEBUG: print "Score: -2000"
            return -2000

        score = successorGameState.getScore()

        # take into account scared ghosts separately
        minManhattanGhost = min(map(lambda ghost: manhattanDistance(newPos, ghost.getPosition()), newGhostStates))
        MAX = 4
        score -= 10*(MAX - min(minManhattanGhost, MAX))

        if DEBUG: print "Minimum distance to ghost:", minManhattanGhost

        minManhattanFood = min(map(lambda food: manhattanDistance(newPos, food), newFood.asList()))
        score += 10 / minManhattanFood

        if DEBUG: print "Minimum distance to food:", minManhattanFood

        if DEBUG: print "Score:", score

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.getValue(gameState,0)[0]

    
    def getValue(self, gameState, curDepth):
        '''Recursive function to calculate the minimax from depth curDepth in game state gameState
        '''
        numAgents = gameState.getNumAgents()
        curAgent = curDepth % numAgents
        if (curDepth == ((self.depth*numAgents)))  or len(gameState.getLegalActions(curAgent)) == 0:
            # If we've reached a terminal state or the depth specified in
            # self.depth, just evaluate the current terminal/leaf state
            return (None, self.evaluationFunction(gameState))
        if (curAgent == 0):
            # curAgent == 0 implies that pacman is going
            # Return the maximum (action, value) tuple (comparing by minimax
            # value) over all possible actions
            return max(map(lambda action: (action,self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1)[1]),gameState.getLegalActions(curAgent)),key = lambda x: x[1])
        else:
            # Now do the same for ghosts but minimizing this time. Also,
            # we no longer care about the actions so we avoid generating
            # the tuples and just return None as the action
            return (None,min(map(lambda action: self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1)[1],gameState.getLegalActions(curAgent))))
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        DEBUG = False
        actions = gameState.getLegalActions(0)
        maxAction = actions[0]
        alpha = float("-inf")
        beta = float("inf")
        maxSoFar = float("-inf")#self.getValue(gameState.generateSuccessor(0,maxAction),1)

        if DEBUG: print "-------------------------"
        if DEBUG: print "alpha", alpha
        if DEBUG: print "beta", beta
        if DEBUG: print "depth", 0 
        if DEBUG: print "curAgent", 0
        if DEBUG: print "numAgents",gameState.getNumAgents()
        if DEBUG: print actions
        if DEBUG: print "max:"
        for action in actions:
            val = self.getValue(gameState.generateSuccessor(0,action), 1, alpha, beta)
            if val > maxSoFar:
                maxAction = action
                maxSoFar = val
            if val > beta: 
                if DEBUG: print "Final:", maxSoFar
                return maxAction
            else: alpha = max(alpha, val)

        return maxAction
        
    def getValue(self, gameState, curDepth, alpha, beta, indent=""):
        DEBUG = False
        numAgents = gameState.getNumAgents()
        curAgent = curDepth % numAgents
        if DEBUG: print indent, "-------------------------"
        if DEBUG: print indent, "alpha:    ", alpha
        if DEBUG: print indent, "beta:     ", beta
        if DEBUG: print indent, "depth:    ", self.depth 
        if DEBUG: print indent, "curAgent: ", curAgent
        if DEBUG: print indent, "numAgents:", numAgents

        #        print indent, gameState,curAgent

        actions = gameState.getLegalActions(curAgent)
        if DEBUG: print indent, "actions:  ", actions

        if (curDepth == self.depth*numAgents)  or len(actions) == 0:
            val = self.evaluationFunction(gameState)
            if DEBUG: print indent, "value:    ", self.evaluationFunction(gameState)
            return float(val)

        if (curAgent == 0):
            if DEBUG: print indent,"max:"
            val = float("-inf")
            for action in actions:
                newVal = self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, alpha, beta, indent + "  ")
                if DEBUG: print indent, "newVal:   ", newVal
                val = max(val, newVal)
                if val > beta: 
                    if DEBUG: print indent, "max:      ", val
                    return val
                else: 
                    alpha = max(alpha, val)
                    if DEBUG: print indent, "alpha:    ", alpha
                    if DEBUG: print indent, "beta:     ", beta
            return val #(?)
        else:
            if DEBUG: print indent,"min:"
            val = float("inf")
            for action in actions:
                newVal = self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, alpha, beta, indent + "  ")
                if DEBUG: print indent, "newVal:   ", newVal
                val = min(val, newVal)
                if val < alpha: 
                    if DEBUG: print indent, "min:      ", val
                    return val
                else: 
                    beta = min(beta, val)
                    if DEBUG: print indent, "alpha:    ", alpha
                    if DEBUG: print indent, "beta:     ", beta
            return val #(?)

from numpy import mean
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.getValue(gameState, 0)[0]

    def getValue(self, gameState, curDepth, indent=""):
        '''See description of ExpectimaxAgent.getAction above'''        
        numAgents = gameState.getNumAgents()
        curAgent = curDepth % numAgents

        if (curDepth == ((self.depth*numAgents)))  or len(gameState.getLegalActions(curAgent)) == 0:
            # If we've reached a terminal state or the depth specified in
            # self.depth, just evaluate the current terminal/leaf state
            return (None, self.evaluationFunction(gameState))
        if (curAgent == 0):
            # Pacman does not move randomly so we just do the same as we
            # did in MinimaxAgent.getValue.
            return max(map(lambda action: (action,self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, indent + "  ")[1]),gameState.getLegalActions(curAgent)),key = lambda x: x[1])
        else:
            # Ghosts move randomly (uniformly) so we just take the mean
            # of the predicted values over all possible actions
            return (None, mean(map(lambda action: self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, indent + "  ")[1],gameState.getLegalActions(curAgent))))
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    # Useful information you can extract from a GameState (pacman.py)
    layout     = currentGameState.data.layout         # add comment here
    walls      = layout.walls                         # add comment here
    pos        = currentGameState.getPacmanPosition() # add comment here
    capsules   = currentGameState.getCapsules()       # add comment here
    food       = currentGameState.getFood()           # add comment here
    origCapNum = len(layout.capsules)                 # add comment here
    capNum     = len(capsules)                        # add comment here

    # add comment here
    ghostStates  = set(map(lambda ghost: (ghost, manhattanDistance(pos, ghost.getPosition())), currentGameState.getGhostStates()))
    badGhosts    = set(filter(lambda (ghost, dist): ghost.scaredTimer <= dist, ghostStates))
    scaredGhosts = ghostStates - badGhosts

    origGhostNum   = len(ghostStates)    # add comment here
    scaredGhostNum = len(scaredGhosts)    # add comment here


    # explain
    winFeature = 0
    if currentGameState.isWin():
        winFeature = 1000000

    # explain
    if (currentGameState.isLose()):
        return -100000


    ############### GHOST HOUSE FEATURE #############
    origGhostPositions =  map(lambda x: x[1],layout.agentPositions)[1:]
    ghostHouseFeature = int(any(map(lambda ghostPos: searchAgents.mazeDistance(pos,ghostPos,currentGameState) <= 1, origGhostPositions)))
    #################################################



    ######## SCARED GHOST FEATURE #########
    scaredGhostFeature = 0
    minScarMazeDist = float("inf")
    if scaredGhostNum > 0:
        distTuple = min(scaredGhosts, key = lambda x: x[1])
        span = searchAgents.mazeDistance(pos, tuple(map(int,distTuple[0].getPosition())), currentGameState)
        minScarMazeDist = span
        scaredGhostFeature += 1 + (origCapNum - capNum - 1)*(origGhostNum+1) + origGhostNum - scaredGhostNum + 1/float(span)
    else:
        scaredGhostFeature += (origCapNum - capNum)*(origGhostNum+1)


    ######## CAPSULE FEATURE #############
    # should make capsule worth even more if bad ghost is close (but not too close)
    capsuleFeature = 0
    minCapMazeDist = float("inf")
    if scaredGhostNum > 0:
        capsuleFeature += 1 + (origCapNum - capNum - 1)*(origGhostNum+1)
    elif capNum > 0:
        distTuple = min(map(lambda cap: (cap,util.manhattanDistance(pos,cap)), capsules), key = lambda x: x[1])
        span = searchAgents.mazeDistance(pos, distTuple[0], currentGameState)
        minCapMazeDist = span
        capsuleFeature += (origCapNum - capNum)*(origGhostNum+1) + 1/float(span)
    

    ####### LOWER BOUND DIST TO GOAL ########
    distBoundDenom = 1 + (len(capsules)+1)*len(ghostStates)
    numFood = currentGameState.getNumFood()
    minBound = float("inf")
    if len(scaredGhosts) == 0 and len(capsules) == 0:
        distLowerBound = distHeuristic(pos, food.asList(), currentGameState)
        distBoundDenom += distLowerBound + 2*numFood
        minBound = distLowerBound
    distBoundFeature = 1 / float(distBoundDenom)


    ######## BAD GHOST FEATURE #########
    badGhostFeature = 0
    badGhostMeanFeature = 0

    if len(badGhosts) > 0:
        for ghost, dist in badGhosts:
            if dist < 4:
                badGhosts.remove((ghost, dist))
                badGhosts.add((ghost, searchAgents.mazeDistance(pos, tuple(map(int,ghost.getPosition())), currentGameState)))

        distTuple = min(badGhosts, key = lambda x: x[1])
        minManhattanBadGhost = distTuple[1]
        meanManhattanBadGhost = float("inf")

        fartherGhosts = map(lambda x: x[1], badGhosts)
        fartherGhosts.remove(minManhattanBadGhost)

        if len(fartherGhosts) > 0:
            meanManhattanBadGhost = sum(fartherGhosts)

        if (0.5*minManhattanBadGhost <= minCapMazeDist 
            and 0.5*minManhattanBadGhost <= minScarMazeDist
            and 0.5*minManhattanBadGhost <= minBound):
            if minManhattanBadGhost < 4:
                badGhostFeature = 1 / float(1 + minManhattanBadGhost)
            if meanManhattanBadGhost < 8:
                badGhostFeature = 1 / float(1 + meanManhattanBadGhost)

    ####### WALL FEATURE #################
    numWalls = 0
    for i in [1, -1]:
        numWalls += int(walls[pos[0]][pos[1]+i])
        numWalls += int(walls[pos[0]+i][pos[1]])
    wallsFeature = 1/(1+numWalls)
    if numWalls == 3 and len(scaredGhosts) == 0:
        wallsFeature = 60

    ######## GHOST SANDWICH FEATURE #########
    # while this works right now really need to generalize and 
    # maybe make a tad better
    horiGhosts = 0
    vertGhosts = 0

    for (ghost, dist) in badGhosts:
        horiGhosts += int(ghost.getPosition()[0] == pos[0] and dist < 4)
        vertGhosts += int(ghost.getPosition()[1] == pos[1] and dist < 4)
    lineFeature = float(horiGhosts > 1 or vertGhosts > 1) 


    weightedSum = (1      * winFeature
                   + 20   * distBoundFeature
                   + 250  * capsuleFeature 
                   + 100  * scaredGhostFeature
                   - 200  * badGhostFeature 
                   - 150  * badGhostMeanFeature 
                   - 10   * wallsFeature
                   - 100  * ghostHouseFeature 
                   - 100  * lineFeature
                   + random.uniform(0, 0.01))
        
    return weightedSum

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
    """
            
    def getAction(self, gameState):
        """
        Returns an action.  You can use any method you want and search to any depth you want.
        Just remember that the mini-contest is timed, so you have to trade off speed and computation.
              
        Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
      """

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def distHeuristic(pacPos, posList, gameState):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """

    # Fetch state.
    #position = state.

    #position, foodGrid = state
    position = pacPos
    foods = posList

    # If there's no food left, return 0.
    if len(foods) == 0:
        return 0
    
    # Create a graph from the food state.
    G = []
    V = len(foods)
    for ((i, food1), (j, food2)) in itertools.combinations(enumerate(foods), 2):
        G.append((i, j, util.manhattanDistance(foods[i], foods[j])))
        
    # Initialize the distance to be the distance to the closest food.
    distTuple = min(map(lambda food: (food,util.manhattanDistance(position,food)), foods), key = lambda x: x[1])
    dist = searchAgents.mazeDistance(position, distTuple[0], gameState)
    # Kruskal's algorithm for finding minimum-cost spanning tree.
    # See pseudocode at https://en.wikipedia.org/wiki/Kruskal's_algorithm
    parent = dict()
    rank = dict()

    # Make unique set ("subtree") for each node to start.
    for node in range(V):
        parent[node] = node
        rank[node] = 0

    # Sort the edges in non-decreasing order by weight.
    G.sort(key = lambda edge: edge[2])
    
    # Keep adding shortest edges and unioning disjoint "trees" until
    # spanning tree is found.
    for (src, dest, weight) in G:
        if find(src, parent) != find(dest, parent):
            union(src, dest, rank, parent)
            dist += weight # Remember, we just want to sum the edge weights.

    # Return the sum of the distances in the MST (plus the distance to the closest node).
    return dist

def find(node, parent):
    """
    The find function for Union-Find in Kruskal's algorithm (path compression).
    See pseudocode at https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    """
    if parent[node] != node:
        parent[node] = find(parent[node], parent)
    return parent[node]

def union(node1, node2, rank, parent):
    """
    The union function for Union-Find in Kruskal's algorithm.
    See pseudocode at https://en.wikipedia.org/wiki/Disjoint-set_data_structure
    """
    root1 = find(node1, parent)
    root2 = find(node2, parent)
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        else:
            parent[root1] = root2
            if rank[root1] == rank[root2]:
                rank[root2] += 1

