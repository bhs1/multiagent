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
#
# Students: Jamie Lesser and Ben Solis-Cohen

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


        "*** YOUR CODE HERE ***"
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


    
    def getValue(self, gameState, curDepth, indent=""):
        numAgents = gameState.getNumAgents()
        curAgent = curDepth % numAgents
        if (curDepth == ((self.depth*numAgents)))  or len(gameState.getLegalActions(curAgent)) == 0:
            return (None, self.evaluationFunction(gameState))
        if (curAgent == 0):
            return max(map(lambda action: (action,self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, indent + "  ")[1]),gameState.getLegalActions(curAgent)),key = lambda x: x[1])
        else:
            return min(map(lambda action: (action, self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, indent + "  ")[1]),gameState.getLegalActions(curAgent)), key = lambda x: x[1])
        
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
        numAgents = gameState.getNumAgents()
        curAgent = curDepth % numAgents
        if (curDepth == ((self.depth*numAgents)))  or len(gameState.getLegalActions(curAgent)) == 0:
            return (None, self.evaluationFunction(gameState))
        if (curAgent == 0):
            return max(map(lambda action: (action,self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, indent + "  ")[1]),gameState.getLegalActions(curAgent)),key = lambda x: x[1])
        else:
            return (None, mean(map(lambda action: self.getValue(gameState.generateSuccessor(curAgent,action), curDepth + 1, indent + "  ")[1],gameState.getLegalActions(curAgent))))
        
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    # Useful information you can extract from a GameState (pacman.py)
    layout     = currentGameState.data.layout         # Layout of this game.
    walls      = layout.walls                         # Grid of where walls are.
    pos        = currentGameState.getPacmanPosition() # Pacman's current position.
    food       = currentGameState.getFood()           # Grid of where food is.
    foodNum    = len(food.asList())                   # Current number of food.
    capsules   = currentGameState.getCapsules()       # Available capsule positions.
    origCapNum = len(layout.capsules)                 # Original number of capsules.
    capNum     = len(capsules)                        # Current number of capsules.


    # Split ghosts between those who will be scared by the time 
    # pacman gets to them, and ones we consider dangerous.
    ghostStates  = set(map(lambda ghost: (ghost, manhattanDistance(pos, ghost.getPosition())), currentGameState.getGhostStates()))
    badGhosts    = set(filter(lambda (ghost, dist): ghost.scaredTimer <= dist, ghostStates))
    scaredGhosts = ghostStates - badGhosts
    origGhostPositions =  map(lambda x: x[1],layout.agentPositions)[1:]

    origGhostNum   = len(ghostStates)    # Number of ghost when game started.
    scaredGhostNum = len(scaredGhosts)   # Number of scared ghosts right now.


    # LOSE FEATURE
    # Losing is really bad. Really really bad.
    # Don't even bother looking at other features.
    if currentGameState.isLose():
        return -100000


    # WIN FEATURE
    # Winning is great! 
    # But some wins are better than others.
    winFeature = 1000000 * int(currentGameState.isWin())


    # GHOST HOUSE FEATURE 
    # Discourage being close to where the ghosts start.
    # Boolean value reflects pacman being within 1 unit 
    # from wherever a ghost starts.
    ghostHouseFeature = int(any(map(lambda ghostPos: 
                                    searchAgents.mazeDistance(pos,ghostPos,currentGameState) <= 1, 
                                    origGhostPositions)))


    # SCARED GHOST FEATURE
    # Encourage eating a scared ghost. 
    # This feature is intimiately related to the capsule feature 
    # as it should be incentivised to finish eating the available
    # scared ghosts before eating another capsule.
    scaredGhostFeature = 0
    minScaredGhostDist = float("inf")
    
    if scaredGhostNum > 0:
        minScaredGhostPos   = tuple(map(int, min(scaredGhosts, key = lambda x: x[1])[0].getPosition()))
        minScaredGhostDist  = searchAgents.mazeDistance(pos, minScaredGhostPos, currentGameState)
        scaredGhostFeature += origGhostNum - scaredGhostNum + 1/float(minScaredGhostDist)

    elif capNum > 0:
        scaredGhostFeature += (origCapNum - capNum) * (origGhostNum + 1)


    # CAPSULE FEATURE
    # Encourage eating a capsule. 
    # This feature is intimiately related to the scared ghost feature 
    # as it should be incentivised to eat the 
    # scared ghosts before eating another capsule.
    capsuleFeature = 0
    minCapsuleDist = float("inf")

    if scaredGhostNum > 0:
        capsuleFeature = 1 + (origCapNum - capNum - 1) * (origGhostNum + 1)

    elif capNum > 0:
        minCapsulePos  = min(map(lambda cap: (cap,util.manhattanDistance(pos,cap)), capsules), key = lambda x: x[1])[0]
        minCapsuleDist = searchAgents.mazeDistance(pos, minCapsulePos, currentGameState)
        capsuleFeature =  1/float(minCapsuleDist) + (origCapNum - capNum) * (origGhostNum + 1)


    # GOAL BOUND FEATURE
    # Encourage decreasing the estimated path to finishing the food.
    # Adopted from Ben's food heuristic last project.
    goalBoundDenom = 1 + (len(capsules)+1)*len(ghostStates)
    goalBound = float("inf")
    
    if len(scaredGhosts) == 0 and len(capsules) == 0:
        goalBound = distHeuristic(pos, food.asList(), currentGameState)
        goalBoundDenom += goalBound + 2*foodNum
    
    goalBoundFeature = 1 / float(goalBoundDenom)


    # BAD GHOST FEATURE
    # Bad ghosts are bad, run away!
    # Our main priority is getting away from reall close ghosts,
    # but increasing distances to all ghosts is also ideal.
    minBadGhostFeature = 0
    sumBadGhostFeature = 0

    if len(badGhosts) > 0:
        # If a ghost is really close, take the time to compute the actual 
        # maze distance so we know if pacman is safe or not. 
        for ghost, dist in badGhosts:
            if dist < 4:
                badGhosts.remove((ghost, dist))
                badGhosts.add((ghost, searchAgents.mazeDistance(pos, tuple(map(int,ghost.getPosition())), currentGameState)))

        minBadGhostDist = min(badGhosts, key = lambda x: x[1])[1]
        sumBadGhostDist = sum(map(lambda x: x[1], badGhosts))

        if minBadGhostDist < 5:
            minBadGhostFeature = 1 / float(1 + minBadGhostDist)
        if sumBadGhostDist < 8:
            sumBadGhostFeature = 1 / float(1 + sumBadGhostDist)


    # WALL FEATURE
    # Generally safer to be near less walls because then 
    # it's harder to get stuck between ghosts.
    # If there are three walls around pacman, that is really dangerous!
    wallNum = 0

    for i in [1, -1]:
        wallNum += int(walls[pos[0]][pos[1]+i])
        wallNum += int(walls[pos[0]+i][pos[1]])
    
    wallsFeature = 1 / float(1 + wallNum)
    
    if wallNum == 3 and len(scaredGhosts) == 0:
        wallsFeature = 30


    # Line FEATURE
    # Generally discourage being in a line with two other ghosts
    # because this can lead to being stuck between two ghosts.
    lineFeature = 0
    horiGhosts  = 0
    vertGhosts  = 0

    for (ghost, dist) in badGhosts:
        horiGhosts += int(ghost.getPosition()[0] == pos[0] and dist < 4)
        vertGhosts += int(ghost.getPosition()[1] == pos[1] and dist < 4)
        if horiGhosts > 1 or vertGhosts > 1:
            lineFeature = 1
            break
    

    # WEIGHTED SUM OF FEATURES
    # Some features are more important than others.
    return ( 1      * winFeature
             + 20   * goalBoundFeature
             + 300  * capsuleFeature 
             + 150  * scaredGhostFeature
             - 200  * minBadGhostFeature 
             - 40   * sumBadGhostFeature 
             - 40   * ghostHouseFeature 
             - 30   * lineFeature
             - 3    * wallsFeature
             + random.uniform(0, 0.01) )


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

