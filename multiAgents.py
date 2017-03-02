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
import random, util

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
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    badGhosts = [ghostState for ghostState in ghostStates if ghostState.scaredTimer == 0]
    nearbyScaredGhosts = [ghostState for ghostState in ghostStates 
                          if ((ghostState.scaredTimer > 0) 
                              and (manhattanDistance(pos, ghostState.getPosition()) < ghostState.scaredTimer))]
    capsules = currentGameState.getCapsules();

    DEBUG = False
    
    if DEBUG: print "___________________________________________"
    if DEBUG: print "Is win state:", currentGameState.isWin()
    if (currentGameState.isWin()):
        if DEBUG: print "Score: 10000"
        return 20000

    if DEBUG: print "Is lose state:", currentGameState.isLose()
    # Move into helper function for testing death
    if (currentGameState.isLose()):
        if DEBUG: print "Score: -10000"
        return -20000


    ######## SCORE FEATURE ##########
    scoreFeature = currentGameState.getScore()

    ######## BAD GHOST FEATURE #########
    badGhostFeature = 0
    if len(badGhosts) > 0:
        minManhattanBadGhost = min(map(lambda ghost: manhattanDistance(pos, ghost.getPosition()), badGhosts))
        if DEBUG: print "Minimum distance to bad ghost:", minManhattanBadGhost
        MAX = 4
        badGhostFeature = 1 / float(min(minManhattanBadGhost, MAX))
    
    ######## SCARED GHOST FEATURE #########
    ####### EAT GHOST FEATURE #############
    # add in try-catch
    scaredGhostFeature = 0
    eatGhostFeature = 0
    if len(nearbyScaredGhosts) > 0:
        minManhattanScaredGhost = min(map(lambda ghost: manhattanDistance(pos, ghost.getPosition()), nearbyScaredGhosts))
        if DEBUG: print "Minimum distance to scared ghost:", minManhattanScaredGhost
        scaredGhostFeature = 1 / float(minManhattanScaredGhost)
        if minManhattanScaredGhost < 3:
            eatGhostFeature = 1 / float(minManhattanScaredGhost)


    ######## MAX FOOD FEATURE #############
    maxDistToFood = max(map(lambda f: manhattanDistance(pos, f), food.asList()))
    if DEBUG: print "Maximum distance to food:", maxDistToFood
    maxFoodFeature =  1 / float(maxDistToFood)

    ######## MIN FOOD FEATURE #############
    minDistToFood = min(map(lambda f: manhattanDistance(pos, f), food.asList()))
    if DEBUG: print "Minimum distance to food:", minDistToFood
    minFoodFeature =  1 / float(minDistToFood)

    ######## CAPSULE FEATURE #############
    # should make capsule worth even more if bad ghost is close (but not too close)
    capsuleFeature = 0
    if len(capsules) > 0:
        minDistToCapsule = min(map(lambda cap: manhattanDistance(pos, cap), capsules))
        if DEBUG: print "Minimum distance to capsule:", minDistToCapsule
        MAX = 6
        capsuleFeature =  1 / float(max(minDistToCapsule, MAX))
    
    ######## NUM FOOD FEATUER ############
    numFood = currentGameState.getNumFood()
    numFoodFeature = 1 / float(numFood)
    if DEBUG: print "Number of food left:", numFood

    # THOUGHTS:
    #
    # want to minimize maximum distance to food
    #
    # if maximum distance to food is same, want to minimize distance to food
    #
    # if maximum distance to food is same, and minimum distance to food is same, 
    # want to minimize number of food
    #
    # it should be more important to minimize number of food than keep max food distance the same
    #
    # score increases by almost 10 for eating a food so everything should be contributing at least twice that
    # 
    # need to capture more in the min dist to scaredghost - not enough
    # because eating the ghost actually increases the min dist
    # so if min dist is 1, actually want to return something even higher

    weightedSum = (  1     * scoreFeature 

                     + 20  * maxFoodFeature
                     + 30  * minFoodFeature
                     + 80  * numFoodFeature

                     + 100  * capsuleFeature
                     + 15  * scaredGhostFeature
                     + 300  * eatGhostFeature

                     - 1000  * badGhostFeature 
                     )

    # if you want to minimize something, return positive inverse (which feature already is)
    # if you care more about minimizing one thing than the other multiply it by more

    if DEBUG:
        print "Weighted Score Feature:       ", 1   * scoreFeature
        print ""
        print "Weighted Max Food Feature:    ", 20 * maxFoodFeature
        print "Weighted Min Food Feature:    ", 30 * minFoodFeature
        print "Weighted Num Food Feature:    ", 90 * numFoodFeature
        print ""
        print "Weighted Capsule Feature:     ", 40 * capsuleFeature
        print "Weighted Scared Ghost Feature:", 50 * scaredGhostFeature
        print "Weighted Eat Ghost Feature:   ", 200 * eatGhostFeature        
        print ""
        print "Weighted Bad Ghost Feature:   ", -500 * badGhostFeature
        print ""
        print "Total sum:                    ", weightedSum 
        print ""

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



