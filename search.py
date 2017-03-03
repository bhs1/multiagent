# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util,heapq

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def genericSearch(problem, frontier):

    # Initial node has no prior action or parent node.
    frontier.push((problem.getStartState(), None, None))

    # Initialize an explored set to avoid repeating paths.
    explored = set()

    # Loop until we return a solution or the frontier is emptied (failure).
    while not frontier.isEmpty():
        #print(frontier.list)
        
        # Remove a leaf from the frontier.
        leaf = frontier.pop()

        # If it's at a goal state, return the solution (we're done)!
        if problem.isGoalState(leaf[0]):
            return getSolution(leaf, problem)

        #If node not explored yet
        if leaf[0] not in explored:
            
            # Add the state to the explored set.
            explored.add(leaf[0])

            # Expand the node's successors.
            for (nextState, action, cost) in problem.getSuccessors(leaf[0]):        
                frontier.push((nextState, action, leaf))

    # A failure will return an empty list of actions.
    return []

def getSolution(node, problem):
    """
    Helper function to follow backpointers and produce a valid solution.
    """
    sol = list()
    while(node[2] != None):
        sol.append(node[1])
        node = node[2]
    sol.reverse()
    return sol

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return genericSearch(problem, util.Stack())

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return genericSearch(problem, util.Queue())

def uniformCostSearch(problem):
    """Search the node that has the lowest combined cost and heuristic first."""
    """Search the node of least total cost first."""    
    return aStarSearch(problem)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def printFrontier(frontier):
    for node in frontier.heap:
        print(node[2][0][0],str(node[2][0][1]),node[0])
        
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    """Search the node of least total cost first."""    
    frontier = util.PriorityQueue()
    # Initial node has no prior action or parent node.
    frontier.push((problem.getStartState(), None, None, 0),heuristic(problem.getStartState(),problem))

    # Initialize an explored set to avoid repeating paths.
    explored = set()

    # Loop until we return a solution or the frontier is emptied (failure).
    while not frontier.isEmpty():

        # Remove a leaf from the frontier.
        leaf = frontier.pop()        

        # If it's at a goal state, return the solution (we're done)!
        if problem.isGoalState(leaf[0]):
            return getSolution(leaf, problem)

        #If we haven't explored the node yet
        if leaf[0] not in explored:
            
            # Add the state to the explored set.
            explored.add(leaf[0])
        
            # Expand the node's successors.
            for (nextState, action, cost) in problem.getSuccessors(leaf[0]):
                newChild = (nextStat1e,action,leaf,leaf[3]+cost)
                newChildF = heuristic(newChild[0],problem) + newChild[3]
                frontier.push(newChild,newChildF)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
