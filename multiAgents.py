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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"

        # Use the successor state because this heuristic evaluates the result of taking this action.
        pos = successorGameState.getPacmanPosition()
        x, y = pos
        foodList = successorGameState.getFood().asList()
        score = successorGameState.getScore()

        # Avoid stopping because it usually wastes tempo and increases ghost risk.
        if action == 'Stop':
            return -10**9

        # Reward eating food and strongly prefer states with fewer remaining food pellets.
        if foodList:
            closestFoodDist = min(manhattanDistance(pos, food) for food in foodList)
            score += 12.0 / (closestFoodDist + 1.0)
            score -= 4.0 * len(foodList)

        # Split food into horizontal and vertical groups around Pacman's current position.
        leftFoods = [food for food in foodList if food[0] < x]
        rightFoods = [food for food in foodList if food[0] > x]
        downFoods = [food for food in foodList if food[1] < y]
        upFoods = [food for food in foodList if food[1] > y]

        # Choose the smaller non-empty side first, similar to clearing small components before large ones.
        def chooseSmallSide(groupA, groupB):
            if groupA and groupB:
                return groupA if len(groupA) <= len(groupB) else groupB
            return groupA if groupA else groupB

        # Reward moving toward the smaller horizontal food component.
        horizontalTarget = chooseSmallSide(leftFoods, rightFoods)
        if horizontalTarget:
            nearestHorizontalDist = min(manhattanDistance(pos, food) for food in horizontalTarget)
            score += 8.0 / (nearestHorizontalDist + 1.0)
            score += 2.0 / (len(horizontalTarget) + 1.0)

        # Reward moving toward the smaller vertical food component.
        verticalTarget = chooseSmallSide(downFoods, upFoods)
        if verticalTarget:
            nearestVerticalDist = min(manhattanDistance(pos, food) for food in verticalTarget)
            score += 8.0 / (nearestVerticalDist + 1.0)
            score += 2.0 / (len(verticalTarget) + 1.0)

        # Handle ghosts with a more aggressive risk-reward heuristic.
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(pos, ghostPos)
            scaredTime = ghostState.scaredTimer

            # If Pacman reaches an active ghost, this state is fatal.
            if ghostDist == 0 and scaredTime == 0:
                return -10**9

            # Strongly reward chasing ghosts that are safely edible.
            if scaredTime > ghostDist + 1:
                score += 260.0 / (ghostDist + 1.0)
                continue

            # Slightly reward approaching scared ghosts even if eating them is not guaranteed.
            if scaredTime > 0:
                score += 60.0 / (ghostDist + 1.0)
                continue

            # Penalize only nearby active ghosts, allowing Pacman to play more aggressively.
            if ghostDist <= 2:
                score -= 220.0 / ((ghostDist + 0.5) ** 2)
            elif ghostDist <= 4:
                score -= 45.0 / (ghostDist + 1.0)

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        
        best_action = None
        best_score = float("-inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.rec_minimax(successor, 0, 1)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def rec_minimax(self, gameState, curr_depth, agent_index):
        if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent_index)

        if not actions:
            return self.evaluationFunction(gameState)

        num_agents = gameState.getNumAgents()
        next_agent = (agent_index + 1) % num_agents
        next_depth = curr_depth + 1 if next_agent == 0 else curr_depth

        scores = []

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            score = self.rec_minimax(successor, next_depth, next_agent)
            scores.append(score)

        if agent_index == 0:
            return max(scores)

        return min(scores)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        best_action = None
        best_score = float("-inf")

        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = self.rec_alphabeta(successor, 0, 1, alpha, beta)

            # Pacman set max
            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        return best_action

    def rec_alphabeta(self, gameState, curr_depth, agent_index, alpha, beta):
        # States transitions
        if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agent_index)

        if not actions:
            return self.evaluationFunction(gameState)

        num_agents = gameState.getNumAgents()
        next_agent = (agent_index + 1) % num_agents
        next_depth = curr_depth + 1 if next_agent == 0 else curr_depth

        # Pacman
        if agent_index == 0:
            value = float("-inf")

            for action in actions:
                successor = gameState.generateSuccessor(agent_index, action)
                score = self.rec_alphabeta(successor, next_depth, next_agent, alpha, beta)
                value = max(value, score)

                # Pruning
                if value > beta:
                    return value

                alpha = max(alpha, value)

            return value

        # Ghosts
        value = float("inf")

        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            score = self.rec_alphabeta(successor, next_depth, next_agent, alpha, beta)
            value = min(value, score)

            # Pruning
            if value < alpha:
                return value

            beta = min(beta, value)

        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
