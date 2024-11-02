from config import * 

"""
    Represents a single node in the A* search algorithm.

    @param board: The game board represented as a 2D list.
    @param coords: List of coordinates representing the snake's body.
    @param nodeID: Unique identifier for the node.
    @param parent: The ID of the parent node for path tracing.
    @param f: Total cost (g + h) for the node.
    @param g: Cost from the start node to the current node.
    @param head: Coordinates of the snake's head.
    @param direction: The direction the snake is moving in.
"""
class Node(object):
    def __init__(self, board, coords, nodeID, parent, f, g, head, direction):
        self.board = board
        self.coords = coords
        self.f = f
        self.g = g
        self.head = head
        self.info = {'id': nodeID, 'parent': parent, 'direction': direction}

    """
        Less-than comparison for priority queue.

        @param other: The other Node object to compare with.
        @return: True if this node has a lower f value, or equal f but higher g value.
    """
    def __lt__(self, other):
        if self.f == other.f:
            return self.g > other.g
        else:
            return self.f < other.f

    """
        Equality comparison.

        @param other: The other Node object to compare with.
        @return: True if this node is equal to the other node based on the f value.
    """
    def __eq__(self, other):
        if not other:
            return False
        if not isinstance(other, Node):
            return False
        return self.f == other.f

"""
    Handles the AI logic for the snake using the A* algorithm.

    @param direction: The initial direction of the snake.
"""
class SnakeAI(object):
    def __init__(self, direction):
        self.coords = []
        self.path = None
        self.nodeID = 0
        self.direction = self.getDirection(direction)
        self.board = self.getBoard()
        
    """
        Initializes the game board with walls on the edges.

        @return: A 2D list representing the board.
    """
    def getBoard(self):
        board = [[0 for x in range(27)] for y in range(27)]
        for i in range(27):
            board[0][i] = board[26][i] = board[i][0] = board[i][26] = 2
        return board

    """
        Converts screen coordinates to board grid coordinates.

        @param coord: The screen coordinates (tuple).
        @return: The corresponding grid coordinates (tuple).
    """
    def getXY(self, coord):
        x = (coord[0] - 10) // 20 + 1
        y = (coord[1] - 90) // 20 + 1
        return x, y
    
    """
        Converts a direction to its corresponding index.

        @param direction: The direction (UP, DOWN, LEFT, RIGHT).
        @return: The index of the direction.
    """
    def getDirection(self, direction):
        move = [UP, DOWN, LEFT, RIGHT]
        return move.index(direction)
    
    """
        Clears the game board, removing any snake trails except for walls.
    """
    def clearBoard(self):
        for x in range(1, len(self.board) - 1, 1):
            for y in range(1, len(self.board[0]) - 1, 1):
                self.board[y][x] = 0

    """
        Marks the snake's body and the goal position on the board.

        @param coords: List of snake's body coordinates.
        @param coord: The goal position (tuple).
    """
    def denoteXY(self, coords, coord):
        self.coords.clear()
        x, y = self.getXY(coord)
        self.goal = x, y
        self.board[y][x] = 1
        self.head = self.getXY(coords[0])

        for coord in coords:
            x, y = self.getXY(coord)
            self.coords.append((x, y))
            self.board[y][x] = 2
    
    """
        Finds the next direction for the snake to move.

        @param coords: List of snake's body coordinates.
        @param coord: The food's position (tuple).
        @return: The index of the next direction, or -1 if no path is found.
    """
    def getNextDirection(self, coords, coord):
        if not self.path:
            self.findPath(coords, coord)
        if self.path:
            return self.path.pop()
        else:
            return -1
        
    """
        Creates a copy of the snake's body coordinates.

        @param coords: The original coordinates list.
        @return: A copied list of coordinates.
    """
    def copyCoords(self, coords):
        coordies = []

        for coord in coords:
            coordies.append(coord)

        return coordies

    """
        Copies the board state and marks the snake's body and the goal.

        @param coords: The snake's body coordinates.
        @return: A 2D list representing the board with the snake and goal.
    """
    def copyBoard(self, coords):
        board = self.getBoard()

        for coord in coords:
            x, y = coord
            board[y][x] = 2
        x, y = self.goal
        board[y][x] = 1

        return board
    
    """
        Calculates the distance from the current position to the goal.

        @param x: The x-coordinate.
        @param y: The y-coordinate.
        @return: The heuristic distance to the goal.
    """
    def getHeuristic(self, x, y):
        x1, y1 = self.goal
        return (abs(x - x1) + abs(y - y1))
    
    """
        Sets up the board and starts the A* pathfinding.

        @param coords: List of snake's body coordinates.
        @param coord: The goal position (tuple).
    """
    def findPath(self, coords, coord):
        self.clearBoard()
        self.denoteXY(coords, coord)
        self.path = self.aStar()

    """
        Implements the A* search algorithm to find the shortest path.

        @return: A list representing the path to the goal, or None if no path is found.
    """
    def aStar(self):
        h = self.getHeuristic(self.head[0], self.head[1])
        g = 0
        node = Node(self.board, self.coords, 0, 0, h, g, self.coords[0], self.direction)
        open = []
        close = []
        self.expandNode(open, node)

        while open:
            node = heapq.heappop(open)

            if g < node.g:
                g = node.g
            if g - node.g > 1:
                continue

            if node.head == self.goal:
                return self.makePath(close, node.info)

            close.append(node.info)
            self.expandNode(open, node)

        if close:
            path = [close[0]['direction']]
            return path

        return

    """
        Checks for holes or potential traps around the snake.

        @param x: The x-coordinate.
        @param y: The y-coordinate.
        @param direction: The current direction index.
        @param board: The game board.
        @return: A heuristic penalty value indicating danger level.
    """
    def isHole(self, x, y, direction, board):
        if y - 1 >= 0 and direction > 1:
            if x == 25 and board[y - 1][x] == 2:
                return 10

        if y + 1 < 27 and direction > 1:
            if board[y + 1][x] == 2 and x == 1:
                return 10

        if y + 1 < 27 and y - 1 >= 0 and direction > 1:
            if board[y + 1][x] == 2 and board[y - 1][x] == 2:
                return 10
            if (board[y + 1][x] == 2 or board[y - 1][x] == 2):
                return 0

        if x - 1 >= 0 and direction < 2:
            if y == 25 and board[y][x - 1] == 2:
                return 10

        if x + 1 < 27 and direction < 2:
            if board[y][x + 1] == 2 and y == 1:
                return 10

        if x + 1 < 27 and x - 1 >= 0 and direction < 2:
            if board[y][x + 1] == 2 and board[y][x - 1] == 2:
                return 10
            if (board[y][x + 1] == 2 or board[y][x - 1] == 2):
                return 0

        return 3

    """
        Expands a node by generating all possible moves.

        @param open: The open list (priority queue).
        @param nodes: The current node to be expanded.
    """
    def expandNode(self, open, nodes):
        moves = [UP, DOWN, LEFT, RIGHT]
        x, y = nodes.head

        for i in range(4):
            dx, dy = moves[i]
            x1 = x + dx
            y1 = y + dy
            if nodes.board[y1][x1] < 2:
                coords = self.copyCoords(nodes.coords)
                head = (x1, y1)
                coords.insert(0, head)
                coords.pop()
                board = self.copyBoard(coords)
                h = self.getHeuristic(x1, y1)
                if h > 0:
                    h1 = self.isHole(x1, y1, i, board)
                    h += h1
                g = nodes.g + 1
                self.nodeID += 1
                id = self.nodeID
                parent = nodes.info['id']
                node = Node(board, coords, id, parent, g + h, g, head, i)
                heapq.heappush(open, node)

    """
        Constructs the path from the start to the goal by backtracking.

        @param closed: The closed list containing visited nodes.
        @param information: The current node's information.
        @return: A list of directions representing the path.
    """
    def makePath(self, closed, information):
        path = [information['direction']]

        while closed:
            info = closed.pop()
            if info['id'] == information['parent']:
                path.append(info['direction'])
                information = info

        return path