from config import *
from snakeAI import *
from graph_window import GraphWindow

"""
    Represents the snake in the game.

    Attributes:
        color (tuple): The color of the snake.
        length (int): The length of the snake.
        direction (tuple): The current direction of the snake's movement.
        coords (list): The coordinates of the snake's body.
"""
class Snake(object):
    def __init__(self):
        self.color = SNAKECOLOR
        self.create()

    """
        Sets up the initial state of the snake.
    """
    def create(self):
        self.length = 2 
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.coords = [(SCREEN_SIZE // 2, SCREEN_SIZE // 2 + TOP_MARGIN - 10)]
    
    """
        Updates the direction of the snake, avoiding reverse movement.

        @param direction: tuple, The new direction for the snake to move in.
        @return: None
    """
    def control(self, direction):
        if (direction[0] * -1, direction[1] * -1) == self.direction:
            return
        else:
            self.direction = direction

    """
        Moves the snake in the current direction and checks for self-collision.

        @return: bool, False if the snake collides with itself, True otherwise.
    """
    def move(self):
        cur = self.coords[0]
        x, y = self.direction

        new = (((cur[0] - MARGIN) + (x * GRID_SIZE)) % SCREEN_SIZE,
               ((cur[1] - TOP_MARGIN) + (y * GRID_SIZE)) % SCREEN_SIZE)
        new = (new[0] + MARGIN, new[1] + TOP_MARGIN)

        self.coords.insert(0, new)

        if len(self.coords) > self.length:
            self.coords.pop()

        if new in self.coords[1:]:
            return False

        return True

    def draw(self):
        head = self.coords[0]
        for c in self.coords:
            drawRect(c[0] + 1, c[1] + 1, GRID_SIZE - 1, GRID_SIZE - 1, self.color)

        drawRect(c[0] + 1, c[1] + 1, GRID_SIZE - 1, GRID_SIZE - 1, SNAKECOLOR)
        drawRect(head[0] - 1, head[1], GRID_SIZE -  1, GRID_SIZE - 1, HEADCOLOR)

    def eat(self):
        self.length += 1

"""
    Represents the food object in the game.

    Attributes:
        color (tuple): The color of the food.
        coord (tuple): The coordinates of the food.
"""
class Feed(object):
    def __init__(self):
        self.color = FOODCOLOR
        self.create()

    def create(self):
        self.coord = (random.randint(0, GRID_WIDTH - 1) * GRID_SIZE + MARGIN,
                      random.randint(0, GRID_HEIGHT - 1) * GRID_SIZE + TOP_MARGIN)

    def draw(self):
        drawRect(self.coord[0], self.coord[1], GRID_SIZE, GRID_SIZE, self.color)

"""
    The main game loop that runs the snake game and sends data to the graph process.

    @param data_queue: multiprocessing.Queue, The queue to send game data to the graph process.
    @return: None
"""
def game_loop(data_queue):
    
    global CLOCK 
    global DISPLAY

    total_apples_collected = 0
    games_played = 0

    pygame.init()
    CLOCK = pygame.time.Clock()
    DISPLAY = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), 0, 32)
    pygame.display.set_caption('MLBD AI Snake Project')

    while True:
        snake = Snake()
        feed = Feed()
        DISPLAY.fill(BGCOLOR)
        pygame.display.flip()

        apples_collected = runGame(snake, feed)
        total_apples_collected += apples_collected
        games_played += 1

        data_queue.put((games_played, apples_collected))

        gameOver(apples_collected, total_apples_collected, games_played)

"""
    The loop that manages and displays the graph, updating it with data from the queue.

    @param data_queue: multiprocessing.Queue, The queue to receive game data.
    @return: None
"""
def graph_loop(data_queue):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    graph = GraphWindow()
    
    def update_graph(frame):
        while not data_queue.empty():
            game_number, apples = data_queue.get()
            graph.add_game_data(game_number, apples)
        graph.update_graph()

    ani = animation.FuncAnimation(graph.fig, update_graph, interval=1000)

    plt.show()

def main():
    data_queue = multiprocessing.Queue()

    game_process = multiprocessing.Process(target=game_loop, args=(data_queue,))
    graph_process = multiprocessing.Process(target=graph_loop, args=(data_queue,))

    game_process.start()
    graph_process.start()

    game_process.join()
    graph_process.join()

def runGame(snake, feed):
    screenRect, screenSurf = drawRect(MARGIN, TOP_MARGIN, SCREEN_SIZE, SCREEN_SIZE, SCREENCOLOR)
    infoRect, infoSurf = drawRect(MARGIN, MARGIN, SCREEN_SIZE, TOP_MARGIN - 20)

    path = None
    keys = [K_UP, K_DOWN, K_LEFT, K_RIGHT]
    sa = SnakeAI(snake.direction)
    apples_collected = 0

    while True:
        for e in pygame.event.get():
            if e.type == QUIT:
                terminate()
            if e.type == KEYDOWN and e.key in keys:
                execEvent(snake, e.key)

        path = sa.getNextDirection(snake.coords, feed.coord)
        
        if path >= 0 and path < len(keys):
            execEvent(snake, keys[path])

        if not snake.move():
            snake.draw()
            return apples_collected

        renderRect(screenSurf, screenRect, SCREENCOLOR)
        renderRect(infoSurf, infoRect, BGCOLOR)

        if eatCheck(snake, feed, sa):
            apples_collected += 1

        drawGrid()
        showTitle()
        showGameInfo(snake.length)
        pygame.display.update(screenRect)
        pygame.display.update(infoRect)
        CLOCK.tick(FPS)

def eatCheck(snake, feed, sa):
    snake.draw()
    feed.draw()
    
    if snake.coords[0] == feed.coord:
        snake.eat()
        
        while True:
            feed.create()
        
            if feed.coord not in snake.coords:
                break
        
        return True
    
    return False

def execEvent(snake, key):
    event = {K_UP: UP, K_DOWN: DOWN, K_LEFT: LEFT, K_RIGHT: RIGHT}
    
    if key in event:
        snake.control(event[key])

def terminate():
    pygame.quit()
    sys.exit()

def renderRect(surf, rect, color):
    surf.fill(color)
    DISPLAY.blit(surf, rect)

def drawRect(left, top, width, height, color=BLACK):
    surf = pygame.Surface((width, height))
    rect = pygame.Rect(left, top, width, height)
    renderRect(surf, rect, color)
    return (rect, surf)

def makeText(font, text, color, bgcolor, x, y):
    surf = font.render(text, True, color, bgcolor)
    rect = surf.get_rect()
    rect.center = (x, y)
    DISPLAY.blit(surf, rect)
    return rect

def showTitle():
    font = pygame.font.Font('freesansbold.ttf', 25)
    text = ('AI Snake made with A* Algorithm')
    x = (MARGIN + SCREEN_SIZE) // 2
    y = 35
    return makeText(font, text, BLACK, BGCOLOR, x, y)

def showGameInfo(length):
    font = pygame.font.Font('freesansbold.ttf', 20)
    text = ("Score: " + str(length - 2))
    x = (MARGIN + SCREEN_SIZE) // 2
    y = 70
    return makeText(font, text, BLACK, BGCOLOR, x, y)

def drawGrid():
    for x in range(MARGIN + GRID_SIZE, WINDOW_WIDTH - MARGIN, GRID_SIZE):
        pygame.draw.line(DISPLAY, BGCOLOR, (x, TOP_MARGIN), (x, 600))
    for y in range(TOP_MARGIN, WINDOW_HEIGHT, GRID_SIZE):
        pygame.draw.line(DISPLAY, BGCOLOR, (0, y), (600, y))

def gameOver(score, total_apples_collected, games_played):
    font = pygame.font.Font('freesansbold.ttf', 50)
    DISPLAY.fill(SCREENCOLOR)

    x = (SCREEN_SIZE // 2) + MARGIN
    y = (WINDOW_HEIGHT // 2) - 60
    makeText(font, 'Game Over', GRAY, None, x, y)

    score_text = f"Score: {score}"
    makeText(font, score_text, GRAY, None, x, y + 60)

    avg_score = total_apples_collected / games_played if games_played > 0 else 0
    avg_text = f"Avg Apples: {avg_score:.2f}"
    makeText(font, avg_text, GRAY, None, x, y + 120)

    pygame.display.update()

    pygame.time.wait(3000)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    main()