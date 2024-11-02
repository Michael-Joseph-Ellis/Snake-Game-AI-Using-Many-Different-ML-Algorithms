import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
    A class to create and manage a graph window using matplotlib, which visualizes
    the number of apples collected per game.

    Attributes:
        games_played (list): A list storing the number of games played.
        apples_collected (list): A list storing the number of apples collected in each game.
        fig (matplotlib.figure.Figure): The figure object for the graph.
        ax (matplotlib.axes.Axes): The axes object for the graph.
        line (matplotlib.lines.Line2D): The line object for plotting the data.
"""
class GraphWindow:
    def __init__(self):
        self.games_played = []
        self.apples_collected = []
        
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Apples Collected per Game")
        self.ax.set_xlabel("Game Number")
        self.ax.set_ylabel("Apples Collected")
        self.line, = self.ax.plot([], [], 'r-')  
        
    """
        Adds new game data to the lists for plotting.

        @param game_number: int, The game number.
        @param apples: int, The number of apples collected in the game.
        @return: None
    """
    def add_game_data(self, game_number, apples):
        self.games_played.append(game_number)
        self.apples_collected.append(apples)

    """
        Updates the graph with the latest game data. Recomputes the data limits and
        resizes the view to fit the new data.

        @return: None
    """
    def update_graph(self):
        self.line.set_data(self.games_played, self.apples_collected)
        self.ax.relim()
        self.ax.autoscale_view()  
        plt.draw()
        
    """
        Displays the graph window and sets up an animation to update the graph periodically.

        The graph is updated every second using matplotlib.animation.FuncAnimation.

        @return: None
    """
    def show_graph(self):
        def animate(i):
            self.update_graph()

        ani = animation.FuncAnimation(self.fig, animate, interval=1000) 
        plt.show()

if __name__ == "__main__":
    graph = GraphWindow()
    graph.add_game_data(1, 5) 
    graph.add_game_data(2, 3)  
    graph.show_graph()
