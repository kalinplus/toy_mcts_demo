# import numpy as np
# from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
# from mctspy.tree.search import MonteCarloTreeSearch
# from mctspy.games.examples.tictactoe import TicTacToeGameState
#
# state = np.zeros((3, 3))
# initial_board_state = TicTacToeGameState(state=state, next_to_move=1)
#
# root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_board_state)
# mcts = MonteCarloTreeSearch(root)
# best_node = mcts.best_action(10000)

import numpy as np
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from mctspy.games.examples.connect4 import Connect4GameState
import tkinter as tk

# define inital state
state = np.zeros((7, 7))
board_state = Connect4GameState(
    state=state, next_to_move=np.random.choice([-1, 1]), win=4)

# link pieces to icons
pieces = {0: " ", 1: "X", -1: "O"}

# Initialize tkinter window and canvas for display
root = tk.Tk()
root.title("Connect 4 - Monte Carlo AI")
canvas_size = 500
cell_size = canvas_size // 7
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="blue")
canvas.pack()

# print a single row of the board
# def stringify(row):
#     return " " + " | ".join(map(lambda x: pieces[int(x)], row)) + " "

# Draw the Connect 4 board grid
def draw_grid():
    for row in range(7):
        for col in range(7):
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="white", tags="grid")

def update_board(board):
    canvas.delete("pieces")
    for row in range(7):
        for col in range(7):
            piece = int(board[row, col])
            # yellow: 1, or O; red: -1, or X
            color = "red" if piece == 1 else "yellow" if piece == -1 else "white"
            x0, y0 = col * cell_size, row * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill=color, tags="pieces")
    root.update()

# display(board_state.board)
def make_ai_move():
    global board_state
    # Calculate best move for the current player
    root_node = TwoPlayersGameMonteCarloTreeSearchNode(state=board_state)
    mcts = MonteCarloTreeSearch(root_node)
    best_node = mcts.best_action(total_simulation_seconds=1)

    # Update board state
    board_state = best_node.state
    update_board(board_state.board)

    # Check for game result
    if board_state.game_result is not None:
        show_winner(board_state.game_result)
    else:
        # Schedule the next AI move
        root.after(1000, make_ai_move)  # Wait 1 second before next move


def show_winner(result):
    if result in pieces:
        # color = "yellow" if pieces[result]
        winner_text = f"Winner: {pieces[result]}"
    else:
        winner_text = "It's a Draw!"
    winner_window = tk.Toplevel(root)
    winner_window.title("Game Over")

    winner_canvas = tk.Canvas(winner_window, width=300, height=200)
    winner_canvas.pack()

    winner_canvas.create_text(150, 50, text=winner_text, fill="black", font=("arial", 20))

# Draw initial grid and update board
draw_grid()
update_board(board_state.board)

# Start the automatic play by making the first AI move
make_ai_move()

# Start the Tkinter main loop
root.mainloop()