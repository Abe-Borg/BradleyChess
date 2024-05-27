A reinforced learning implementation of a chess engine. The implementation uses the SARSA algorithm. 

The chess reinforced learning agents learn by playing games from a chess database exactly as shown. That's the first step of training in a two-step process. The second part of training lets the reinforced learning agents choose their own chess moves. The agents (White and Black players) train each other by playing against each other.

Main Components of Program
Bradley Class
This is a composite class that manages components and data flow during training and game play. All communication with external applications is also managed by this class. The methods of this class are focused on coordinating actions during the training and gameplay phase. However, the methods don't change the chessboard or choose the chess moves.

Environ Class
This class manages the chessboard and is the only part of the program that will actually change the chessboard. It also sets and manages what is called the state. In this application the state is the current chessboard configuration, the current turn, and the legal moves at each turn.

Agent Class
This class is reponsible for choosing the chess moves. The first training phase (remember, there are two training phases) the agents will play through a database of games exactly as shown and learn that way. I tried many different versions of this part of the project, and this implemenation was the most effective. This style of training teaches the agents to play good positional chess. Later during phase two, the user can change hyperparameters to make the agents more aggressive or even to prefer certain openings for example.

The single point of communication between this program and something external (like a web app) is facilitated by two methods, Bradley.recv_opp_move() and Bradley.rl_agent_chess_move(). 

Chess games are stored in Pandas DataFrames.