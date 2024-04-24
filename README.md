A reinforced learning implementation of a chess engine. The implementation uses the SARSA algorithm. This project was completed as part of the capstone course at CSU, Monterey Bay, for the completion of the Bachelor's of Computer Science program.

This project is a continuation of previous work completed in the data science course, https://github.com/abecsumb/DataScienceProject/blob/main/Chess_Data_Preparation.ipynb

This repo contains the machine learning part of the capstone project (my part). However, the overall project submitted to faculty was a team effort. It was a web application implemented using Flask and React. Sarom Thin (https://github.com/lom360) completed work for the backend, and Mehar Rekhi (https://github.com/mehr998) completed work on the frontend. A video presentation of the project is at this link, https://www.youtube.com/watch?v=HlXtLt6fiLE .The code in this repo represents the core of the BradleyChess team project, and it can also be a standalone application.

The chess reinforced learning agents learn by playing games from a chess database exactly as shown. That's the first step of training in a two-step process. The second part of training lets the reinforced learning agents choose their own chess moves. The agents (White and Black players) train each other by playing against each other.

Main Components of Program
Bradley Class
This is a composite class that manages components and data flow during training and game play. All communication with external applications is also managed by this class. The methods of this class are focused on coordinating actions during the training and gameplay phase. However, the methods don't change the chessboard or choose the chess moves.

Environ Class
This class manages the chessboard and is the only part of the program that will actually change the chessboard. It also sets and manages what is called the state. In this application the state is the current chessboard configuration, the current turn, and the legal moves at each turn.

Agent Class
This class is reponsible for choosing the chess moves. The first training phase (remember, there are two training phases) the agents will play through a database of games exactly as shown and learn that way. I tried many different versions of this part of the project, and this implemenation was the most effective. This style of training teaches the agents to play good positional chess. Later during phase two, the user can change hyperparameters to make the agents more aggressive or even to prefer certain openings for example.

Rationale for Project
I chose this project for my capstone because I am pursuing a career in data science. Specifically machine learning, and ideally, reinforced learning. I structured the program as a sandbox that would be interesting and useful for students who are learning the basics of data science and to get an introduction to reinforced learning. I made sure to keep the code as simple as possible, but also as flexible as possible. The student can adjust the different hyperparameters and observe how the agents behave. Their behaviour changes dramatically based on training time and the value of variables like learning rate and the discount factor. The student can also change the way the agent chooses actions during the game play mode. Currently, the agent will choose the largest value in the Q table at each turn. Also, this project and also the data science class project (linked above) cover the fundamental topics of an introductory data science course that uses Python as the language. The student will recognize the use of Pandas, Matplotlib, Seaborn, Numpy, and some basic machine learning algorithms like kNN classification and decision trees.

Data science enthusiast will recogize ways to make this project better and to create more sophisticated agents. I started this project knowing nothing about reinforced learning. Therefore, the implementation is basic. I did not fully understand the math and theory behind the SARSA algorithm either. Regardless, I believe the form of the SARSA algorithm is implemented correctly.

Finally, the single point of communication between this program and something external (like a web app) is facilitated by two methods, Bradley.recv_opp_move() and Bradley.rl_agent_chess_move(). 
