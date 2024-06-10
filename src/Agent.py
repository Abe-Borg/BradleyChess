import game_settings
import pandas as pd
import numpy as np
import helper_methods

class Agent:
    """
        The `Agent` class is responsible for deciding what chess move to play 
        based on the current state. The state is passed to the agent by 
        the `Environ` class.

        Args:
            - color (str): A string indicating the color of the agent, either 'W' or 'B'.
            - learn_rate (float): A float between 0 and 1 that represents the learning rate.
            - discount_factor (float): A float between 0 and 1 that represents the discount factor.
        Attributes:
            - color (str): A string indicating the color of the agent, either 'W' or 'B'.
            - learn_rate (float): A float between 0 and 1 that represents the learning rate.
            - discount_factor (float): A float between 0 and 1 that represents the discount factor.
            - is_trained (bool): A boolean indicating whether the agent has been trained.
            - Q_table (pd.DataFrame): A Pandas DataFrame containing the Q-values for the agent.
    """
    def __init__(self, color: str, learn_rate = 0.6, discount_factor = 0.35):
        """
            Initializes an Agent object with a color, a learning rate, a discount factor, and an errors file.
            This method initializes an Agent object by setting the color, the learning rate, and the 
            discount factor. It also opens the errors file in append mode and initializes the Q-table.
            
            Side Effects:
            Opens the errors file in append mode.
            Modifies the learn_rate, discount_factor, color, is_trained, and Q_table attributes.
        """
        self.errors_file = open(game_settings.agent_errors_filepath, 'a')
        self.step_by_step_file = open(game_settings.agent_step_by_step_filepath, 'a')
        
        self.learn_rate = learn_rate
        self.discount_factor = discount_factor
        self.color = color
        self.is_trained: bool = False
        self.Q_table: pd.DataFrame = None # Q table will be assigned at program execution.

        self.step_by_step_file.write(f'Agent.__init__: color: {color}, learn_rate: {learn_rate}, discount_factor: {discount_factor}, is_trained: {self.is_trained}\n')
    ### end of __init__ ###

    def __del__(self):
        self.step_by_step_file.write('hi and bye from Agent.__del__\n')
        self.step_by_step_file.close()
        self.errors_file.close()
    ### end of __del__ ###

    def choose_action(self, environ_state: dict[str, str, list[str]], curr_game: str = 'Game 1') -> str:
        """
            Chooses the next chess move for the agent based on the current state.
            This method chooses the next chess move for the agent based on the current state of the environment. If 
            there are no legal moves, it logs an error and returns an empty string. If there are legal moves that are 
            not in the Q-table, it updates the Q-table with these moves. Depending on whether the agent is trained, it 
            uses either the game mode policy or the training mode policy to choose the next move.

            Args:
                environ_state (dict[str, str, list[str]]): A dictionary representing the current state of the 
                environment. The dictionary has the following keys:
                    - 'turn_index': The current turn index.
                    - 'curr_turn': The current turn, represented as a string.
                    - 'legal_moves': A list of strings, where each string is a legal move at the current turn.
                curr_game (str, optional): A string representing the current game being played. Defaults to 'Game 1'.

            Returns:
                str: A string representing the chess move chosen by the agent. If there are no legal moves, returns an 
                empty string.

            Side Effects:
                Modifies the Q-table if there are legal moves that are not in the Q-table.
                Writes into the errors file if there are no legal moves.
        """
        if environ_state['legal_moves'] == []:
            self.errors_file.write(f'Agent.choose_action: legal_moves is empty. curr_game: {curr_game}, curr_turn: {environ_state['curr_turn']}\n')
            return ''

        # check if any of the legal moves is not already in the Q table
        moves_not_in_Q_table: list[str] = [move for move in environ_state['legal_moves'] if move not in self.Q_table.index]

        if moves_not_in_Q_table:
            self.update_Q_table(moves_not_in_Q_table)

        if self.is_trained:
            return self.policy_game_mode(environ_state['legal_moves'], environ_state['curr_turn'])
        else:
            return self.policy_training_mode(curr_game, environ_state["curr_turn"])
    ### end of choose_action ###
    
    def policy_training_mode(self, curr_game: str, curr_turn: str) -> str:
        """
            Determines how the agent chooses a move at each turn during training.
            This method determines how the agent chooses a move at each turn during training. It retrieves the move 
            corresponding to the current game and turn from the chess data.

            Args:
                curr_game (str): A string representing the current game being played.
                curr_turn (str): A string representing the current turn, e.g. 'W1'.

            Returns:
                str: A string representing the chess move chosen by the agent. The move is retrieved from the chess 
                data based on the current game and turn.

            Side Effects:
                None.
        """
        return game_settings.CHESS_DATA.at[curr_game, curr_turn]
    ### end of policy_training_mode ###

    def policy_game_mode(self, legal_moves: list[str], curr_turn: str) -> str:
        """
            Determines how the agent chooses a move during a game between a human player and the agent.
            This method determines how the agent chooses a move during a game between a human player and the agent. 
            The agent searches its Q-table to find the moves with the highest Q-values at each turn. Sometimes, based 
            on a probability defined in the game settings, the agent will pick a random move from the Q-table instead 
            of the move with the highest Q-value.

            Args:
                legal_moves (list[str]): A list of strings representing the legal moves for the current turn.
                curr_turn (str): A string representing the current turn, e.g. 'W1'.

            Returns:
                str: A string representing the chess move chosen by the agent. The move is either the one with the 
                highest Q-value or a random move from the Q-table, depending on a probability defined in the game 
                settings.

            Side Effects:
                None.
        """
        dice_roll = helper_methods.get_number_with_probability(game_settings.chance_for_random_move)
        legal_moves_in_q_table = self.Q_table[curr_turn].loc[self.Q_table[curr_turn].index.intersection(legal_moves)]

        if dice_roll == 1:
            chess_move = legal_moves_in_q_table.sample().index[0]
        else:
            chess_move = legal_moves_in_q_table.idxmax()
        return chess_move
    ### end of policy_game_mode ###

    def change_Q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        """
            Adds points to a cell in the Q-table.
            This method adds points to a cell in the Q-table. The cell is determined by the chess move and the current 
            turn. The points are added to the existing Q-value of the cell.

            Args:
                chess_move (str): A string representing the chess move, e.g. 'e4'. This determines the row of the cell 
                in the Q-table.
                curr_turn (str): A string representing the current turn, e.g. 'W10'. This determines the column of the 
                cell in the Q-table.
                pts (int): An integer representing the number of points to add to the Q-table cell.

            Side Effects:
                Modifies the Q-table by adding points to the cell determined by the chess move and the current turn.
        """
        self.Q_table.at[chess_move, curr_turn] += pts
    ### end of change_Q_table_pts ###

    def update_Q_table(self, new_chess_moves: list[str]) -> None:
        """
            Updates the Q-table with new chess moves.
            This method updates the Q-table with new chess moves. It creates a new DataFrame with the new chess moves 
            and the same columns as the Q-table, and appends this DataFrame to the Q-table. The new chess moves are 
            added as new rows in the Q-table, and the Q-values for these moves are initialized to 0.

            Args:
                new_chess_moves (list[str]): A list of strings representing the new chess moves. These moves are not 
                already in the Q-table.

            Side Effects:
                Modifies the Q-table by appending a new DataFrame with the new chess moves.
        """
        q_table_new_values: pd.DataFrame = pd.DataFrame(0, index = new_chess_moves, columns = self.Q_table.columns, dtype = np.int64)
        self.Q_table = pd.concat([self.Q_table, q_table_new_values])
    ### update_Q_table ###

    def reset_Q_table(self) -> None:
        """Resets the Q table to all zeros.
        """
        self.Q_table.iloc[:, :] = 0    
    ### end of reset_Q_table ###
