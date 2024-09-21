import game_settings
import pandas as pd
import numpy as np
import helper_methods
import logging
from typing import Union
import custom_exceptions
from utils.logging_config import setup_logger 

# agent_logger = logging.getLogger(__name__)
# agent_logger.setLevel(logging.ERROR)
# error_handler = logging.FileHandler(game_settings.agent_errors_filepath)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# error_handler.setFormatter(formatter)
# agent_logger.addHandler(error_handler)

# # use DEBUG for detailed internal state information and INFO for high-level events.
agent_logger = setup_logger(__name__, game_settings.agent_errors_filepath)

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
            - q_table (pd.DataFrame): A Pandas DataFrame containing the q-values for the agent.
    """
    def __init__(self, color: str, learn_rate = 0.6, discount_factor = 0.35, q_table: pd.DataFrame = None):
        """
            Initializes an Agent object with a color, a learning rate, a discount factor
            This method initializes an Agent object by setting the color, the learning rate, and the 
            discount factor.
            
            Side Effects: 
            Modifies the learn_rate, discount_factor, color, is_trained, and q_table attributes.
        """
        try:
            self.learn_rate = learn_rate
            self.discount_factor = discount_factor
            self.color = color
            self.is_trained: bool = False
            self.q_table: pd.DataFrame = q_table # q table will be assigned at program execution.
        except Exception as e:
            agent_logger.error(f'at __init__: failed to initialize agent. Error: {e}\n', exc_info=True)
            raise custom_exceptions.AgentInitializationError(f'failed to initialize agent due to error: {e}') from e
    ### end of __init__ ###

    def choose_action(self, chess_data, environ_state: dict[str, str, list[str]], curr_game: str = 'Game 1') -> str:
        """
            Chooses the next chess move for the agent based on the current state.
            This method chooses the next chess move for the agent based on the current state of the environment. If 
            there are no legal moves, it logs an error and returns an empty string. If there are legal moves that are 
            not in the q-table, it updates the q-table with these moves. Depending on whether the agent is trained, it 
            uses either the game mode policy or the training mode policy to choose the next move.

            Args:
                environ_state (dict[str, str, list[str]]): A dictionary representing the current state of the environment.
                curr_game (str, optional): A string representing the current game being played. Defaults to 'Game 1'.
            Returns:
                str: A string representing the chess move chosen by the agent. If there are no legal moves, returns an 
                empty string.

            Side Effects:
                Modifies the q-table if there are legal moves that are not in the q-table.
                Writes into the errors file if there are no legal moves.
        """
        if environ_state['legal_moves'] == []:
            agent_logger.info(f'Agent.choose_action: legal_moves is empty. curr_game: {curr_game}, curr_turn: {environ_state['curr_turn']}\n')
            return ''
        
        try:
            self.update_q_table(environ_state['legal_moves']) # this func also checks if there are any new unique move strings
        except Exception as e:
            error_message = f'Failed to update Q-table. curr_game: {curr_game}, curr_turn: {environ_state["curr_turn"]} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e

        try:
            if self.is_trained:
                return self.policy_game_mode(environ_state['legal_moves'], environ_state['curr_turn'])
            else:
                return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
        except Exception as e:
            error_message = f'Failed to choose action. curr_game: {curr_game}, curr_turn: {environ_state["curr_turn"]} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e
    ### end of choose_action ###
    
    def policy_training_mode(self, chess_data, curr_game: str, curr_turn: str) -> str:
        """
            Determines how the agent chooses a move at each turn during training.
            This method determines how the agent chooses a move at each turn during training. It retrieves the move 
            corresponding to the current game and turn from the chess data.

            Args:
                chess_data (pd.DataFrame): A Pandas DataFrame containing the chess moves for each game and turn.
                curr_game (str): A string representing the current game being played.
                curr_turn (str): A string representing the current turn, e.g. 'W1'.

            Returns:
                str: A string representing the chess move chosen by the agent. The move is retrieved from the chess 
                data based on the current game and turn.

            Side Effects:
                None.
        """
        try:
            chess_move = chess_data.at[curr_game, curr_turn]
            return chess_move
        except Exception as e:
            error_message = f'Failed to choose action at policy_training_mode. curr_game: {curr_game}, curr_turn: {curr_turn} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e
    ### end of policy_training_mode ###

    def policy_game_mode(self, legal_moves: list[str], curr_turn: str) -> str:
        """
            Determines how the agent chooses a move during a game between a human player and the agent.
            This method determines how the agent chooses a move during a game between a human player and the agent. 
            The agent searches its q-table to find the moves with the highest q-values at each turn. Sometimes, based 
            on a probability defined in the game settings, the agent will pick a random move from the q-table instead 
            of the move with the highest q-value.

            Args:
                legal_moves (list[str]): A list of strings representing the legal moves for the current turn.
                curr_turn (str): A string representing the current turn, e.g. 'W1'.
            Returns:
                str: A string representing the chess move chosen by the agent. The move is either the one with the 
                highest q-value or a random move from the q-table, depending on a probability defined in the game 
                settings.
            Side Effects:
                None.
        """
        dice_roll = helper_methods.get_number_with_probability(game_settings.chance_for_random_move)
        
        try:
            legal_moves_in_q_table = self.q_table[curr_turn].loc[self.q_table[curr_turn].index.intersection(legal_moves)]
        except Exception as e:
            error_message = f'at policy_game_mode: legal moves not found in q_table or legal_moves is empty. curr_turn: {curr_turn} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e

        if dice_roll == 1:
            chess_move = legal_moves_in_q_table.sample().index[0]
        else:
            chess_move = legal_moves_in_q_table.idxmax()
        return chess_move
    ### end of policy_game_mode ###

    def change_q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        """
            Adds points to a cell in the q-table.
            This method adds points to a cell in the q-table. The cell is determined by the chess move and the current 
            turn. The points are added to the existing q-value of the cell.

            Args:
                chess_move (str): A string representing the chess move, e.g. 'e4'. This determines the row of the cell 
                in the q-table.
                curr_turn (str): A string representing the current turn, e.g. 'W10'. This determines the column of the 
                cell in the q-table.
                pts (int): An integer representing the number of points to add to the q-table cell.

            Side Effects:
                Modifies the q-table by adding points to the cell determined by the chess move and the current turn.
        """
        try:    
            self.q_table.at[chess_move, curr_turn] += pts
        except Exception as e:
            error_message = f'@ change_q_table_pts(). Failed to change q_table points. chess_move: {chess_move}, curr_turn: {curr_turn}, pts: {pts} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e
    ### end of change_q_table_pts ###

    def update_q_table(self, new_chess_moves: Union[str, list[str]]) -> None:
        """
            Updates the Q-table with new chess moves, if they are not already present.

            This method checks if the provided chess moves are already in the Q-table.
            If all moves are present, it returns without making changes.
            If any moves are new, it adds them to the Q-table with initial Q-values of 0.

            Args:
                new_chess_moves (Union[str, list[str]]): A single chess move as a string or a list of strings representing the new chess moves.
            Side Effects:
                May modify the Q-table by appending new moves if they're not already present.
        """
        if isinstance(new_chess_moves, str):
            new_chess_moves = [new_chess_moves]
        
        # Convert to set for efficient lookup
        new_moves_set = set(new_chess_moves)

        # Check if all moves are already in the Q-table
        existing_moves = set(self.q_table.index)
        truly_new_moves = new_moves_set - existing_moves

        # If no new moves, return early
        if not truly_new_moves:
            return

        try:
            q_table_new_values: pd.DataFrame = pd.DataFrame(
                0, 
                index = list(truly_new_moves), 
                columns = self.q_table.columns, 
                dtype = np.int64
            )

            self.q_table = pd.concat([self.q_table, q_table_new_values])
        except Exception as e:
            error_message = f'@ update_q_table(). Failed to update q_table. new_chess_moves: {new_chess_moves}, dur to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e
    ### update_q_table ###
