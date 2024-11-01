from utils import game_settings, helper_methods, custom_exceptions, constants
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
from utils.logging_config import setup_logger
agent_logger = setup_logger(__name__, game_settings.agent_errors_filepath)

class Agent:
    """
        `Agent` class decides which chess move to play based on the current state.

        Args:
            color (str): The color of the agent, either 'W' or 'B'.
            learn_rate (float): The learning rate between 0 and 1.
            discount_factor (float): The discount factor between 0 and 1.
            q_table (Optional[pd.DataFrame]): An existing Q-table DataFrame.

        Attributes:
            is_trained (bool): Indicates whether the agent has been trained.
            q_table (pd.DataFrame): The Q-values for the agent's decisions.
    """
    def __init__(self, color: str, learn_rate: float = constants.default_learning_rate, discount_factor: float = constants.default_discount_factor, q_table: Optional[pd.DataFrame] = None):
        """
            Initializes the Agent with the specified parameters.
            Args:
                color (str): The color of the agent, either 'W' for white or 'B' for black.
                learn_rate (float, optional): The learning rate between 0 and 1.
                discount_factor (float, optional): The discount factor between 0 and 1.
                q_table (Optional[pd.DataFrame], optional): An existing Q-table DataFrame. 
        """
        self.color = color
        self.learn_rate = learn_rate
        self.discount_factor = discount_factor
        self.is_trained: bool = False
        self.q_table = q_table if q_table is not None else pd.DataFrame()
        ### end of __init__ ###

    def choose_action(self, chess_data: pd.DataFrame, environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        """
            Chooses the next chess move based on the current environment state.
            Args:
                chess_data (pd.DataFrame): DataFrame containing chess moves for each game.
                environ_state (Dict[str, Union[int, str, List[str]]]): The current state of the environment.
                curr_game (str, optional): The identifier for the current game. Defaults to 'Game 1'.
            Returns:
                str: The chosen chess move.
        """
        if not chess_data:
            chess_data = {}

        legal_moves = environ_state['legal_moves']
        if not legal_moves:
            agent_logger.info(f'Agent.choose_action: legal_moves is empty. curr_game: {curr_game}, curr_turn: {environ_state['curr_turn']}\n')
            return ''
      
        self.update_q_table(legal_moves)
        
        if self.is_trained:
            return self.policy_game_mode(legal_moves, environ_state['curr_turn'])
        else:
            return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
    ### end of choose_action ###
    
    def policy_training_mode(self, chess_data: pd.DataFrame, curr_game: str, curr_turn: str) -> str:
        """
            determines how the agent chooses a move at each turn during training. It retrieves the move 
            corresponding to the current game and turn from the chess data.

            Args:
                chess_data (pd.DataFrame): A Pandas DataFrame containing the chess moves for each game and turn.
                curr_game (str): A string representing the current game being played.
                curr_turn (str): A string representing the current turn, e.g. 'W1'.
            Returns:
                str: A string representing the chess move chosen by the agent. The move is retrieved from the chess 
                data based on the current game and turn.
        """
        try:
            chess_move = chess_data.at[curr_game, curr_turn]
            return chess_move
        except KeyError as e:
            error_message = f'Failed to choose action at policy_training_mode. curr_game: {curr_game}, curr_turn: {curr_turn} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e
    ### end of policy_training_mode ###

    def policy_game_mode(self, legal_moves: List[str], curr_turn: str) -> str:
        """
            determines how the agent chooses a move during a game between a human player and the agent. 
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
        """
        dice_roll = helper_methods.get_number_with_probability(constants.chance_for_random_move)
        
        legal_moves_in_q_table = self.q_table[curr_turn].loc[self.q_table[curr_turn].index.intersection(legal_moves)]
        if legal_moves_in_q_table.empty:
            error_message = f'at policy_game_mode: legal moves not found in q_table or legal_moves is empty.'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message)

        if dice_roll == 1:
            chess_move = legal_moves_in_q_table.sample().index[0]
        else:
            chess_move = legal_moves_in_q_table.idxmax()
        return chess_move
    ### end of policy_game_mode ###

    def change_q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        """
            adds points to a cell in the q-table. The cell is determined by the chess move and the current 
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
        except KeyError as e:
            error_message = f'@ change_q_table_pts(). Failed to change q_table points. chess_move: {chess_move}, curr_turn: {curr_turn}, pts: {pts} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e
    ### end of change_q_table_pts ###

    def update_q_table(self, new_chess_moves: Union[str, List[str]]) -> None:
        """
            checks if the provided chess moves are already in the Q-table.
            If all moves are present, it returns without making changes.
            If any moves are new, it adds them to the Q-table with initial Q-values of 0.

            Args:
                new_chess_moves (Union[str, list[str]]): A single chess move as a string or a list of strings representing the new chess moves.
            Side Effects:
                May modify the Q-table by appending new moves if they're not already present.
        """
        if isinstance(new_chess_moves, str):
            new_chess_moves = [new_chess_moves]
        
        truly_new_moves = set(new_chess_moves) - set(self.q_table.index)
        if not truly_new_moves:
            return
        
        q_table_new_values: pd.DataFrame = pd.DataFrame(
            0, 
            index = truly_new_moves,
            columns = self.q_table.columns, 
            dtype = np.int64
        )

        self.q_table = pd.concat([self.q_table, q_table_new_values])
    ### update_q_table ###


