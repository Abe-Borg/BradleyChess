from utils import helper_methods, constants
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional

class Agent:
    def __init__(self, color: str, learn_rate: float = constants.default_learning_rate, discount_factor: float = constants.default_discount_factor, q_table: Optional[pd.DataFrame] = None):
        self.color = color
        self.learn_rate = learn_rate
        self.discount_factor = discount_factor
        self.is_trained: bool = False
        self.q_table = q_table if q_table is not None else pd.DataFrame()

    def choose_action(self, chess_data: pd.DataFrame, environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        if not chess_data:
            chess_data = {}
        legal_moves = environ_state['legal_moves']
        if not legal_moves:
            return ''
        self.update_q_table(legal_moves)
        if self.is_trained:
            return self.policy_game_mode(legal_moves, environ_state['curr_turn'])
        else:
            return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
    
    def policy_training_mode(self, chess_data: pd.DataFrame, curr_game: str, curr_turn: str) -> str:
        chess_move = chess_data.at[curr_game, curr_turn]
        return chess_move

    def policy_game_mode(self, legal_moves: List[str], curr_turn: str) -> str:
        dice_roll = helper_methods.get_number_with_probability(constants.chance_for_random_move)        
        legal_moves_in_q_table = self.q_table[curr_turn].loc[self.q_table[curr_turn].index.intersection(legal_moves)]
        if dice_roll == 1:
            chess_move = legal_moves_in_q_table.sample().index[0]
        else:
            chess_move = legal_moves_in_q_table.idxmax()
        return chess_move

    def change_q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        self.q_table.at[chess_move, curr_turn] += pts

    def update_q_table(self, new_chess_moves: Union[str, List[str]]) -> None:
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