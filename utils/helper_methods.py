import pandas as pd
from utils import constants
import random

def agent_selects_and_plays_chess_move(chess_agent, environ) -> str:
    curr_state = environ.get_curr_state() 
    chess_move: str = chess_agent.choose_action(curr_state)
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return chess_move

def receive_opponent_move(chess_move: str, environ) -> bool:                                                                                 
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return True

def bootstrap_agent(chess_agent, existing_q_table_path: str):
    chess_agent.q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    chess_agent.is_trained = True
    return chess_agent

def get_number_with_probability(probability: float) -> int:
    if random.random() < probability:
        return 1
    else:
        return 0

def is_game_over(environ) -> bool:
    return (
        environ.board.is_game_over() or
        environ.turn_index >= constants.max_turn_index or
        (len(environ.get_legal_moves()) == 0)
    )

def get_game_outcome(environ) -> str:
    return environ.board.outcome().result()

def get_game_termination_reason(environ) -> str:
    return str(environ.board.outcome().termination)