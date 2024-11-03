import pandas as pd
from utils import game_settings, constants
import random
from agents import Agent

def agent_selects_and_plays_chess_move(chess_agent, environ) -> str:
    """
        The Agent selects a chess move and loads it onto the chessboard. It is used 
        during actual gameplay between the computer and the user, not during training. 
        Returns:
            str: A string representing the selected chess move.
        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
    """
    curr_state = environ.get_curr_state() 
    chess_move: str = chess_agent.choose_action(curr_state)
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return chess_move

def receive_opponent_move(chess_move: str, environ) -> bool:                                                                                 
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return True

def bootstrap_agent(chess_agent, existing_q_table_path: str) -> Agent.Agent:
    chess_agent.q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    chess_agent.is_trained = True
    return chess_agent

def get_number_with_probability(probability: float) -> int:
    """Generate a random number with a given probability.
    Args:
        probability (float): A float representing the probability of generating a 1.
    Returns:
        int: A random integer value of either 0 or 1.
    """
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