# training_functions.py

from typing import Callable, List
import pandas as pd
from agents import Agent
import chess
from utils import game_settings, constants
from environment.Environ import Environ
import re
from multiprocessing import Pool, cpu_count
import logging

# Set up file-based logging (critical items only)
logger = logging.getLogger("training_functions")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.training_functions_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def process_games_in_parallel(game_indices: List[str], worker_function: Callable[..., pd.DataFrame], *args):
    num_processes = min(cpu_count(), len(game_indices))
    chunks = chunkify(game_indices, num_processes)
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_function, [(chunk, *args) for chunk in chunks])
    
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, pd.DataFrame) or isinstance(result, tuple):
            valid_results.append(result)
        else:
            logger.critical(f"Result {i} is invalid: {type(result)}")
            
    return valid_results

def train_rl_agents(chess_data, est_q_val_table, white_q_table, black_q_table):
    game_indices = list(chess_data.index)

    results = process_games_in_parallel(game_indices, worker_train_games, chess_data, est_q_val_table, white_q_table, black_q_table)
    
    w_agent_q_tables = [result[0] for result in results if isinstance(result, tuple)]
    b_agent_q_tables = [result[1] for result in results if isinstance(result, tuple)]
    w_agent = Agent('W', q_table=white_q_table.copy())
    b_agent = Agent('B', q_table=black_q_table.copy())
    w_agent.q_table = merge_q_tables(w_agent_q_tables)
    b_agent.q_table = merge_q_tables(b_agent_q_tables)
    w_agent.is_trained = True
    b_agent.is_trained = True
    return w_agent, b_agent

def train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ):
    num_moves = chess_data.at[game_number, 'PlyCount']
    curr_state = environ.get_curr_state()
    while curr_state['turn_index'] < num_moves:
        try:
            w_next_q_value = handle_agent_turn(
                agent=w_agent,
                chess_data=chess_data,
                curr_state=curr_state,
                game_number=game_number,
                environ=environ,
                curr_q_value=w_curr_q_value,
                est_q_val_table=est_q_val_table
            )
            w_curr_q_value = w_next_q_value
            curr_state = environ.get_curr_state()
        except Exception as e:
            logger.critical(f'error during white agent turn in game {game_number}: {str(e)}')
            break

        if environ.board.is_game_over():
            break

        try:
            b_next_q_value = handle_agent_turn(
                agent=b_agent,
                chess_data=chess_data,
                curr_state=curr_state,
                game_number=game_number,
                environ=environ,
                curr_q_value=b_curr_q_value,
                est_q_val_table=est_q_val_table
            )

            b_curr_q_value = b_next_q_value
            curr_state = environ.get_curr_state()
        except Exception as e:
            logger.critical(f'error during black agent turn in game {game_number}: {str(e)}')
            break

        if environ.board.is_game_over():
            break


def find_next_q_value(curr_qval: int, learn_rate: float, reward: int, discount_factor: float, est_qval: int) -> int:
    return curr_qval + learn_rate * (reward + ((discount_factor * est_qval) - curr_qval))


def apply_move_and_update_state(chess_move: str, environ) -> None:
    environ.board.push_san(chess_move)
    environ.update_curr_state()

def get_reward(chess_move: str) -> int:
    total_reward = 0
    if re.search(r'[NRBQ]', chess_move):
        total_reward += constants.CHESS_MOVE_VALUES['piece_development']
    if 'x' in chess_move:
        total_reward += constants.CHESS_MOVE_VALUES['capture']
    if '=' in chess_move:
        total_reward += constants.CHESS_MOVE_VALUES['promotion']
        if '=Q' in chess_move:
            total_reward += constants.CHESS_MOVE_VALUES['promotion_queen']
    return total_reward


def assign_points_to_q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
    chess_agent.update_q_table([chess_move])
    chess_agent.change_q_table_pts(chess_move, curr_turn, curr_q_val)

def chunkify(lst, n):
    size = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
        
    return chunks

def worker_train_games(game_indices_chunk, chess_data, est_q_val_table, white_q_table, black_q_table):
    w_agent = Agent('W', q_table=white_q_table.copy())
    b_agent = Agent('B', q_table=black_q_table.copy())
    environ = Environ()

    for game_number in game_indices_chunk:
        try:
            w_curr_q_value = constants.initial_q_val
            b_curr_q_value = constants.initial_q_val
            train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ)
        except Exception as e:
            logger.critical(f"Error processing game {game_number} in worker_train_games: {str(e)}")
            continue

        environ.reset_environ()
    return w_agent.q_table, b_agent.q_table

def merge_q_tables(q_tables_list):
    merged_q_table = pd.concat(q_tables_list, axis=0)
    merged_q_table = merged_q_table.groupby(merged_q_table.index).sum(min_count=1)
    merged_q_table.fillna(0, inplace=True)
    return merged_q_table

def handle_agent_turn(agent, chess_data, curr_state, game_number, environ, curr_q_value, est_q_val_table):
    curr_turn = curr_state['curr_turn']
    chess_move = agent.choose_action(chess_data, curr_state, game_number)
    
    if chess_move not in environ.get_legal_moves():
        logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}. Skipping.")
        return curr_q_value, 0

    apply_move_and_update_state(chess_move, environ)
    reward = get_reward(chess_move)
    curr_state = environ.get_curr_state()

    if environ.board.is_game_over() or not curr_state['legal_moves']:
        est_q_value = 0
    else:
        next_turn = curr_state['curr_turn']
        est_q_value = est_q_val_table.at[game_number, next_turn]
    
    next_q_value = find_next_q_value(curr_q_value, agent.learn_rate, reward, agent.discount_factor, est_q_value)
    agent.change_q_table_pts(chess_move, curr_turn, next_q_value - curr_q_value)
    return next_q_value


