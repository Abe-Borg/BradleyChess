from utils import game_settings, constants
import pandas as pd
import time
from training import training_functions
import logging
from agents import Agent
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("train_new_agents")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(game_settings.training_functions_logger_filepath)
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


if __name__ == '__main__':
    start_time = time.time()
    
    # !!!!!!!!!!!!!!!!! change this each time for new section of the database  !!!!!!!!!!!!!!!!!
    # estimated q table number must match chess_data number
    estimated_q_values_table = pd.read_pickle(game_settings.est_q_vals_filepath_part_1, compression = 'zip')
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression = 'zip')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    white_q_table = pd.read_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
    black_q_table = pd.read_pickle(game_settings.imman_agent_q_table_path, compression = 'zip')

    try:
        Bradley, Imman = training_functions.train_rl_agents(chess_data, estimated_q_values_table, white_q_table, black_q_table)
    except Exception as e:
        logger.critical(f'training interrupted because of:  {e}')
        exit(1)
    
    white_q_table.to_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
    black_q_table.to_pickle(game_settings.imman_agent_q_table_path, compression = 'zip')
    
    end_time = time.time()
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time} for {constants.training_sample_size} games\n')