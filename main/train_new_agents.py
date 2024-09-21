from utils import helper_methods
from utils import game_settings
import pandas as pd
import time
from training import training_functions
from agents import Agent
import chess
from utils.logging_config import setup_logger

train_new_agents_logger = setup_logger(__name__, game_settings.train_new_agents_logger_filepath)

if __name__ == '__main__':
    start_time = time.time()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    # !!!!!!!!!!!!!!!!! change this each time for new section of the database  !!!!!!!!!!!!!!!!!
    estimated_q_values_table = pd.read_pickle(game_settings.est_q_vals_filepath_part_1, compression = 'zip')
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression = 'zip')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    try:
        training_functions.train_rl_agents(chess_data, estimated_q_values_table, bradley, imman)
    except Exception as e:
        print(f'training interrupted because of:  {e}')
        quit()
        
    end_time = time.time()
    helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time} for {game_settings.training_sample_size} games\n')
    quit()