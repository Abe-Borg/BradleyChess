from utils import helper_methods, game_settings, custom_exceptions, constants
import pandas as pd
import time
from training import training_functions
from agents import Agent
from utils.logging_config import setup_logger

train_new_agents_logger = setup_logger(__name__, game_settings.train_new_agents_logger_filepath)

if __name__ == '__main__':
    start_time = time.time()
    
    # !!!!!!!!!!!!!!!!! change this each time for new section of the database  !!!!!!!!!!!!!!!!!
    # estimated q table number must match chess_data number
    estimated_q_values_table = pd.read_pickle(game_settings.est_q_vals_filepath_part_1, compression = 'zip')
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression = 'zip')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    try:
        Bradley, Imman = training_functions.train_rl_agents(chess_data, estimated_q_values_table)    
    except custom_exceptions.TrainingError as e:
        print(f'training interrupted because of:  {e}')
        train_new_agents_logger.error(f'An error occurred: {e}')
        exit(1)
        
    helper_methods.pikl_q_table(Bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.pikl_q_table(Imman, 'B', game_settings.imman_agent_q_table_path)
    
    end_time = time.time()
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time} for {constants.training_sample_size} games\n')