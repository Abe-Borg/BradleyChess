import helper_methods
import game_settings
import pandas as pd
import time
import Environ
import training_functions
import Agent
import chess

if __name__ == '__main__':
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    # change this each time for new section of the database
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression = 'zip')

    try:
        training_functions.train_rl_agents(chess_data, bradley, imman, environ)
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