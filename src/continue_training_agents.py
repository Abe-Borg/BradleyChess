import helper_methods
import game_settings
import pandas as pd
import time
import Bradley

chess_data_file_path = game_settings.chess_games_filepath_part_10
game_settings.CHESS_DATA = pd.read_pickle(chess_data_file_path, compression = 'zip')

game_settings.CHESS_DATA = game_settings.CHESS_DATA.head(1)
game_settings.CHESS_DATA.set_flags(write = False)

if __name__ == '__main__':
    bradley = helper_methods.init_bradley()
    helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)

    start_time = time.time()
    try:
        bradley.continue_training_rl_agents(game_settings.agent_vs_agent_num_games)
    except Exception as e:
        print(f'training interrupted because of:  {e}')
        quit()
        
    helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    end_time = time.time()
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time}')
    quit()