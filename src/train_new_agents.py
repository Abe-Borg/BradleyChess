import helper_methods
import game_settings
import pandas as pd
import time
import Bradley

chess_data_file_path = game_settings.chess_games_filepath_part_10
game_settings.CHESS_DATA = pd.read_pickle(chess_data_file_path, compression = 'zip')

game_settings.CHESS_DATA = game_settings.CHESS_DATA.head(1)

if __name__ == '__main__':
    bradley = Bradley.Bradley()
    start_time = time.time()
    
    try:
        bradley.train_rl_agents()
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