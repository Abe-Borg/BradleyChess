import helper_methods
import game_settings
import pandas as pd
import time
import Bradley

chess_data_file_path = game_settings.chess_games_filepath_part_10
game_settings.CHESS_DATA = pd.read_pickle(chess_data_file_path, compression = 'zip')
game_settings.CHESS_DATA.set_flags(write = False)

est_q_vals_file_path = game_settings.est_q_vals_filepath_part_50

if __name__ == '__main__':
    bradley = Bradley.Bradley()
    start_time = time.time()
    try:
        bradley.generate_q_est_vals(est_q_vals_file_path)
    except Exception as e:
        print(f'generate q est interrupted because of:  {e}')
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time
    print('generate q est is complete')
    print(f'it took: {total_time} seconds\n')