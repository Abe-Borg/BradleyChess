# generate_q_est_values.py
import pandas as pd
import time
import sys
import os
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.training_functions import generate_q_est_df
from utils import game_settings, constants

if __name__ == '__main__':
    start_time = time.time()
    
    # !!!!!!!!!!!!!!!!! change this each time for new section of the database  !!!!!!!!!!!!!!!!!
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression='zip').head(20)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    try:
        estimated_q_values = generate_q_est_df(chess_data)
    except Exception as e:
        print(f'q table generation interrupted because of:  {e}')
        print(traceback.format_exc())  # This will print the full traceback
        exit(1)

    print(estimated_q_values.iloc[0, :5])

    # estimated_q_values.to_pickle(game_settings.est_q_vals_filepath_part_1, compression = 'zip')
    
    end_time = time.time()
    total_time = end_time - start_time
    print('q table generation is complete')
    print(f'total time: {total_time} seconds')