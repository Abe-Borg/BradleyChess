import game_settings
import pandas as pd
import time
import Bradley
import Environ
import chess
import Agent

chess_data_file_path = game_settings.chess_games_filepath_part_10
game_settings.CHESS_DATA = pd.read_pickle(chess_data_file_path, compression = 'zip')

game_settings.CHESS_DATA = game_settings.CHESS_DATA.head(1)
game_settings.CHESS_DATA.set_flags(write = False)
 
if __name__ == '__main__':
    bradley = Bradley.Bradley()
    start_time = time.time()
    try:
        bradley.simply_play_games()
        bradley.engine.quit()    
    except Exception as e:
        print(f'simply play games, interrupted because of:  {e}')
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time
    print('initialized q table is complete')
    print(f'it took: {total_time} seconds\n')
    
    print('play through 1 game is complete')
