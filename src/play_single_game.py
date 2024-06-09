import helper_methods
import game_settings
import pandas as pd
import time
import Bradley
import Environ
import chess
import Agent

 
if __name__ == '__main__':

    chess_data_file_path = game_settings.chess_games_filepath_part_10

    chess_data = pd.read_pickle(chess_data_file_path, compression = 'zip')
    game = chess_data.head(1)    
    bradley = Bradley.Bradley(game)

    try:
        bradley.simply_play_games()
        bradley.engine.quit()    
    except Exception as e:
        print(f'simply play games, interrupted because of:  {e}')
        quit()
    
    print('play through 1 game is complete')
