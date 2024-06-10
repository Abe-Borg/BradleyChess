import helper_methods
import game_settings
import Bradley

chess_data_file_path = game_settings.chess_games_filepath_part_10
game_settings.CHESS_DATA = pd.read_pickle(chess_data_file_path, compression = 'zip')

game_settings.CHESS_DATA = game_settings.CHESS_DATA.head(1)
 
if __name__ == '__main__':
    bradley = Bradley.Bradley()
    try:
        bradley.simply_play_games()
        bradley.engine.quit()    
    except Exception as e:
        print(f'simply play games, interrupted because of:  {e}')
        quit()
    
    print('play through 1 game is complete')
