import game_settings
import pandas as pd
import time
import Bradley

chess_data_file_path = game_settings.chess_games_filepath_part_10
game_settings.CHESS_DATA = pd.read_pickle(chess_data_file_path, compression = 'zip')
 
if __name__ == '__main__':        
    print(f'Total number of rows before cleanup: {game_settings.CHESS_DATA.shape[0]}')

    bradley = Bradley.Bradley()
    start_time = time.time()

    try:
        bradley.identify_corrupted_games()
        bradley.engine.quit()
        game_settings.CHESS_DATA.drop(bradley.corrupted_games_list, inplace = True)
        print(f'Total number of rows after cleanup: {game_settings.CHESS_DATA.shape[0]}')

    except Exception as e:
        print(f'corrupted games identification interrupted because of:  {e}')
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time
    print('corrupted games identification is complete')
    print(f'it took: {total_time} seconds\n')
    print(f'number of corrupted games: {len(bradley.corrupted_games_list)}')
    print(f'corrupted games: {bradley.corrupted_games_list}\n')

    game_settings.CHESS_DATA.to_pickle(chess_data_file_path, compression = 'zip')