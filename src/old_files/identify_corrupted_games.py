import game_settings
import pandas as pd
import time
import Bradley

# !!! MAKE SURE to set desired chess_data path in game settings before executing this script !!! # 

if __name__ == '__main__':     
    start_time = time.time()
   
    print(f'Total number of rows before cleanup: {game_settings.chess_data[0]}')
    bradley = Bradley.Bradley()

    try:
        bradley.identify_corrupted_games()
        game_settings.CHESS_DATA.drop(bradley.corrupted_games_list, inplace = True)
        print(f'Total number of rows after cleanup: {game_settings.chess_data[0]}')
        bradley.engine.quit()
    except Exception as e:
        print(f'corrupted games identification interrupted because of:  {e}')
        bradley.engine.quit()
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print('corrupted games identification is complete')
    print(f'it took: {total_time} seconds\n')
    print(f'number of corrupted games: {len(bradley.corrupted_games_list)}')
    print(f'corrupted games: {bradley.corrupted_games_list}\n')

    game_settings.chess_data.to_pickle(game_settings.chess_data_filepath, compression = 'zip')

    quit()