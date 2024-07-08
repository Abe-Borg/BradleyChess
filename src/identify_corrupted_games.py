import game_settings
import pandas as pd
import time
import Bradley



def identifying_corrupted_games(chess_data_filepath):
    start_time = time.time()
    chess_data = pd.read_pickle(chess_data_filepath, compression = 'zip')
    bradley = Bradley.Bradley()
   
    print(f'Total number of rows before cleanup: {chess_data[0]}')
    
    try:
        bradley.profile_corrupted_games_identification(chess_data)
        chess_data.drop(list(bradley.corrupted_games_list), inplace = True)
        print(f'Total number of rows after cleanup: {chess_data[0]}')
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

    chess_data.to_pickle(chess_data_filepath, compression = 'zip')
    quit()


if __name__ == '__main__':     
    identifying_corrupted_games(pd.read_pickle(game_settings.chess_games_filepath_part_32, compression = 'zip'))