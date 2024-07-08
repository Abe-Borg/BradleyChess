import game_settings
import pandas as pd
import time
import Bradley
import logging

logging.basicConfig(filename=game_settings.bradley_errors_filepath, level=logging.INFO)

def identifying_corrupted_games(chess_data_filepath):
    start_time = time.time()
    chess_data = pd.read_pickle(chess_data_filepath, compression = 'zip')
    bradley = Bradley.Bradley()
   
    print(f'Total number of rows before cleanup: {len(chess_data)}')
    
    try:
        bradley.profile_corrupted_games_identification(chess_data)
        print(f"Type of corrupted_games_list: {type(bradley.corrupted_games_list)}")
        print(f"Content of corrupted_games_list: {bradley.corrupted_games_list}")

        if bradley.corrupted_games_list:
            chess_data.drop(list(bradley.corrupted_games_list), inplace = True)
        else:
            print('no corrupted games found')
        
        print(f'Total number of rows after cleanup: {len(chess_data)}')
    except Exception as e:
        print(f'corrupted games identification interrupted because of:  {e}')
        import traceback
        traceback.print_exc()
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print('corrupted games identification is complete')
    print(f'it took: {total_time} seconds\n')
    print(f'number of corrupted games: {len(bradley.corrupted_games_list)}')

    chess_data.to_pickle(chess_data_filepath, compression = 'zip')
    quit()


if __name__ == '__main__':     
    identifying_corrupted_games(game_settings.chess_games_filepath_part_40)