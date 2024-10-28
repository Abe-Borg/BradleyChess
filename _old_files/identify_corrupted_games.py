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


'''

Script for Identifying and Removing Corrupted Chess Games from Dataset

This script identifies and removes corrupted games from a chess dataset stored in a pickle file. It utilizes a custom `Bradley` class for identifying corrupted games based on certain criteria. The cleaned dataset is then saved back to the original file.

Dependencies:
- game_settings
- pandas
- time
- Bradley
- logging

Functions:
    identifying_corrupted_games(chess_data_filepath):
        Identifies and removes corrupted games from the chess dataset.
        Args:
            chess_data_filepath (str): The path to the chess data pickle file.

Main Execution:
    - The script is designed to be run as a standalone program.
    - It processes a specified pickle file, identifies corrupted games, removes them, and saves the cleaned data.
    - Measures the time taken for the identification and cleanup process.

Function Details:

def identifying_corrupted_games(chess_data_filepath):
   
    Identifies and removes corrupted games from the chess dataset.

    Parameters:
        chess_data_filepath (str): The path to the chess data pickle file.

    This function performs the following steps:
    1. Loads the chess data from a pickle file.
    2. Initializes an instance of the Bradley class.
    3. Prints the total number of rows in the dataset before cleanup.
    4. Identifies corrupted games using the Bradley class's profile_corrupted_games_identification method.
    5. If corrupted games are found, they are removed from the dataset.
    6. Prints the total number of rows in the dataset after cleanup.
    7. Saves the cleaned dataset back to the original file.
    8. Prints the total time taken for the process and the number of corrupted games found.
    9. Logs any errors encountered during the process.

'''

   

