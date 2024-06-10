import game_settings
import pandas as pd
import time

chess_data_file_path = game_settings.chess_games_filepath_part_1
game_settings.CHESS_DATA = pd.read_pickle(chess_data_file_path, compression = 'zip')
 
if __name__ == '__main__':
    start_time = time.time()

    white_color = 'W'
    black_color = 'B'

    try:
        turns_list_white = [f'{white_color}{i + 1}' for i in range(game_settings.max_num_turns_per_player)]
        turns_list_black = [f'{black_color}{i + 1}' for i in range(game_settings.max_num_turns_per_player)]

        # Extract columns for the specified color/player
        move_columns_white = [col for col in game_settings.CHESS_DATA.columns if col.startswith(white_color)]
        move_columns_black = [col for col in game_settings.CHESS_DATA.columns if col.startswith(black_color)]

        # Extract unique moves for the specified color/player
        # Flatten the array and then create a Pandas Series to find unique values
        unique_moves_white = pd.Series(game_settings.CHESS_DATA[move_columns_white].values.flatten()).unique()
        unique_moves_black = pd.Series(game_settings.CHESS_DATA[move_columns_black].values.flatten()).unique()

        q_table_white: pd.DataFrame = pd.DataFrame(0, index = unique_moves_white, columns = turns_list_white, dtype = np.int64)
        q_table_black: pd.DataFrame = pd.DataFrame(0, index = unique_moves_black, columns = turns_list_black, dtype = np.int64)

        q_table_white.to_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
        q_table_black.to_pickle(game_settings.imman_agent_q_table_path, compression = 'zip')
    except Exception as e:
        print(f'initialize q table, interrupted because of:  {e}')
        quit()

    end_time = time.time()
    total_time = end_time - start_time
    print('initialized q table is complete')
    print(f'it took: {total_time} seconds\n')



