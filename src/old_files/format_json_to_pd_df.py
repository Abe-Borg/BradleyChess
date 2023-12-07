import pandas as pd
import json
import time
import game_settings

def convert_json_to_df(input_file_path, output_file_path):
    # Load and read the JSON file
    with open(input_file_path, 'r') as file:
        chess_data = json.load(file)

    # Initialize an empty list to store the data for DataFrame
    data_for_df = []

    # Process each game in the dataset
    for game in chess_data:
        # Initialize a dictionary for the current game with Num Moves and Result
        game_dict = {'Num Moves': game['PlyCount'], 'Result': game['Result']}
    
        # Split the moves into Wn and Bn columns
        moves = game['ChessMoves']
        for i, move in enumerate(moves):
            # Determine column name based on the move number and player
            col_prefix = 'W' if i % 2 == 0 else 'B'
            move_num = i // 2 + 1
            col_name = f"{col_prefix}{move_num}"
            game_dict[col_name] = move
    
        # Append the game dictionary to the list
        data_for_df.append(game_dict)

    # Create the DataFrame from the list of game dictionaries
    chess_df = pd.DataFrame(data_for_df)

    # Set the index to be the game number
    chess_df.index = ['Game ' + str(i+1) for i in chess_df.index]

    # Fill NaN values with an empty string for visual consistency
    chess_df = chess_df.fillna('')

    # fix the column order
    chess_df = chess_df[[c for c in chess_df if c not in ['Result']] + ['Result']]
    chess_df = chess_df[(chess_df['Num Moves'] > 0) & (chess_df['Num Moves'] <= game_settings.max_num_turns_per_player * 2)]

    # Export the DataFrame to a pickle file
    chess_df.to_pickle(output_file_path, compression = 'zip')


if __name__ == '__main__':
    start_time = time.time()

    # convert_json_to_df(game_settings.chess_json_file_path_part_1, game_settings.chess_pd_dataframe_file_path_part_1)
    # convert_json_to_df(game_settings.chess_json_file_path_part_2, game_settings.chess_pd_dataframe_file_path_part_2)
    # convert_json_to_df(game_settings.chess_json_file_path_part_3, game_settings.chess_pd_dataframe_file_path_part_3)
    # convert_json_to_df(game_settings.chess_json_file_path_part_4, game_settings.chess_pd_dataframe_file_path_part_4)
    # convert_json_to_df(game_settings.chess_json_file_path_part_5, game_settings.chess_pd_dataframe_file_path_part_5)
    # convert_json_to_df(game_settings.chess_json_file_path_part_6, game_settings.chess_pd_dataframe_file_path_part_6)
    # convert_json_to_df(game_settings.chess_json_file_path_part_7, game_settings.chess_pd_dataframe_file_path_part_7)
    # convert_json_to_df(game_settings.chess_json_file_path_part_8, game_settings.chess_pd_dataframe_file_path_part_8)
    # convert_json_to_df(game_settings.chess_json_file_path_part_9, game_settings.chess_pd_dataframe_file_path_part_9)
    # convert_json_to_df(game_settings.chess_json_file_path_part_10, game_settings.chess_pd_dataframe_file_path_part_10)
    # convert_json_to_df(game_settings.chess_json_file_path_part_11, game_settings.chess_pd_dataframe_file_path_part_11)

    convert_json_to_df(game_settings.chess_json_file_path_part_12, game_settings.chess_pd_dataframe_file_path_part_12)
    
    # convert_json_to_df(game_settings.chess_json_file_path_part_13, game_settings.chess_pd_dataframe_file_path_part_13)
    # convert_json_to_df(game_settings.chess_json_file_path_part_14, game_settings.chess_pd_dataframe_file_path_part_14)
    # convert_json_to_df(game_settings.chess_json_file_path_part_15, game_settings.chess_pd_dataframe_file_path_part_15)
    # convert_json_to_df(game_settings.chess_json_file_path_part_16, game_settings.chess_pd_dataframe_file_path_part_16)
    # convert_json_to_df(game_settings.chess_json_file_path_part_17, game_settings.chess_pd_dataframe_file_path_part_17)
    # convert_json_to_df(game_settings.chess_json_file_path_part_18, game_settings.chess_pd_dataframe_file_path_part_18)
    # convert_json_to_df(game_settings.chess_json_file_path_part_19, game_settings.chess_pd_dataframe_file_path_part_19)
    
    end_time = time.time()
    total_time = end_time - start_time
     
    print('json to dataframe conversion is complete\n')
    print(f'it took: {total_time} seconds')
