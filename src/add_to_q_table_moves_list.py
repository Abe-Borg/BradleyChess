import pandas as pd
import numpy as np
import helper_methods
import game_settings
import Bradley
import time

def add_to_q_table_moves_list(chess_data):
    start_time = time.time()
    bradley = Bradley.Bradley()

    helper_methods.bootstrap_agent_fill_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent_fill_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)


    print(f'White Q table size before games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size before games: {bradley.B_rl_agent.Q_table.shape}\n')

    try:
        all_white_moves = set()
        all_black_moves = set()

        for _, game in chess_data.iterrows():
            # Extract moves
            white_moves = game.filter(regex=r'^W\d+$').dropna().unique()
            black_moves = game.filter(regex=r'^B\d+$').dropna().unique()
        
            all_white_moves.update(white_moves)
            all_black_moves.update(black_moves)

        # Update Q-tables with new moves
        new_white_moves = list(all_white_moves - set(bradley.W_rl_agent.Q_table.index))
        new_black_moves = list(all_black_moves - set(bradley.B_rl_agent.Q_table.index))

        bradley.W_rl_agent.update_Q_table(new_white_moves)
        bradley.B_rl_agent.update_Q_table(new_black_moves)
        
        print(f'New white moves: {new_white_moves}')
        print(f'New black moves: {new_black_moves}\n')

        print(f'White Q table size after games: {bradley.W_rl_agent.Q_table.shape}')
        print(f'Black Q table size after games: {bradley.B_rl_agent.Q_table.shape}\n')

        end_time = time.time()
        total_time = end_time - start_time
        print(f'it took: {total_time}\n')

        bradley.W_rl_agent.Q_table.to_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
        bradley.B_rl_agent.Q_table.to_pickle(game_settings.imman_agent_q_table_path, compression = 'zip')

        bradley.engine.quit()
    except Exception as e:
        print(f'program interrupted because of:  {e}')
        bradley.engine.quit()

if __name__ == '__main__':
    chess_data_76 = pd.read_pickle(game_settings.chess_games_filepath_part_76, compression = 'zip')
    add_to_q_table_moves_list(chess_data_76)
    chess_data_77 = pd.read_pickle(game_settings.chess_games_filepath_part_77, compression = 'zip')
    add_to_q_table_moves_list(chess_data_77)
    chess_data_78 = pd.read_pickle(game_settings.chess_games_filepath_part_78, compression = 'zip')
    add_to_q_table_moves_list(chess_data_78)
    chess_data_79 = pd.read_pickle(game_settings.chess_games_filepath_part_79, compression = 'zip')
    add_to_q_table_moves_list(chess_data_79)
    chess_data_80 = pd.read_pickle(game_settings.chess_games_filepath_part_80, compression = 'zip')
    add_to_q_table_moves_list(chess_data_80)


    
    
