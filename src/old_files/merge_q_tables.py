import pandas as pd
import numpy as np
import helper_methods
import game_settings
import Bradley
import time

def merge_q_tables():
    start_time = time.time()
    
    # Initialize the Bradley object
    bradley = Bradley.Bradley()
    
    # Bootstrap the agent's Q tables
    helper_methods.bootstrap_agent_fill_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent_fill_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)

    print(f'White Q table size before games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size before games: {bradley.B_rl_agent.Q_table.shape}\n')

    # Extract the indices (chess moves) as series
    white_move_list = bradley.W_rl_agent.Q_table.index.to_series()
    black_move_list = bradley.B_rl_agent.Q_table.index.to_series()

    print(f'White move list size: {white_move_list.shape}')
    print(f'Black move list size: {black_move_list.shape}\n')

    print(f'First 5 entries in white move list: {white_move_list.head()}')
    print(f'Last 5 entries in white move list: {white_move_list.tail()}\n')

    # Merge the two series into one and drop duplicates
    merged_indices = pd.concat([white_move_list, black_move_list]).drop_duplicates().reset_index(drop=True)

    print(f'\nQ tables merged into one pandas series\n')
    print(f'Datatype of merged_indices: {type(merged_indices)}\n')
    print(f'Size of merged_indices: {merged_indices.shape}\n')
    print(f'First 5 entries in merged_indices: {merged_indices.head()}')
    print(f'Last 5 entries in merged_indices: {merged_indices.tail()}\n')

    end_time = time.time()
    total_time = end_time - start_time

    print(f'It took: {total_time} seconds\n')
    
    bradley.engine.quit()
    return merged_indices

if __name__ == '__main__':
    try:
        merge_move_list = merge_q_tables()
        merge_move_list.to_pickle(game_settings.unique_chess_moves_list_path, compression = 'zip')
    except Exception as e:
        print(f'Merge_q_tables failed, interrupted because of: {e}')
        quit()
