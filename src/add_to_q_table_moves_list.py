import pandas as pd
import numpy as np
import helper_methods
import game_settings
import Bradley
import time


if __name__ == '__main__':

    start_time = time.time()
    bradley = Bradley.Bradley()

    helper_methods.bootstrap_agent_fill_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent_fill_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)


    print(f'White Q table size before games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size before games: {bradley.B_rl_agent.Q_table.shape}\n')

    try:
        all_white_moves = set()
        all_black_moves = set()

        white_columns = [f'W{i}' for i in range(1, game_settings.max_num_turns_per_player + 1)]  # W1 to W200
        black_columns = [f'B{i}' for i in range(1, game_settings.max_num_turns_per_player + 1)]  # B1 to B200

        for game_num_str, game in game_settings.chess_data.iterrows():
            # Extract White moves
            white_moves = game[white_columns].dropna().unique()
            # Extract Black moves
            black_moves = game[black_columns].dropna().unique()
        
            all_white_moves.update(white_moves)
            all_black_moves.update(black_moves)

        # Update Q-tables with new moves
        new_white_moves = [move for move in all_white_moves if move not in bradley.W_rl_agent.Q_table.index]
        new_black_moves = [move for move in all_black_moves if move not in bradley.B_rl_agent.Q_table.index]

        bradley.W_rl_agent.update_Q_table(new_white_moves)
        bradley.B_rl_agent.update_Q_table(new_black_moves)
        
        print(f'datatype of white moves list {type(white_moves)}')  # Should output: <class 'numpy.ndarray'>
        print(f'white moves list cell data types {white_moves.dtype}\n')  # Should output: dtype('<U2') or similar, indicating strings
        print(f'First few elements of white_moves: {white_moves[:5]}')

        print(f'datatype of black moves list {type(black_moves)}')  # Should output: <class 'numpy.ndarray'>
        print(f'white moves list cell data types {black_moves.dtype}\n')  # Should output: dtype('<U2') or similar, indicating strings
        print(f"First few elements of black_moves: {black_moves[:5]}")

        print(f'all white moves: {all_white_moves}')
        print(f'all black moves: {all_black_moves}\n')

        # print datatype for all_white_moves and all_black_moves
        print(f'datatype of all white moves list {type(all_white_moves)}')  # Should output: <class 'set'>
        print(f'datatype of all black moves list {type(all_black_moves)}\n')  # Should output: <class 'set'>

        print(f'New white moves: {new_white_moves}')
        print(f'New black moves: {new_black_moves}\n')

        # print datatype of new_white_moves and new_black_moves
        print(f'datatype of new white moves list {type(new_white_moves)}')  # Should output: <class 'list'>
        print(f'datatype of new black moves list {type(new_black_moves)}\n')  # Should output: <class 'list'>

        print(f'White Q table size after games: {bradley.W_rl_agent.Q_table.shape}')
        print(f'Black Q table size after games: {bradley.B_rl_agent.Q_table.shape}\n')

        print(f'White Q table head after games: {bradley.W_rl_agent.Q_table.head()}')
        print(f'Black Q table head after games: {bradley.B_rl_agent.Q_table.head()}')

        end_time = time.time()
        total_time = end_time - start_time
        print(f'it took: {total_time}\n')

        bradley.engine.quit()
    except Exception as e:
        print(f'program interrupted because of:  {e}')
        bradley.engine.quit()    

    
    
