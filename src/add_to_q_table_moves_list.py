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
    chess_data_80 = pd.read_pickle(game_settings.chess_games_filepath_part_80, compression = 'zip')
    add_to_q_table_moves_list(chess_data_80)
    chess_data_80 = None
    chess_data_81 = pd.read_pickle(game_settings.chess_games_filepath_part_81, compression = 'zip')
    add_to_q_table_moves_list(chess_data_81)
    chess_data_81 = None
    chess_data_82 = pd.read_pickle(game_settings.chess_games_filepath_part_82, compression = 'zip')
    add_to_q_table_moves_list(chess_data_82)
    chess_data_82 = None
    chess_data_83 = pd.read_pickle(game_settings.chess_games_filepath_part_83, compression = 'zip')
    add_to_q_table_moves_list(chess_data_83)
    chess_data_83 = None
    chess_data_84 = pd.read_pickle(game_settings.chess_games_filepath_part_84, compression = 'zip')
    add_to_q_table_moves_list(chess_data_84)
    chess_data_84 = None
    chess_data_85 = pd.read_pickle(game_settings.chess_games_filepath_part_85, compression = 'zip')
    add_to_q_table_moves_list(chess_data_85)
    chess_data_85 = None
    chess_data_86 = pd.read_pickle(game_settings.chess_games_filepath_part_86, compression = 'zip')
    add_to_q_table_moves_list(chess_data_86)
    chess_data_86 = None
    chess_data_87 = pd.read_pickle(game_settings.chess_games_filepath_part_87, compression = 'zip')
    add_to_q_table_moves_list(chess_data_87)
    chess_data_87 = None
    chess_data_88 = pd.read_pickle(game_settings.chess_games_filepath_part_88, compression = 'zip')
    add_to_q_table_moves_list(chess_data_88)
    chess_data_88 = None
    chess_data_89 = pd.read_pickle(game_settings.chess_games_filepath_part_89, compression = 'zip')
    add_to_q_table_moves_list(chess_data_89)
    chess_data_89 = None
    chess_data_90 = pd.read_pickle(game_settings.chess_games_filepath_part_90, compression = 'zip')
    add_to_q_table_moves_list(chess_data_90)
    chess_data_90 = None
    chess_data_91 = pd.read_pickle(game_settings.chess_games_filepath_part_91, compression = 'zip')
    add_to_q_table_moves_list(chess_data_91)
    chess_data_91 = None
    chess_data_92 = pd.read_pickle(game_settings.chess_games_filepath_part_92, compression = 'zip')
    add_to_q_table_moves_list(chess_data_92)
    chess_data_92 = None
    chess_data_93 = pd.read_pickle(game_settings.chess_games_filepath_part_93, compression = 'zip')
    add_to_q_table_moves_list(chess_data_93)
    chess_data_93 = None
    chess_data_94 = pd.read_pickle(game_settings.chess_games_filepath_part_94, compression = 'zip')
    add_to_q_table_moves_list(chess_data_94)
    chess_data_94 = None
    chess_data_95 = pd.read_pickle(game_settings.chess_games_filepath_part_95, compression = 'zip')
    add_to_q_table_moves_list(chess_data_95)
    chess_data_95 = None
    chess_data_96 = pd.read_pickle(game_settings.chess_games_filepath_part_96, compression = 'zip')
    add_to_q_table_moves_list(chess_data_96)
    chess_data_96 = None
    chess_data_97 = pd.read_pickle(game_settings.chess_games_filepath_part_97, compression = 'zip')
    add_to_q_table_moves_list(chess_data_97)
    chess_data_97 = None
    chess_data_98 = pd.read_pickle(game_settings.chess_games_filepath_part_98, compression = 'zip')
    add_to_q_table_moves_list(chess_data_98)
    chess_data_98 = None
    chess_data_99 = pd.read_pickle(game_settings.chess_games_filepath_part_99, compression = 'zip')
    add_to_q_table_moves_list(chess_data_99)
    chess_data_99 = None
    chess_data_100 = pd.read_pickle(game_settings.chess_games_filepath_part_100, compression = 'zip')
    add_to_q_table_moves_list(chess_data_100)
    chess_data_100 = None


    
    
