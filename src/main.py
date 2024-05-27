import helper_methods
import game_settings
import pandas as pd
import time
import Bradley

# import logging
# import log_config
# logger = logging.getLogger(__name__)
 
if __name__ == '__main__':

    # ========== GENERATE Q ESTIMATES ==========
    # chess_data_file_path = game_settings.chess_games_filepath_part_50
    # est_q_vals_file_path = game_settings.est_q_vals_filepath_part_50

    # chess_data = pd.read_pickle(chess_data_file_path, compression = 'zip')
    # chess_data = chess_data.head(game_settings.training_sample_size)
    # bradley = Bradley.Bradley(chess_data)
    
    # start_time = time.time()

    # try:
    #     bradley.generate_q_est_vals(est_q_vals_file_path) # this method closes the game engine
    # except Exception as e:
    #     print(f'generate q est interrupted because of:  {e}')
    #     quit()
    
    # end_time = time.time()
    # total_time = end_time - start_time
    # print('generate q est is complete')
    # print(f'it took: {total_time} seconds\n')
    

    # ========== IDENTIFY AND REMOVE CORRUPTED GAMES FROM CHESS DATABASE ==========
    chess_data_file_path = game_settings.chess_games_filepath_part_90

    chess_data = pd.read_pickle(chess_data_file_path, compression = 'zip')
    chess_data.drop(chess_data[chess_data['PlyCount'] < 24].index, inplace = True)
    
    print(f'Total number of rows before cleanup: {chess_data.shape[0]}')

    bradley = Bradley.Bradley(chess_data)
    start_time = time.time()

    try:
        bradley.identify_corrupted_games()
        bradley.engine.quit()    
        chess_data.drop(bradley.corrupted_games_list, inplace = True)
        print(f'Total number of rows after cleanup: {chess_data.shape[0]}')

    except Exception as e:
        print(f'corrupted games identification interrupted because of:  {e}')
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time
    print('corrupted games identification is complete')
    print(f'it took: {total_time} seconds\n')
    print(f'number of corrupted games: {len(bradley.corrupted_games_list)}')
    print(f'corrupted games: {bradley.corrupted_games_list}\n')

    chess_data.to_pickle(chess_data_file_path, compression = 'zip')


    # # ========================= train new agents ========================= # 
    # # read chess data from pkl file
    # chess_data_file_path = game_settings.chess_pd_dataframe_file_path_part_1
    # chess_data = pd.read_pickle(chess_data_file_path, compression = 'zip')
    
    # bradley = Bradley.Bradley(chess_data)
    

    # start_time = time.time()
    # try:
    #     bradley.train_rl_agents()
    # except Exception as e:
    #     print(f'training interrupted because of:  {e}')
    #     quit()
        
    # end_time = time.time()
    # helper_methods.pikl_q_table(bradley, 'W',game_settings.bradley_agent_q_table_path)
    # helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    
    # total_time = end_time - start_time
    # print('training is complete')
    # print(f'it took: {total_time} for {game_settings.training_sample_size} games\n')
    # quit()

    # # # # # ========================= bootstrap and continue training agents ========================= #
    # bradley = helper_methods.init_bradley(training_chess_data)    # the size of the training set in this step doesnt matter. It's just for initializing the object.
    # helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)

    # start_time = time.time()
    # try:
    #     bradley.continue_training_rl_agents(game_settings.agent_vs_agent_num_games)
    # except Exception as e:
    #     print(f'training interrupted because of:  {e}')
    #     quit()
        
    # helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    # end_time = time.time()
    # total_time = end_time - start_time
    # print('training is complete')
    # print(f'it took: {total_time}')
    # quit()


    # # # ========================= bootstrap and play against human =========================  #
    # bradley = helper_methods.init_bradley(training_chess_data)
    # helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
    
    # rl_agent_color = input('Enter color for agent to be , \'W\' or \'B\': ')
    
    # if rl_agent_color == 'W':
    #     play_game(bradley, rl_agent_color)
    # else: 
    #     play_game(bradley, 'B')
    

    # # # ========================= bootstrap agents and have them play each other =========================  #
    # bradley = helper_methods.init_bradley(training_chess_data)
    # helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    # helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
    # helper_methods.agent_vs_agent(bradley)