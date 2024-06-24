import helper_methods
import game_settings
import Bradley
import time


# !!! MAKE SURE to set desired chess_data path in game settings before executing this script !!! #

 
if __name__ == '__main__':
    start_time = time.time()
    bradley = Bradley.Bradley()

    # initialize agents with q tables
    helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)

    print(f'White Q table size before games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size before games: {bradley.B_rl_agent.Q_table.shape}\n')

    try:
        bradley.simply_play_games()
        bradley.engine.quit()
    except Exception as e:
        print(f'simply play games, interrupted because of:  {e}')
        bradley.engine.quit()    
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print(f'\nq table populated with new moves from current section of chess db')
    print(f'it took: {total_time}\n')

    print(f'White Q table size after games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size after games: {bradley.B_rl_agent.Q_table.shape}')

    bradley.W_rl_agent.Q_table.to_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
    bradley.B_rl_agent.Q_table.to_pickle(game_settings.imman_agent_q_table_path, compression = 'zip')

    quit()