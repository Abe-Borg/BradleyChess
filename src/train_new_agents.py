import helper_methods
import game_settings
import pandas as pd
import time
import Bradley

# !!! MAKE SURE to set desired chess_data path in game settings before executing this script !!! #


if __name__ == '__main__':
    start_time = time.time()
    bradley = Bradley.Bradley()
    
    try:
        bradley.train_rl_agents()
        bradley.engine.quit()
    except Exception as e:
        print(f'training interrupted because of:  {e}')
        bradley.engine.quit()
        quit()
        
    end_time = time.time()
    helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time} for {game_settings.training_sample_size} games\n')
    quit()