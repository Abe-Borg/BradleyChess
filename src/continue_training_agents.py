import helper_methods
import game_settings
import time
import Bradley

if __name__ == '__main__':
    start_time = time.time()
    bradley = Bradley.Bradley()
    
    try:
        helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
        helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
        bradley.continue_training_rl_agents(game_settings.agent_vs_agent_num_games)

        helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
        helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
        bradley.engine.quit()
    except Exception as e:
        print(f'training interrupted because of:  {e}')
        bradley.engine.quit()
        quit()
        
    end_time = time.time()
    total_time = end_time - start_time

    print('agent v agent training round is complete')
    print(f'it took: {total_time}')
    quit()