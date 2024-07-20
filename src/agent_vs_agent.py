import helper_methods
import game_settings
import Bradley
import time

if __name__ == '__main__':
    start_time = time.time()
    bradley = Bradley.Bradley()

    try:
        helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
        helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
        agent_vs_agent(bradley)
        bradley.engine.quit()
    except Exception as e:
        print(f'agent vs agent interrupted because of:  {e}')
        bradley.engine.quit()
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print('single agent vs agent game is complete')
    print(f'it took: {total_time}')
    quit()

