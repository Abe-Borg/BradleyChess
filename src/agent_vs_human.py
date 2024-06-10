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
        
        rl_agent_color = input('Enter color for agent to be , \'W\' or \'B\': ')
    
        if rl_agent_color == 'W':
            helper_methods.play_game(bradley, rl_agent_color)
        else: 
            helper_methods.play_game(bradley, 'B')
        
        bradley.engine.quit()
    except Exception as e:
        print(f'agent vs human interrupted because of:  {e}')
        bradley.engine.quit()
        quit()

    end_time = time.time()
    total_time = end_time - start_time
    print('agent vs human game is complete')
    print(f'it took: {total_time}')
    quit()