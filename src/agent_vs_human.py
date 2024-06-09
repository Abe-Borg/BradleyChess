import helper_methods
import game_settings
import Bradley

if __name__ == '__main__':        
    bradley = Bradley.Bradley()
    helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
    
    rl_agent_color = input('Enter color for agent to be , \'W\' or \'B\': ')
    
    if rl_agent_color == 'W':
        helper_methods.play_game(bradley, rl_agent_color)
    else: 
        helper_methods.play_game(bradley, 'B')