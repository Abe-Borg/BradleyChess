import helper_methods
import game_settings
import time
import training_functions
import Agent
import Environ
from utils.logging_config import setup_logger

agent_vs_agent_logger = setup_logger(__name__, game_settings.agent_vs_agent_logger_filepath)

if __name__ == '__main__':
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')
    
    helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)
    num_games_to_play = game_settings.agent_vs_agent_num_games

    try:
        training_functions.continue_training_rl_agents(num_games_to_play, bradley, imman, environ)
        helper_methods.pikl_q_table(bradley, game_settings.bradley_agent_q_table_path)
        helper_methods.pikl_q_table(imman, game_settings.imman_agent_q_table_path)

    except Exception as e:
        print(f'training interrupted because of:  {e}')
        quit()
        
    end_time = time.time()
    total_time = end_time - start_time

    print('agent v agent training round is complete')
    print(f'it took: {total_time}')
    quit()