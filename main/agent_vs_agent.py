from utils import helper_methods, game_settings, custom_exceptions
import time
from environment import Environ
from agents import Agent
from utils.logging_config import setup_logger 
agent_vs_agent_logger = setup_logger(__name__, game_settings.agent_vs_agent_logger_filepath)

def agent_vs_agent(environ, w_agent, b_agent, print_to_screen = False, current_game: int = 0) -> None:
    agent_vs_agent_logger.info(f'Playing game {current_game}\n')
    try:    
        while helper_methods.is_game_over(environ) == False:
            chess_move = helper_methods.agent_selects_and_plays_chess_move(w_agent, environ)
            agent_vs_agent_logger.info(f'\nCurrent turn: {environ.get_curr_turn()}')
            agent_vs_agent_logger.info(f'White agent played {chess_move}')
            agent_vs_agent_logger.info(f'Current turn is: {environ.get_current_turn()}. \nWhite agent played {chess_move}\n')
     
            if helper_methods.is_game_over(environ) == False:
                chess_move = helper_methods.agent_selects_and_plays_chess_move(b_agent, environ)
                agent_vs_agent_logger.info(f'Black agent played {chess_move} curr board is:\n{environ.board}\n')
    except custom_exceptions.GamePlayError as e:
        agent_vs_agent_logger.error(f'An error occurred at agent_vs_agent: {e}')
        raise

    agent_vs_agent_logger.info('Game is over\n')
    agent_vs_agent_logger.info(f'Final board is:\n{environ.board}\n')
    agent_vs_agent_logger.info(f'game result is: {environ.get_game_result()}\n')
    environ.reset_environ()
### end of agent_vs_agent

if __name__ == '__main__':
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    try:
        bradley = helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
        imman = helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)
        number_of_games = int(input('How many games do you want the agents to play? '))
        
        for current_game in range(int(number_of_games)):
            agent_vs_agent(environ, bradley, imman, print_to_screen, current_game)
    except Exception as e:
        print(f'agent vs agent interrupted because of:  {e}')
        agent_vs_agent_logger.error(f'An error occurred: {e}\n')        
        exit(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    print('single agent vs agent game is complete')
    print(f'it took: {total_time}')

