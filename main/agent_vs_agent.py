from utils import helper_methods
from utils import game_settings
import time
from environment import Environ
from agents import Agent
import logging
from utils import custom_exceptions
from utils.logging_config import setup_logger 

agent_vs_agent_logger = setup_logger(__name__, game_settings.agent_vs_agent_logger_filepath)

def agent_vs_agent(environ, w_agent, b_agent, print_to_screen = False, current_game = 0) -> None:
    try:
        print(f'Playing game {current_game}\n')
        while helper_methods.is_game_over(environ) == False:
            if print_to_screen:
                print(f'\nCurrent turn: {environ.get_curr_turn()}')
                chess_move = helper_methods.agent_selects_and_plays_chess_move(w_agent, environ)
                time.sleep(3)
                print(f'White agent played {chess_move}')
            else:
                agent_vs_agent_logger.info(f'Current turn is: {environ.get_current_turn()}. \nWhite agent played {chess_move}\n')
            
            # sometimes game ends after white's turn
            if helper_methods.is_game_over(environ) == False:
                if print_to_screen:
                    chess_move = helper_methods.agent_selects_and_plays_chess_move(b_agent, environ)
                    time.sleep(3)
                    print(f'Black agent played {chess_move} curr board is:\n{environ.board}\n')
                else:
                    agent_vs_agent_logger.info(f'Black agent played {chess_move} curr board is:\n{environ.board}\n')
    except Exception as e:
        error_message = f'An error occurred at agent_vs_agent: {e}'
        agent_vs_agent_logger.error(error_message)
        raise custom_exceptions.GamePlayError(error_message) from e

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
        print_to_screen = (input('Do you want to print the games to the screen? (y/n) ')).lower()[0]

        for current_game in range(int(number_of_games)):
            if print_to_screen == 'y':
                print(f'Game {current_game + 1}')

            agent_vs_agent(environ, bradley, imman, print_to_screen, current_game)

    except Exception as e:
        print(f'agent vs agent interrupted because of:  {e}')
        agent_vs_agent_logger.error(f'An error occurred: {e}\n')        
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print('single agent vs agent game is complete')
    print(f'it took: {total_time}')
    quit()

