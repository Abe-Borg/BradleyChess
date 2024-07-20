import helper_methods
import game_settings
import Bradley
import time
import Environ
import Agent

def agent_vs_agent(environ, w_agent, b_agent, print_to_screen = False, current_game = 0) -> None:
    try:
        # play all moves in a single game
        while helper_methods.is_game_over(environ) == False:
            if print_to_screen:
                print(f'\nCurrent turn: {environ.get_curr_turn()}')
                chess_move = helper_methods.agent_selects_and_plays_chess_move(chess_agent, environ)
                time.sleep(3)
                print(f'White agent played {chess_move} curr board is:\n{environ.board}\n')
            else:
                pass
                # make a log file that will record the game moves. and track the current game
            
            # sometimes game ends after white's turn
            if helper_methods.is_game_over(environ) == False:
                if print_to_screen:
                    print(f'\nCurrent turn: {environ.get_curr_turn()}')
                    chess_move = helper_methods.agent_selects_and_plays_chess_move(chess_agent, environ)
                    time.sleep(3)
                    print(f'Black agent played {chess_move} curr board is:\n{environ.board}\n')
                else:
                    pass
                    # make a log file that will record the game moves.
    except Exception as e:
        # agent_vs_agent_file.write(f'An unhandled error occurred: {e}\n')
        raise Exception from e

    # game is over, reset environ
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

        # ask user to input number of games to play
        number_of_games = int(input('How many games do you want the agents to play? '))
        print_to_screen = (input('Do you want to print the games to the screen? (y/n) ')).lower()[0]

        # while there are games still to play, call agent_vs_agent
        for current_game in range(int(number_of_games)):
            if print_to_screen == 'y':
                print(f'Game {current_game + 1}')

            agent_vs_agent(environ, bradley, imman, print_to_screen, current_game)

    except Exception as e:
        print(f'agent vs agent interrupted because of:  {e}')
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print('single agent vs agent game is complete')
    print(f'it took: {total_time}')
    quit()

