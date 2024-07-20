import helper_methods
import game_settings
import Bradley
import time
import Environ
import Agent

if __name__ == '__main__':
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    try:
        helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
        helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)

        # ask user to input number of games to play
        number_of_games = int(input('How many games do you want the agents to play? '))
        print_to_screen = (input('Do you want to print the games to the screen? (y/n) ')).lower()[0]

        # while there are games still to play, call agent_vs_agent
        for current_game in range(int(number_of_games)):
            if print_to_screen == 'y':
                print(f'Game {current_game + 1}')

            helper_methods.agent_vs_agent(environ, bradley, imman, print_to_screen, current_game)

    except Exception as e:
        print(f'agent vs agent interrupted because of:  {e}')
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print('single agent vs agent game is complete')
    print(f'it took: {total_time}')
    quit()

