from utils import helper_methods, game_settings, custom_exceptions
import time
from environment import Environ
from agents import Agent

def play_game_vs_human(environ, agent) -> None:
    player_turn = 'W'
    try:
        while not helper_methods.is_game_over(environ):
            print(f'\nCurrent turn is :  {environ.get_curr_turn()}\n')
            chess_move = handle_move(player_turn, agent, environ)
            print(f'{player_turn} played {chess_move}\n')
            player_turn = 'B' if player_turn == 'W' else 'W'

        print(f'Game is over, result is: {helper_methods.get_game_outcome(environ)}')
        print(f'The game ended because of: {helper_methods.get_game_termination_reason(environ)}')
    except custom_exceptions.GamePlayError as e:
        print(f'An error occurred at play_game_vs_human: {e}')
        raise 
    finally:
        environ.reset_environ()
### end of play_game

def handle_move(player_color: str, agent, environ) -> str:
    if player_color == agent.color:
        print('=== RL AGENT\'S TURN ===\n')
        chess_move = helper_methods.agent_selects_and_plays_chess_move(agent, environ)
    else:
        print('=== OPPONENT\'S TURN ===')
        while True:
            chess_move = input('Enter chess move: or type \'exit\' to quit: ')
            try:
                if helper_methods.receive_opponent_move(chess_move, environ):
                    return chess_move
                if chess_move == 'exit':
                    print('Exiting game...')
                    quit()
            except Exception as e:
                print('Failed to load move. Try again.')
### end of handle_move

if __name__ == '__main__':    
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    try:
        bradley = helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
        imman = helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)
        
        rl_agent_color = input('Enter color for agent to be , \'W\' or \'B\': ')
    
        if rl_agent_color == 'W':
            play_game_vs_human(environ, bradley)
        else: 
            play_game_vs_human(environ, imman)
        
    except Exception as e:
        print(f'agent vs human interrupted because of:  {e}')
        exit(1)

    end_time = time.time()
    total_time = end_time - start_time
    print('agent vs human game is complete')
    print(f'it took: {total_time}')
