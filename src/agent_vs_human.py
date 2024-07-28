import helper_methods
import game_settings
import time
import Environ
import Agent

def play_game_vs_human(environ, chess_agent) -> None:
    """
        precondition: environ object is initialized to new game, chess_agent is initialized and trained
    """
    player_turn = 'W'
    while helper_methods.is_game_over(environ) == False:
        try:
            print(f'\nCurrent turn is :  {environ.get_curr_turn()}\n')
            chess_move = handle_move(player_turn, chess_agent)
            print(f'{player_turn} played {chess_move}\n')
        except Exception as e:
            print(f'An error occurred at play_game_vs_human: {e}')
            raise Exception from e

        player_turn = 'B' if player_turn == 'W' else 'W'

    print(f'Game is over, result is: {helper_methods.get_game_outcome(environ)}')
    print(f'The game ended because of: {helper_methods.get_game_termination_reason(environ)}')
    environ.reset_environ()
### end of play_game

def handle_move(player_color: str, chess_agent) -> str:
    if player_color == chess_agent.color:
        print('=== RL AGENT\'S TURN ===\n')
        chess_move = helper_methods.agent_selects_and_plays_chess_move(chess_agent, environ)
    else:
        print('=== OPPONENT\'S TURN ===')
        chess_move = input('Enter chess move: ')
        
        try:
            while not helper_methods.receive_opponent_move(chess_move, environ):
                print('Invalid move, try again.')
                chess_move = input('Enter chess move: ')
            return chess_move
        except Exception as e:
            # put in logger here.
            raise Exception from e
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
        quit()

    end_time = time.time()
    total_time = end_time - start_time
    print('agent vs human game is complete')
    print(f'it took: {total_time}')
    quit()