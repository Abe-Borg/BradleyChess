import Bradley
import pandas as pd
import game_settings
import random
# import logging
# import log_config
# logger = logging.getLogger(__name__)

def play_game(bubs: Bradley.Bradley, rl_agent_color: str) -> None:
    def handle_move(player_color):
        if player_color == rl_agent.color:
            print('=== RL AGENT\'S TURN ===\n')
            return bubs.rl_agent_selects_chess_move(player_color)
        else:
            print('=== OPPONENT\'S TURN ===')
            move = input('Enter chess move: ')
            while not bubs.receive_opp_move(move):
                print('Invalid move, try again.')
                move = input('Enter chess move: ')
            return move

    def log_error(e):
        errors_file.write(f'An error occurred: {e}\n')

    rl_agent = bubs.W_rl_agent if rl_agent_color == 'W' else bubs.B_rl_agent
    player_turn = 'W'

    with open(game_settings.helper_methods_errors_filepath, 'a') as errors_file:
        while bubs.is_game_over() == False:
            try:
                print(f'\nCurrent turn is :  {bubs.environ.get_curr_turn()}\n')
                chess_move = handle_move(player_turn)
                print(f'{player_turn} played {chess_move}\n')
            except Exception as e:
                log_error(e)

            player_turn = 'B' if player_turn == 'W' else 'W'

        print(f'Game is over, result is: {bubs.get_game_outcome()}')
        print(f'The game ended because of: {bubs.get_game_termination_reason()}')
        bubs.reset_environ()
### end of play_game

def agent_vs_agent(bubs: Bradley.Bradley) -> None:
    def play_turn(agent_color: str):
        try:
            chess_move = bubs.rl_agent_selects_chess_move(agent_color)
            agent_vs_agent_file.write(f'{agent_color} agent played {chess_move}\n')
        except Exception as e:
            agent_vs_agent_file.write(f'An error occurred: {e}\n')
            raise Exception from e

    with open(game_settings.agent_vs_agent_filepath, 'a') as agent_vs_agent_file:
        try:
            while bubs.is_game_over() == False:
                agent_vs_agent_file.write(f'\nCurrent turn: {bubs.environ.get_curr_turn()}')
                play_turn('W')
                
                if bubs.is_game_over() == False:
                    play_turn('B')

            agent_vs_agent_file.write('Game is over, chessboard looks like this:\n')
            agent_vs_agent_file.write(bubs.environ.board + '\n\n')
            agent_vs_agent_file.write(f'Game result is: {bubs.get_game_outcome()}\n')
            agent_vs_agent_file.write(f'Game ended because of: {bubs.get_game_termination_reason()}\n')
        except Exception as e:
            agent_vs_agent_file.write(f'An unhandled error occurred: {e}\n')

        bubs.reset_environ()
### end of agent_vs_agent

def pikl_q_table(bubs: Bradley.Bradley, rl_agent_color: str, q_table_path: str) -> None:
    """Save the Q-table of a trained RL agent to a file.
    Args:
        bubs (Bradley.Bradley): An object of the `Bradley` class representing the chess game environment.
        rl_agent_color (str): A string representing the color of the RL agent ('W' for white or 'B' for black).
        q_table_path (str): A string representing the path to the output file.
    Returns:
        None
    """
    if rl_agent_color == 'W':
        rl_agent = bubs.W_rl_agent
    else:
        rl_agent = bubs.B_rl_agent

    rl_agent.Q_table.to_pickle(q_table_path, compression = 'zip')
### end of pikl_Q_table

def bootstrap_agent(bubs: Bradley.Bradley, rl_agent_color: str, existing_q_table_path: str) -> None:
    """Assign an agent's Q-table to an existing Q-table.
    Args:
        bubs (imman.Bradley): An object of the `Bradley` class representing the chess game environment.
        rl_agent_color (str): A string representing the color of the RL agent ('W' for white or 'B' for black).
        existing_q_table_path (str): A string representing the path to the existing Q-table file.
    Returns:
        None
    """
    if rl_agent_color == 'W':
        rl_agent = bubs.W_rl_agent
    else:
        rl_agent = bubs.B_rl_agent

    rl_agent.Q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    rl_agent.is_trained = True
### end of bootstrap_agent

def get_number_with_probability(probability: float) -> int:
    """Generate a random number with a given probability.
    Args:
        probability (float): A float representing the probability of generating a 1.
    Returns:
        int: A random integer value of either 0 or 1.
    """
    if game_settings.PRINT_DEBUG:
        debug_file = open(game_settings.helper_methods_debug_filepath, 'a')
        debug_file.write('========== Hello from Helper Methods get_number_with_probability ==========\n')
        debug_file.write(f'probability: {probability}\n')

    if random.random() < probability:
        if game_settings.PRINT_DEBUG:
            debug_file.write("Random number is less than probability, returning 1\n")
            debug_file.write("========== Bye from Helper Methods get_number_with_probability ==========\n\n\n")
        debug_file.close()
        return 1
    else:
        if game_settings.PRINT_DEBUG:
            debug_file.write("Random number is >= than probability, returning 0\n")
            debug_file.write("========== Bye from Helper Methods get_number_with_probability ==========\n\n\n")
        debug_file.close()
        return 0
### end of get_number_with_probability