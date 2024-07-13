import pandas as pd
import game_settings
import random
import chess.engine
import logging

def play_game(bubs, chess_agent) -> None:
    player_turn = 'W'
    while bubs.is_game_over() == False:
        try:
            print(f'\nCurrent turn is :  {bubs.environ.get_curr_turn()}\n')
            chess_move = handle_move(player_turn, chess_agent)
            print(f'{player_turn} played {chess_move}\n')
        except Exception as e:
            # put in logger here.
            raise Exception from e

        player_turn = 'B' if player_turn == 'W' else 'W'

    print(f'Game is over, result is: {bubs.get_game_outcome()}')
    print(f'The game ended because of: {bubs.get_game_termination_reason()}')
    bubs.reset_environ()
### end of play_game

def handle_move(player_color, chess_agent) -> str:
    if player_color == chess_agent.color:
        print('=== RL AGENT\'S TURN ===\n')
        return bubs.rl_agent_selects_chess_move(player_color, chess_agent)
    else:
        print('=== OPPONENT\'S TURN ===')
        move = input('Enter chess move: ')
        while not bubs.receive_opp_move(move):
            print('Invalid move, try again.')
            move = input('Enter chess move: ')
        return move

def agent_vs_agent(bubs, w_agent, b_agent) -> None:
    try:
        while bubs.is_game_over() == False:
            # agent_vs_agent_file.write(f'\nCurrent turn: {bubs.environ.get_curr_turn()}')
            play_turn(w_agent)
            
            # sometimes game ends after white's turn
            if bubs.is_game_over() == False:
                play_turn(b_agent)

        # agent_vs_agent_file.write('Game is over, chessboard looks like this:\n')
        # agent_vs_agent_file.write(bubs.environ.board + '\n\n')
        # agent_vs_agent_file.write(f'Game result is: {bubs.get_game_outcome()}\n')
        # agent_vs_agent_file.write(f'Game ended because of: {bubs.get_game_termination_reason()}\n')
    except Exception as e:
        # agent_vs_agent_file.write(f'An unhandled error occurred: {e}\n')
        raise Exception from e

    bubs.reset_environ()
### end of agent_vs_agent

def play_turn(chess_agent):
    try:
        chess_move = bubs.rl_agent_selects_chess_move(chess_agent)
        # agent_vs_agent_file.write(f'{chess_agent.color} agent played {chess_move}\n')
    except Exception as e:
        # agent_vs_agent_file.write(f'An error occurred: {e}\n')
        raise Exception from e

def pikl_q_table(bubs, chess_agent, q_table_path: str) -> None:
    chess_agent.q_table.to_pickle(q_table_path, compression = 'zip')
### end of pikl_Q_table

def bootstrap_agent(chess_agent, existing_q_table_path: str) -> None:
    chess_agent.q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    chess_agent.is_trained = True
    return chess_agent
### end of bootstrap_agent

def get_number_with_probability(probability: float) -> int:
    """Generate a random number with a given probability.
    Args:
        probability (float): A float representing the probability of generating a 1.
    Returns:
        int: A random integer value of either 0 or 1.
    """
    if random.random() < probability:
        return 1
    else:
        return 0
### end of get_number_with_probability

    def reset_q_table(q_table) -> None:
        q_table.iloc[:, :] = 0
        return q_table    
    ### end of reset_Q_table ###

def start_chess_engine(): 
    chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    return chess_engine