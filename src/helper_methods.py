import pandas as pd
import game_settings
import random
import chess.engine
import logging
import Agent

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

def bootstrap_agent(chess_agent, existing_q_table_path: str) -> Agent.Agent:
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

def is_game_over(environ) -> bool:
    """
        This method determines whether the game is over based on three conditions: if the game is over according to 
        the chessboard, if the current turn index has reached the maximum turn index defined in the game settings, 
        or if there are no legal moves left.

        Arg: environ object, which manages a chessboard

        Returns:
            bool: A boolean value indicating whether the game is over. Returns True if any of the three conditions 
            are met, and False otherwise.
        Side Effects:
            None.
    """
    return (
        environ.board.is_game_over() or
        environ.turn_index >= game_settings.max_turn_index or
        (len(environ.get_legal_moves()) == 0)
    )
### end of is_game_over

def get_game_outcome(environ) -> str:
    """
        Returns the outcome of the chess game.
        This method returns the outcome of the chess game. It calls the `outcome` method on the chessboard, which 
        returns an instance of the `chess.Outcome` class, and then calls the `result` method on this instance to 
        get the outcome of the game. If an error occurs while getting the game outcome, an error message is 
        returned.

        Returns:
            str: A string representing the outcome of the game. The outcome is a string in the format '1-0', '0-1', 
            or '1/2-1/2', representing a win for white, a win for black, or a draw, respectively. If an error 
            occurred while getting the game outcome, the returned string starts with 'error at get_game_outcome: ' 
            and includes the error message.
        Raises:
            GameOutcomeError: If the game outcome cannot be determined.
    """
    try:
        return environ.board.outcome().result()
    except custom_exceptions.GameOutcomeError as e:
        # self.error_logger.error('hello from Bradley.get_game_outcome\n')
        return f'error at get_game_outcome: {e}'
### end of get_game_outcome

def get_game_termination_reason(environ) -> str:
    """
        Returns a string that describes the reason for the game ending.
        This method returns a string that describes the reason for the game ending. It calls the `outcome` method 
        on the chessboard, which returns an instance of the `chess.Outcome` class, and then gets the termination 
        reason from this instance. If an error occurs while getting the termination reason, an error message is 
        returned.

        Returns:
            str: A string representing the reason for the game ending. 
            If an error occurred while getting the termination reason, the returned string starts with 'error at 
            get_game_termination_reason: ' and includes the error message.
        Raises:
            GameTerminationError: If the termination reason cannot be determined.
        Side Effects:
            None.
    """
    try:
        return str(environ.board.outcome().termination)
    except custom_exceptions.GameTerminationError as e:
        # self.error_logger.error('hello from Bradley.get_game_termination_reason\n')
        # self.error_logger.error(f'Error: {e}, failed to get game end reason\n')
        return f'error at get_game_termination_reason: {e}'
    ### end of get_game_termination_reason