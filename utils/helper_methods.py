import pandas as pd
from utils import game_settings
import random
from agents import Agent
from utils import custom_exceptions
from utils.logging_config import setup_logger

helper_methods_logger = setup_logger(__name__, game_settings.helper_methods_errors_filepath)

def agent_selects_and_plays_chess_move(chess_agent, environ) -> str:
    """
        The Agent selects a chess move and loads it onto the chessboard.
        This method allows the agent to select a chess move and load it onto the 
        chessboard. It is used during actual gameplay between the computer and the user, not during training. The 
        method first gets the current state of the environment. If the list of legal moves is empty, an exception 
        is raised. Depending on the color of the RL agent, the appropriate agent selects a move. The selected move 
        is then loaded onto the chessboard and the current state of the environment is updated.

        Args:
        Returns:
            str: A string representing the selected chess move.
        Raises:
            StateUpdateError: If the current state is not valid or fails to update.
            NoLegalMovesError: If the list of legal moves is empty.
            ChessboardLoadError: If the chessboard fails to load the move.
            StateRetrievalError: If the current state is not valid or fails to retrieve.
        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
    """
    curr_state = environ.get_curr_state() 
    chess_move: str = chess_agent.choose_action(curr_state)
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return chess_move
### end of agent_selects_and_plays_chess_move

def receive_opponent_move(chess_move: str, environ) -> bool:                                                                                 
    """
        Receives the opponent's chess move and updates the environment.
        This method receives the opponent's chess move, loads it onto the chessboard, and updates the current state 
        of the environment. If an error occurs while loading the chessboard or updating the current state, an error 
        message is written to the errors file and an exception is raised.

        Args:
            chess_move (str): A string representing the opponent's chess move, such as 'Nf3'.
        Returns:
            bool: A boolean value indicating whether the move was successfully loaded and the current state was 
            successfully updated. Returns False if an error occurred while loading the chessboard, and does not 
            attempt to update the current state.
        Raises:
            Exception: An exception is raised if the chessboard fails to load the move or if the current state fails 
            to update. The original exception is included in the raised exception.
        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
    """
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return True

### end of receive_opp_move

def pikl_q_table(chess_agent, q_table_path: str) -> None:
    chess_agent.q_table.to_pickle(q_table_path, compression = 'zip')
### end of pikl_q_table

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
### end of reset_q_table ###

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
    try:
        return (
            environ.board.is_game_over() or
            environ.turn_index >= game_settings.max_turn_index or
            (len(environ.get_legal_moves()) == 0)
        )
    except Exception as e:
        error_message = f'error at is_game_over: {str(e)}, failed to determine if game is over\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameOverError(error_message) from e
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
    except Exception as e:
        error_message = f'error at get_game_outcome: {str(e)}, failed to get game outcome\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameOutcomeError(error_message) from e
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
    except Exception as e:
        error_message = f'error at get_game_termination_reason: {str(e)}, failed to get game end reason\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameTerminationError(error_message) from e
### end of get_game_termination_reason