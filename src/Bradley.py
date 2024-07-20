import Environ
import game_settings
import custom_exceptions
import logging

class Bradley:
    """
        A class representing the main game controller for a chess AI system.

        The Bradley class manages the game environment, handles opponent moves,
        and coordinates the AI agent's move selection. It interfaces with the
        Environ class to maintain the game state and uses logging to track errors.

        Attributes:
            error_logger (logging.Logger): Logger for error tracking and reporting.
            environ (Environ): An instance of the Environ class representing the game environment.

        Methods:
            __init__(): Initializes the Bradley instance with error logging and game environment.
            receive_opp_move(chess_move: str) -> bool: Processes and applies the opponent's move.
            rl_agent_selects_chess_move(chess_agent) -> str: Selects and applies the AI agent's move.

        The class is designed to handle the flow of a chess game, including move validation,
        state updates, and error handling. It serves as the central controller for the
        chess AI system, coordinating between the game environment and the AI agent.
    """
    def __init__(self):
        self.error_logger = logging.getLogger(__name__)
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(game_settings.bradley_errors_filepath)
        self.error_logger.addHandler(error_handler)

        self.environ = Environ.Environ()       
    ### end of Bradley constructor ###
