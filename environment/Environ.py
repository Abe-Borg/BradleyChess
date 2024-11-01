import chess
from utils import custom_exceptions, game_settings, constants
from utils.logging_config import setup_logger 
from typing import Union, Dict, List
environ_logger = setup_logger(__name__, game_settings.environ_errors_filepath)

class Environ:
    """
    Manages the chess game environment, including the chessboard state, turn tracking, and legal moves.
    """
    def __init__(self):
        """
            Initializes an Environ object with a chessboard
            Attributes:
                - board (chess.Board): An object representing the chessboard.
                - turn_list (list[str]): A list of strings representing the turns in a game. Each string is in the format 
                  'Wn' or 'Bn', where 'W' and 'B' represent white and black players respectively, and 'n' is the turn 
                  number.
                - turn_index (int): An integer representing the current turn index.
                - errors_file (file): A file object representing the errors file. The file is opened in append mode.
        """
        try: 
            self.board: chess.Board = chess.Board()            
            max_turns = constants.max_num_turns_per_player * constants.num_players
            self.turn_list: List[str] = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' for i in range(max_turns)]
            self.turn_index: int = 0
        except Exception as e:
            environ_logger.error(f'at __init__: failed to initialize environ. Error: {e}\n', exc_info=True)
            raise custom_exceptions.EnvironInitializationError(f'failed to initialize environ due to error: {e}') from e
    ### end of constructor

    def get_curr_state(self) -> Dict[str, Union[int, str, List[str]]]:
        """
            constructs a dictionary that represents the current state of the chessboard. 
            Returns:
                dict[str, str, list[str]]: A dictionary representing the current state of the chessboard. The dictionary 
                has the following keys:
                    - 'turn_index': The current turn index.
                    - 'curr_turn': The current turn, represented as a string.
                    - 'legal_moves': A list of strings, where each string is a legal move at the current turn.
            Raises:
                IndexError: Raised when the turn index is out of range of the turn list. The error message is written 
                into the errors file before the exception is re-raised. 
        """
        if not (0 <= self.turn_index < len(self.turn_list)):
            message = f'Turn index out of range: {self.turn_index}'
            environ_logger.error(message)
            raise custom_exceptions.TurnIndexError(message)
    
        curr_turn = self.get_curr_turn()
        legal_moves = self.get_legal_moves()     
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    ### end of get_curr_state
    
    def update_curr_state(self) -> None:
        """
            Advances the turn index to update the current state of the chessboard.
            Raises:
                IndexError: Raised when the turn index reaches or exceeds the maximum turn index defined in the game 
                settings. The error message and the current turn index are written into the errors file before the 
                exception is re-raised.
            Side Effects:
                Modifies the turn index by incrementing it by one.
            raises IndexError: if the turn index is out of bounds
        """
        if self.turn_index >= constants.max_turn_index:
            message = f'ERROR: max_turn_index reached: {self.turn_index} >= {constants.max_turn_index}\n'
            environ_logger.error(message)
            raise IndexError(message)
    
        if self.turn_index >= len(self.turn_list):
            message = f'ERROR: turn index out of bounds: {self.turn_index} >= {len(self.turn_list)}\n'
            environ_logger.error(message)
            raise IndexError(message)
    
        self.turn_index += 1
    ### end of update_curr_state
    
    def get_curr_turn(self) -> str:                        
        """
            Retrieves the current turn from the turn list based on the turn index.
            Returns:
                str: A string representing the current turn.
            Raises:
                IndexError: Raised when the turn index is out of range of the turn list. The error message and the 
                current turn index are written into the errors file before the exception is re-raised. 
        """
        if not (0 <= self.turn_index < len(self.turn_list)):
            environ_logger.error(f'ERROR: Turn index out of range: {self.turn_index}\n')
            raise custom_exceptions.TurnIndexError(f'Turn index out of range: {self.turn_index}')
        
        return self.turn_list[self.turn_index]
        ### end of get_curr_turn
    
    def load_chessboard(self, chess_move: str, curr_game = 'Game 1') -> None:
        """
            attempts to apply chess move to the current state of the chessboard. 
            Args:
                chess_move (str): A string representing the chess move in SAN, such as 'Nf3'.
                curr_game (str, optional): A string representing the current game. Defaults to 'Game 1'.
            Raises:
                ValueError: Raised when the provided move is invalid or cannot be applied to the current chessboard 
                state. The error message, the invalid move, and the current game are written into the errors file 
                before the exception is re-raised.
            Side Effects:
                Modifies the chessboard's state by applying the provided move.
        """
        try:
            self.board.push_san(chess_move)
        except Exception as e:
            error_message = f'An error occurred at load_chessboard: {str(e)}, unable to load chessboard with {chess_move} in {curr_game}'
            environ_logger.error(error_message)
            raise custom_exceptions.InvalidMoveError(error_message) from e
    ### end of load_chessboard    

    def pop_chessboard(self) -> None:
        """
            Reverts the state of the chessboard by undoing the most recent move.
            Raises:
                IndexError: Raised when there are no moves to undo, i.e., the move stack is empty. The error message 
                is written into the errors file and the exception is re-raised with a custom message indicating the 
                inability to pop the chessboard due to the error.
            Side Effects:
                Modifies the board's state by undoing the last move.
        """
        try:
            self.board.pop()
        except Exception as e:
            error_message = f'An error occurred at pop_chessboard. unable to pop chessboard, due to error: {str(e)}'
            environ_logger.error(error_message)
            raise custom_exceptions.ChessboardPopError(error_message) from e
    ### end of pop_chessboard

    def undo_move(self) -> None:
        """
            Reverts the state of the chessboard by undoing the most recent move.
            Raises:
                IndexError: Raised when there are no moves to undo, i.e., the move stack is empty. The error message 
                and the current turn index are written into the errors file before the exception is re-raised.
            Side Effects:
                Modifies the board's state by undoing the last move and decrementing the turn index.
        """
        try:
            self.board.pop()
            if self.turn_index > 0:
                self.turn_index -= 1
        except Exception as e:
            error_message = f'An error occurred at undo_move, unable to undo move due to error: {str(e)}, at turn index: {self.turn_index}'
            environ_logger.error(error_message)
            raise custom_exceptions.ChessboardPopError(error_message) from e
    ### end of undo_move

    def load_chessboard_for_Q_est(self, analysis_results: list[dict]) -> None:
        """
            Updates the chessboard state using the anticipated next move from the analysis results during training.
            This method is designed to work in conjunction with the Stockfish analysis during training. It extracts the 
            anticipated next move from the analysis results and attempts to apply it to the chessboard.

            Args:
                analysis_results (list[dict]): A list of dictionaries containing the analysis results from Stockfish.
                    Each dictionary is expected to have the following keys: 'mate_score', 'centipawn_score', and 
                    'anticipated_next_move'. The 'anticipated_next_move' is a Move.uci string representing the anticipated 
                    next move.
            Raises:
                ValueError: Raised when the anticipated next move is invalid or cannot be applied to the current chessboard 
                state. The error message and the invalid move are written into the errors file before the exception is 
                re-raised.

            Side Effects:
                Modifies the chessboard's state by applying the anticipated next move.
        """
        # this is the anticipated chess move due to opponent's previous chess move. so if White plays Ne4, 
        # what is Black likely to play according to the engine?
        anticipated_chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        environ_logger.debug(f'anticipated_chess_move: {anticipated_chess_move}. This should have the form like this, Move.from_uci(\'e4f6\')\n')

        try:
            move = chess.Move.from_uci(anticipated_chess_move)
            self.board.push(move)    
        except Exception as e:
            error_message = f'An error occurred at load_chessboard_for_Q_est: {str(e)}, unable to load chessboard with {anticipated_chess_move}'
            environ_logger.error(error_message)
            raise custom_exceptions.ChessboardLoadError(error_message) from e
    ### end of load_chessboard_for_Q_est

    def reset_environ(self) -> None:
        self.board.reset()
        self.turn_index = 0
    ### end of reset_environ
    
    def get_legal_moves(self) -> List[str]:   
        """
            Generates a list of all legal moves in Standard Algebraic Notation (SAN) at the current turn.
            Returns:
                list[str]: A list of strings, where each string is a legal move in SAN. The moves are determined based 
                on the current state of the chessboard and the player whose turn it is.
            Raises:
                NoLegalMovesError: Raised when there are no legal moves available at the current turn. The error message 
                and the current turn index are written into the errors file before the exception is re-raised.
        """
        try:
            return [self.board.san(move) for move in self.board.legal_moves]
        except Exception as e:
            error_message = f'An error occurred at get_legal_moves: {str(e)}, legal moves could not be retrieved, at turn index: {self.turn_index}, current turn: {self.get_curr_turn()}, current board state: {self.board}, current legal moves: {self.board.legal_moves}'
            environ_logger.error(error_message)
            raise custom_exceptions.NoLegalMovesError(error_message) from e
    ### end of get_legal_moves
    