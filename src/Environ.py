import game_settings
import chess
import logging

class Environ:
    """
    A class representing the environment of a chess game.
    This class provides methods for loading chess moves onto a chessboard, undoing moves, retrieving the current 
    state of the chessboard, and more. It maintains a turn list and a turn index to track the current turn, and 
    logs errors into an errors file.
    """
    def __init__(self):
        """
            Initializes an Environ object with a chessboard, a turn list, a turn index, a maximum number of turns, 
            and an errors file.
            This method initializes an Environ object by creating a new chessboard, generating a turn list based on 
            the maximum number of turns per player, setting the turn index to 0, and opening the errors file in append 
            mode.

            Attributes:
                board (chess.Board): An object representing the chessboard. Initialized with the standard starting 
                position.
                turn_list (list[str]): A list of strings representing the turns in a game. Each string is in the format 
                'Wn' or 'Bn', where 'W' and 'B' represent white and black players respectively, and 'n' is the turn 
                number. The list is generated based on the maximum number of turns per player defined in the game 
                settings.
                turn_index (int): An integer representing the current turn index. Initialized to 0.
                errors_file (file): A file object representing the errors file. The file is opened in append mode, 
                allowing new errors to be written at the end of the file without overwriting existing content.

            Side Effects:
                Opens the errors file in append mode.
                Modifies the turn list and the turn index. 
        """
        self.error_logger = logging.getLogger(__name__)
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(game_settings.environ_errors_filepath)
        self.error_logger.addHandler(error_handler)

        self.board: chess.Board = chess.Board()
        
        # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
        max_turns = game_settings.max_num_turns_per_player * 2
        self.turn_list: list[str] = [f'{"W" if i % 2 == 0 else "B"}{i // 2 + 1}' for i in range(max_turns)]
        self.turn_index: int = 0
    ### end of constructor

    def get_curr_state(self) -> dict[str, str, list[str]]:
        """
            Retrieves the current state of the chessboard, including the turn index, the current turn, and the legal moves.
            This method constructs a dictionary that represents the current state of the chessboard. The dictionary 
            includes the turn index, the current turn, and a list of all legal moves at the current turn. If the turn 
            index is out of range, an IndexError is raised and logged into the errors file.

            Returns:
                dict[str, str, list[str]]: A dictionary representing the current state of the chessboard. The dictionary 
                has the following keys:
                    - 'turn_index': The current turn index.
                    - 'curr_turn': The current turn, represented as a string.
                    - 'legal_moves': A list of strings, where each string is a legal move at the current turn.
            
            Raises:
                IndexError: Raised when the turn index is out of range of the turn list. The error message is written 
                into the errors file before the exception is re-raised.
            
            Side Effects:
                Writes into the errors file if an IndexError is encountered.
        """
        if not (0 <= self.turn_index < len(self.turn_list)):
            self.error_logger.error(f'ERROR: Turn index out of range: {self.turn_index}\n')
            raise IndexError(f'Turn index out of range: {self.turn_index}')
    
        curr_turn = self.get_curr_turn()
        legal_moves = self.get_legal_moves()

        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    ### end of get_curr_state
    
    def update_curr_state(self) -> None:
        """
            Advances the turn index to update the current state of the chessboard.
            This method is called each time a chess move is loaded onto the chessboard. It increments the turn index 
            to advance the game state. If the turn index reaches the maximum turn index defined in the game settings, 
            an IndexError is raised and logged into the errors file.

            Raises:
                IndexError: Raised when the turn index reaches or exceeds the maximum turn index defined in the game 
                settings. The error message and the current turn index are written into the errors file before the 
                exception is re-raised.

            Side Effects:
                Modifies the turn index by incrementing it by one.
                Writes into the errors file if the maximum turn index is reached or exceeded.
        """
        if self.turn_index >= game_settings.max_turn_index:
            self.error_logger.error(f'ERROR: max_turn_index reached: {self.turn_index} >= {game_settings.max_turn_index}\n')
            raise IndexError(f"Maximum turn index ({game_settings.max_turn_index}) reached!")
    
        if self.turn_index >= len(self.turn_list):
            self.error_logger.error(f'ERROR: turn index out of bounds: {self.turn_index} >= {len(self.turn_list)}\n')
            raise IndexError(f"Turn index out of bounds: {self.turn_index}")
    
        self.turn_index += 1
    ### end of update_curr_state
    
    def get_curr_turn(self) -> str:                        
        """
            Retrieves the current turn from the turn list based on the turn index.
            This method attempts to access the current turn from the turn list using the turn index. If the turn index 
            is out of range, an IndexError is raised and logged into the errors file.

            Returns:
                str: A string representing the current turn. The string corresponds to the value at the current turn 
                index in the turn list. For example, if the turn index is 2 and the turn list is ['W1', 'B1', 'W2', 'B2'], 
                the returned string would be 'W2'.

            Raises:
                IndexError: Raised when the turn index is out of range of the turn list. The error message and the 
                current turn index are written into the errors file before the exception is re-raised.
            
            Side Effects:
                Writes into the errors file if an IndexError is encountered.
        """
        if not (0 <= self.turn_index < len(self.turn_list)):
            self.error_logger.error(f'ERROR: Turn index out of range: {self.turn_index}\n')
            raise IndexError(f'Turn index out of range: {self.turn_index}')
        
        return self.turn_list[self.turn_index]
        ### end of get_curr_turn
    
    def load_chessboard(self, chess_move_str: str, curr_game = 'Game 1') -> None:
        """
            Applies a chess move to the chessboard.

            This method takes a string representing a chess move in Standard Algebraic Notation (SAN), and attempts to 
            apply it to the current state of the chessboard. If the move is invalid or cannot be applied, a ValueError 
            is raised and logged into the errors file.

            Args:
                chess_move_str (str): A string representing the chess move in SAN, such as 'Nf3'.
                curr_game (str, optional): A string representing the current game. Defaults to 'Game 1'.

            Raises:
                ValueError: Raised when the provided move is invalid or cannot be applied to the current chessboard 
                state. The error message, the invalid move, and the current game are written into the errors file 
                before the exception is re-raised.

            Side Effects:
                Modifies the chessboard's state by applying the provided move.
                Writes into the errors file if a ValueError is encountered.
        """
        try:
            self.board.push_san(chess_move_str)
        except ValueError as e:
            self.error_logger.error(f'An error occurred at environ.load_chessboard() for {curr_game}: {e}, unable to load chessboard with {chess_move_str}')
            self.error_logger.error(f'========== End of Environ.load_chessboard ==========\n\n\n')
            raise ValueError(e) from e
    ### end of load_chessboard    

    def pop_chessboard(self) -> None:
        """
            Reverts the state of the chessboard by undoing the most recent move.
            This method attempts to pop the last move from the board's move stack, effectively undoing the last move 
            and reverting the chessboard to its previous state.

            Raises:
                IndexError: Raised when there are no moves to undo, i.e., the move stack is empty. The error message 
                is written into the errors file and the exception is re-raised with a custom message indicating the 
                inability to pop the chessboard due to the error.

            Side Effects:
                Modifies the board's state by undoing the last move.
                Writes into the errors file if an IndexError is encountered.
        """
        try:
            self.board.pop() # this raises an IndexError if the move stack is empty
        except IndexError as e:
            self.error_logger.error(f'An error occurred: {e}, unable to pop chessboard')
            raise IndexError(f"An error occurred: {e}, unable to pop chessboard'")
    ### end of pop_chessboard

    def undo_move(self) -> None:
        """
            Reverts the state of the chessboard by undoing the most recent move.
            This method attempts to pop the last move from the board's move stack and decrement the turn index.
            If the move stack is empty, an IndexError is raised and logged into the errors file.

            Raises:
                IndexError: Raised when there are no moves to undo, i.e., the move stack is empty. The error message 
                and the current turn index are written into the errors file before the exception is re-raised.

            Side Effects:
                Modifies the board's state by undoing the last move and decrementing the turn index.
                Writes into the errors file if an IndexError is encountered.
        """
        try:
            self.board.pop()

            if self.turn_index > 0:
                self.turn_index -= 1
        except IndexError as e:
            self.error_logger.error(f'at, undo_move, An error occurred: {e}, unable to undo move')
            self.error_logger.error(f'turn index: {self.turn_index}\n')
            raise IndexError(e) from e
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
                Writes into the errors file if a ValueError is encountered.
        """
        # this is the anticipated chess move due to opponent's previous chess move. so if White plays Ne4, what is Black likely to play according to the engine?
        anticipated_chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        
        try:
            move = chess.Move.from_uci(anticipated_chess_move)
            self.board.push(move)    
        except ValueError as e:
            self.error_logger.error(f'at, load_chessboard_for_Q_est, An error occurred: {e}, unable to load chessboard with {anticipated_chess_move}')
            raise ValueError(e) from e

    ### end of load_chessboard_for_Q_est

    def reset_environ(self) -> None:
        """
            Resets the chessboard and the turn index.
        """
        self.board.reset()
        self.turn_index = 0
    ### end of reset_environ
    
    def get_legal_moves(self) -> list[str]:   
        """
            Generates a list of all legal moves in Standard Algebraic Notation (SAN) at the current turn.
            This method evaluates the current state of the chessboard and generates a list of all possible legal moves. 
            Each move is converted to SAN for easier readability and consistency.

            Returns:
                list[str]: A list of strings, where each string is a legal move in SAN. The moves are determined based 
                on the current state of the chessboard and the player whose turn it is.
                
            Example:
                If it's white's turn and the possible moves are to move the pawn from e2 to e4 or to move the knight 
                from g1 to f3, the returned list would be ['e4', 'Nf3'].
        """
        return [self.board.san(move) for move in self.board.legal_moves] 
    ### end of get_legal_moves
    