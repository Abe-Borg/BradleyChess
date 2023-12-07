import game_settings
import pandas as pd
import chess

class Environ:
    """Manages the chessboard and determines its state.
    """
    def __init__(self):
        """
        Attributes:
            board (chess.Board): An object representing the chessboard.
            turn_list (list[str]): A list that is utilized to keep track of the current turn (W1, B1 ... Wn, Bn).
            turn_index (int): An integer that is incremented as each player makes a move.
        """
        self.errors_file = open(game_settings.environ_errors_filepath, 'a')
        self.board: chess.Board = chess.Board()

        # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
        max_turns = game_settings.max_num_turns_per_player * 2
        self.turn_list: list[str] = [f'{"W" if i % 2 == 0 else "B"}{i // 2 + 1}' for i in range(max_turns)]
        self.turn_index: int = 0
    ### end of constructor

    def __del__(self):
        self.errors_file.close()
    ### end of Bradley destructor ###

    def get_curr_state(self) -> dict[str, str, list[str]]:
        """Returns a dictionary that describes the current state of the chessboard and the current turn.
        Returns:
            dict[str, str, list[str]]: A dictionary that defines the current state that an agent will act on.
        """
        try:
            curr_turn = self.get_curr_turn()
        except IndexError as e:
            self.errors_file.write(f'An error at get_curr_state occurred: {e}, unable to get current turn')
            raise IndexError from e

        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': self.get_legal_moves()}
    ### end of get_curr_state
    
    def update_curr_state(self) -> None:
        """Updates the current state of the chessboard.
        The state is updated each time a chess move is loaded to the chessboard. 
        Only the index needs to be updated here. The board is updated by other methods.
        """
        if self.turn_index < game_settings.max_turn_index:
            self.turn_index += 1
        else:
            self.errors_file.write(f'ERROR: max_turn_index reached: {self.turn_index} >= {game_settings.max_turn_index}\n')
            raise IndexError(f"Maximum turn index ({game_settings.max_turn_index}) reached!")
    ### end of update_curr_state
    
    def get_curr_turn(self) -> str:                        
        """Returns the string of the current turn.
        Returns:
            str: A string that corresponds to the current turn, such as 'W2' for index 2.
        Raises:
            IndexError: If the turn index is out of range.
        """
        try: 
            curr_turn = self.turn_list[self.turn_index]
            return curr_turn
        except IndexError as e:
            self.errors_file.write(f'at get_curr_turn, list index out of range, turn index is {self.turn_index}, error desc is: {e}')
            raise IndexError from e
    ### end of get_curr_turn
    
    def load_chessboard(self, chess_move_str: str, curr_game = 'Game 1') -> None:
        """Loads a chess move on the chessboard.
        Args:
            chess_move_str (str): A string representing the chess move, such as 'Nf3'.
        Returns:
            bool: A boolean value indicating whether the move was successfully loaded.
        """
        try:
            self.board.push_san(chess_move_str)
        except ValueError as e:
            self.errors_file.write(f'An error occurred at environ.load_chessboard() for {curr_game}: {e}, unable to load chessboard with {chess_move_str}')
            self.errors_file.write(f'========== End of Environ.load_chessboard ==========\n\n\n')
            raise ValueError from e        
    ### end of load_chessboard    

    def pop_chessboard(self) -> None:
        """Pops the most recent move applied to the chessboard.
        Raises:
            IndexError: If the move stack is empty.
        """
        try:
            self.board.pop()
        except IndexError as e:
            self.errors_file.write(f'An error occurred: {e}, unable to pop chessboard')
            raise IndexError(f"An error occurred: {e}, unable to pop chessboard'")
    ### end of pop_chessboard

    def undo_move(self) -> None:
        """Undoes the most recent move applied to the chessboard.
        Raises:
            IndexError: If the move stack is empty.
        """
        try:
            self.board.pop()
            self.turn_index -= 1
        except IndexError as e:
            self.errors_file.write(f'at, undo_move, An error occurred: {e}, unable to undo move')
            self.errors_file.write(f'turn index: {self.turn_index}\n')
            raise IndexError from e
    ### end of undo_move

    def load_chessboard_for_Q_est(self, analysis_results: list[dict]) -> None:
        """Loads the chessboard using a Move.uci string during training.
        This method works in tandem with the Stockfish analysis during training.
        the method loads the anticipated next move from the analysis results.
        
        Args:
            analysis_results (list[dict]): A list of dictionaries containing the analysis results.
                Each dictionary has the form {'mate_score': <some score>, 'centipawn_score': <some score>,
                'anticipated_next_move': <move>}.
        """
        # this is the anticipated chess move due to opponent's previous chess move. so if White plays Ne4, what is Black like to play?
        anticipated_chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        try:
            self.board.push(anticipated_chess_move)
        except ValueError as e:
            self.errors_file.write(f'at, load_chessboard_for_Q_est, An error occurred: {e}, unable to load chessboard with {anticipated_chess_move}')
            raise ValueError from e
    ### end of load_chessboard_for_Q_est

    def reset_environ(self) -> None:
        self.board.reset()
        self.turn_index = 0
    ### end of reset_environ
    
    def get_legal_moves(self) -> list[str]:   
        """Returns a list of legal moves at the current turn.
        Returns:
            list[str]: A list of strings representing the legal moves at the current turn, given the board state.
        """
        legal_moves = [self.board.san(move) for move in self.board.legal_moves]    
        return legal_moves
    ### end of get_legal_moves
    