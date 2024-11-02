import chess
from utils import game_settings, constants
from typing import Union, Dict, List

class Environ:
    def __init__(self):
        self.board: chess.Board = chess.Board()            
        max_turns = constants.max_num_turns_per_player * constants.num_players
        self.turn_list: List[str] = [f'{"W" if i % constants.num_players == 0 else "B"}{i // constants.num_players + 1}' for i in range(max_turns)]
        self.turn_index: int = 0
    ### end of constructor

    def get_curr_state(self) -> Dict[str, Union[int, str, List[str]]]:
        curr_turn = self.get_curr_turn()
        legal_moves = self.get_legal_moves()     
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    ### end of get_curr_state
    
    def update_curr_state(self) -> None:   
        self.turn_index += 1
    ### end of update_curr_state
    
    def get_curr_turn(self) -> str:                        
        return self.turn_list[self.turn_index]
        ### end of get_curr_turn
    
    def undo_move(self) -> None:
        self.board.pop()
        if self.turn_index > 0:
            self.turn_index -= 1
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
        move = chess.Move.from_uci(anticipated_chess_move)
        self.board.push(move)
    ### end of load_chessboard_for_Q_est

    def reset_environ(self) -> None:
        self.board.reset()
        self.turn_index = 0
    ### end of reset_environ
    
    def get_legal_moves(self) -> List[str]:
        return [self.board.san(move) for move in self.board.legal_moves]
    ### end of get_legal_moves