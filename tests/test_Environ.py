import sys
import os
print(sys.path)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import unittest
from unittest.mock import patch, Mock
from src.Environ import Environ
import chess
import src.game_settings

class TestEnviron(unittest.TestCase):

    @patch('src.game_settings.max_turn_index', 5)  # Mock the max_turn_index setting
    def test_update_curr_state_within_bounds(self):
        env = Environ()
        env.turn_index = 3  # Set an initial turn index
        env.update_curr_state()
        self.assertEqual(env.turn_index, 4)  # Check if the turn index increased

    @patch('src.game_settings.max_turn_index', 5)
    def test_update_curr_state_raises_error(self):
        env = Environ()
        env.turn_index = 5  # Set the turn index to the maximum
        with self.assertRaises(IndexError):
            env.update_curr_state()

    @patch('src.game_settings.max_turn_index', 5)
    def test_get_curr_state(self):
        env = Environ()
        env.turn_index = 2
        with patch.object(env, 'get_curr_turn', return_value='W3'):
            with patch.object(env, 'get_legal_moves', return_value=['e4', 'Nf3']):
                state = env.get_curr_state()
                self.assertEqual(state['turn_index'], 2)
                self.assertEqual(state['curr_turn'], 'W3')
                self.assertEqual(state['legal_moves'], ['e4', 'Nf3'])

    @patch('src.game_settings.max_turn_index', 5)
    def test_get_curr_state_index_error(self):
        env = Environ()
        env.turn_index = 10 # Out of bounds
        with self.assertRaises(IndexError):
            env.get_curr_state()

    @patch('src.game_settings.max_turn_index', 5)
    def test_get_curr_turn_valid(self):
        env = Environ()
        env.turn_index = 1
        self.assertEqual(env.get_curr_turn(), 'B1')

    @patch('src.game_settings.max_turn_index', 5)
    def test_get_curr_turn_invalid(self):
        env = Environ()
        env.turn_index = 10  # Out of bounds
        with self.assertRaises(IndexError):
            env.get_curr_turn() 

    @patch('chess.Board.push_san')
    def test_load_chessboard_valid(self, mock_push_san):
        env = Environ()
        env.load_chessboard('e4') 
        mock_push_san.assert_called_once_with('e4')
    
    @patch('chess.Board.push_san')
    def test_load_chessboard_invalid(self, mock_push_san):
        mock_push_san.side_effect = ValueError('Invalid move')
        env = Environ()
        with self.assertRaises(ValueError):
            env.load_chessboard('invalid') 

    @patch('chess.Board.pop')
    def test_pop_chessboard(self, mock_pop):
        env = Environ()
        env.pop_chessboard()
        mock_pop.assert_called_once()
    
    @patch('chess.Board.pop')
    def test_pop_chessboard_error(self, mock_pop):
        mock_pop.side_effect = IndexError('Index out of range')
        env = Environ()
        with self.assertRaises(IndexError):
            env.pop_chessboard()
    
    @patch('chess.Board.pop')
    @patch('src.game_settings.max_turn_index', 5)
    def test_undo_move(self, mock_pop):
        env = Environ()
        env.turn_index = 2
        env.undo_move()
        mock_pop.assert_called_once()
        self.assertEqual(env.turn_index, 1)

    @patch('chess.Board.pop')
    @patch('src.game_settings.max_turn_index', 5)
    def test_undo_move_error(self, mock_pop):
        mock_pop.side_effect = IndexError('Index out of range')
        env = Environ()
        with self.assertRaises(IndexError):
            env.undo_move()
    
    @patch('chess.Board.push')
    def test_load_chessboard_for_q_est_valid(self, mock_push):
        env = Environ()
        env.load_chessboard("e4")
        analysis_results = {'anticipated_next_move': 'e5'}
        
        expected_uci_move = chess.Move.from_uci("e7e5")

        with patch.object(chess.Board, 'legal_moves', new_callable=property):
            chess.Board.legal_moves.fget = lambda self: iter([expected_uci_move])
            env.load_chessboard_for_Q_est(analysis_results)
            
        mock_push.assert_called_once_with(expected_uci_move)


    @patch('chess.Board.push')
    def test_load_chessboard_for_q_est_invalid(self, mock_push):
        mock_push.side_effect = ValueError('Invalid move')
        env = Environ()
        analysis_results = {'anticipated_next_move': 'invalid'}  # Invalid algebraic move
        with self.assertRaises(ValueError):
            env.load_chessboard_for_Q_est(analysis_results)

    def test_reset_environ(self):
        env = Environ()
        env.turn_index = 3
        env.board.push_san('e4')
        env.reset_environ()
        self.assertEqual(env.turn_index, 0)
        self.assertEqual(env.board.fen(), chess.STARTING_FEN)

    def test_reset_environ_multiple_times(self):
        env = Environ()
        for _ in range(5):  # Reset multiple times
            env.turn_index = 3
            env.board.push_san('e4')
            env.reset_environ()
            self.assertEqual(env.turn_index, 0)
            self.assertEqual(env.board.fen(), chess.STARTING_FEN)

    def test_get_legal_moves_different_turns(self):
        env = Environ()
        initial_white_moves = env.get_legal_moves()
        self.assertGreater(len(initial_white_moves), 0)
        env.load_chessboard('e4')
        black_moves = env.get_legal_moves()
        self.assertGreater(len(black_moves), 0)
        self.assertNotEqual(initial_white_moves, black_moves)  # Ensure different legal moves for each side

    def test_get_legal_moves(self):
        env = Environ()
        legal_moves = env.get_legal_moves()
        self.assertIsInstance(legal_moves, list)  # Check if legal moves are returned as a list
        self.assertGreater(len(legal_moves), 0)  # Check if there are legal moves at the start

if __name__ == '__main__':
    unittest.main()