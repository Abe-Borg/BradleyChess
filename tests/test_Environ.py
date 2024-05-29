import sys
import os
import unittest
from unittest.mock import patch, PropertyMock
import chess

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.Environ import Environ

max_num_turns_per_player = 200
max_turn_index = max_num_turns_per_player * 2 - 1

class TestEnviron(unittest.TestCase):
    def test_init(self):
        """
        Tests the initialization of the Environ class.

        This test case creates an instance of the Environ class and checks if:
        - The 'board' attribute is an instance of the chess.Board class.
        - The 'turn_index' attribute is initialized to 0.
        """
        env = Environ()
        self.assertIsInstance(env.board, chess.Board)
        self.assertEqual(env.turn_index, 0)

    def test_update_curr_state_within_bounds(self):
        """
        Tests the update_curr_state method within valid bounds.

        This test case creates an instance of the Environ class, sets an initial turn index, 
        calls the update_curr_state method, and checks if the turn index is incremented correctly.
        """
        env = Environ()
        env.turn_index = 3  # Set an initial turn index
        env.update_curr_state()
        self.assertEqual(env.turn_index, 4)  # Check if the turn index increased

    def test_update_curr_state_raises_error(self):
        """
        Tests the update_curr_state method with an out-of-bounds index.

        This test case creates an instance of the Environ class, sets the turn index to the maximum value,
        and checks if an IndexError is raised when trying to call the update_curr_state method.
        """
        env = Environ()
        env.turn_index = max_turn_index
        with self.assertRaises(IndexError):
            env.update_curr_state()

    def test_get_curr_state(self):
        """
        Tests the get_curr_state method.

        This test case creates an instance of the Environ class, sets an initial turn index, 
        mocks the get_curr_turn and get_legal_moves methods, and checks if the returned state is correct.
        """
        env = Environ()
        env.turn_index = 2
        with patch.object(env, 'get_curr_turn', return_value='W3'):
            with patch.object(env, 'get_legal_moves', return_value=['e4', 'Nf3']):
                state = env.get_curr_state()
                self.assertEqual(state['turn_index'], 2)
                self.assertEqual(state['curr_turn'], 'W3')
                self.assertEqual(state['legal_moves'], ['e4', 'Nf3'])

    def test_get_curr_state_index_error(self):
        """
        Tests the get_curr_state method with an out-of-bounds index.

        This test case creates an instance of the Environ class, sets the turn index to a value beyond the maximum,
        and checks if an IndexError is raised when trying to call the get_curr_state method.
        """
        env = Environ()
        env.turn_index = max_turn_index + 1  # Out of bounds
        with self.assertRaises(IndexError):
            env.get_curr_state()

    def test_get_curr_turn_valid(self):
        """
        Tests the get_curr_turn method with a valid turn index.

        This test case creates an instance of the Environ class, sets the turn index to a valid value,
        and checks if the returned turn is correct.
        """
        env = Environ()
        env.turn_index = 1
        self.assertEqual(env.get_curr_turn(), 'B1')

    def test_get_curr_turn_invalid(self):
        """
        Tests the get_curr_turn method with an invalid turn index.

        This test case creates an instance of the Environ class, sets the turn index to a value beyond the maximum,
        and checks if an IndexError is raised when trying to call the get_curr_turn method.
        """
        env = Environ()
        env.turn_index = max_turn_index + 1  # Out of bounds
        with self.assertRaises(IndexError):
            env.get_curr_turn() 

    @patch('chess.Board.push_san')
    def test_load_chessboard_valid(self, mock_push_san):
        """
        Tests the load_chessboard method with a valid move.

        This test case creates an instance of the Environ class, calls the load_chessboard method with a valid move,
        and checks if the push_san method of the chess.Board class is called with the correct argument.
        """
        env = Environ()
        env.load_chessboard('e4') 
        mock_push_san.assert_called_once_with('e4')
    
    @patch('chess.Board.push_san')
    def test_load_chessboard_invalid(self, mock_push_san):
        """
        Tests the load_chessboard method with an invalid move.

        This test case creates an instance of the Environ class, mocks the push_san method of the chess.Board class to raise a ValueError,
        calls the load_chessboard method with an invalid move, and checks if a ValueError is raised.
        """
        mock_push_san.side_effect = ValueError('Invalid move')
        env = Environ()
        with self.assertRaises(ValueError):
            env.load_chessboard('invalid') 

    @patch('chess.Board.pop')
    def test_pop_chessboard(self, mock_pop):
        """
        Tests the pop_chessboard method.

        This test case creates an instance of the Environ class, mocks the pop method of the chess.Board class,
        calls the pop_chessboard method, and checks if the pop method of the chess.Board class is called once.
        """
        env = Environ()
        env.pop_chessboard()
        mock_pop.assert_called_once()
    
    @patch('chess.Board.pop')
    def test_pop_chessboard_error(self, mock_pop):
        """
        Tests the pop_chessboard method with an out-of-bounds index.

        This test case creates an instance of the Environ class, mocks the pop method of the chess.Board class to raise an IndexError,
        calls the pop_chessboard method, and checks if an IndexError is raised.
        """
        mock_pop.side_effect = IndexError('Index out of range')
        env = Environ()
        with self.assertRaises(IndexError):
            env.pop_chessboard()
    
    @patch('chess.Board.pop')
    def test_undo_move(self, mock_pop):
        """
        Tests the undo_move method.

        This test case creates an instance of the Environ class, sets an initial turn index, 
        calls the undo_move method, and checks if the pop method of the chess.Board class is called once 
        and if the turn index is decremented correctly.
        """
        env = Environ()
        env.turn_index = 2
        env.undo_move()
        mock_pop.assert_called_once()
        self.assertEqual(env.turn_index, 1)

    @patch('chess.Board.pop')
    def test_undo_move_error(self, mock_pop):
        """
        Tests the undo_move method with an out-of-bounds index.

        This test case creates an instance of the Environ class, mocks the pop method of the chess.Board class to raise an IndexError,
        calls the undo_move method, and checks if an IndexError is raised.
        """
        mock_pop.side_effect = IndexError('Index out of range')
        env = Environ()
        with self.assertRaises(IndexError):
            env.undo_move()
    
    @patch('chess.Board.push')
    def test_load_chessboard_for_q_est_valid(self, mock_push):
        """
        Tests the load_chessboard_for_q_est method with a valid move.

        This test case creates an instance of the Environ class, calls the load_chessboard method with a valid move,
        sets up analysis results with an anticipated next move, and checks if the expected UCI move is correct.
        """
        env = Environ()
        env.load_chessboard("e4")
        analysis_results = {'anticipated_next_move': 'e7e5'}
        expected_uci_move = chess.Move.from_uci("e7e5")

        with patch('chess.Board.legal_moves', new_callable=PropertyMock) as mock_legal_moves:
            mock_legal_moves.return_value = iter([expected_uci_move])
            env.load_chessboard_for_Q_est(analysis_results)

        calls = [unittest.mock.call(chess.Move.from_uci("e2e4")), unittest.mock.call(expected_uci_move)]
        mock_push.assert_has_calls(calls, any_order=False)

    @patch('chess.Board.push')
    def test_load_chessboard_for_q_est_invalid(self, mock_push):
        """
        Tests the load_chessboard_for_q_est method with an invalid move.

        This test case creates an instance of the Environ class, mocks the push method of the chess.Board class to raise a ValueError,
        sets up analysis results with an invalid anticipated next move, and checks if a ValueError is raised.
        """
        mock_push.side_effect = ValueError('Invalid move')
        env = Environ()
        analysis_results = {'anticipated_next_move': 'invalid'}  # Invalid algebraic move
        with self.assertRaises(ValueError):
            env.load_chessboard_for_Q_est(analysis_results)

    def test_reset_environ(self):
        """
        Tests the reset_environ method.

        This test case creates an instance of the Environ class, sets an initial turn index, makes a move on the board,
        calls the reset_environ method, and checks if the turn index is reset to 0 and the board is reset to the starting position.
        """
        env = Environ()
        env.turn_index = 3
        env.board.push_san('e4')
        env.reset_environ()
        self.assertEqual(env.turn_index, 0)
        self.assertEqual(env.board.fen(), chess.STARTING_FEN)

    def test_reset_environ_multiple_times(self):
        """
        Tests the reset_environ method multiple times.

        This test case creates an instance of the Environ class, sets an initial turn index, makes a move on the board,
        and calls the reset_environ method multiple times. After each reset, it checks if the turn index is reset to 0 
        and the board is reset to the starting position.
        """
        env = Environ()
        for _ in range(5):  # Reset multiple times
            env.turn_index = 3
            env.board.push_san('e4')
            env.reset_environ()
            self.assertEqual(env.turn_index, 0)
            self.assertEqual(env.board.fen(), chess.STARTING_FEN)

    def test_get_legal_moves_different_turns(self):
        """
        Tests the get_legal_moves method for different turns.

        This test case creates an instance of the Environ class, gets the initial legal moves for white, makes a move on the board,
        gets the legal moves for black, and checks if the number of legal moves for each side is greater than 0 and if the legal moves for each side are different.
        """
        env = Environ()
        initial_white_moves = env.get_legal_moves()
        self.assertGreater(len(initial_white_moves), 0)
        env.load_chessboard('e4')
        black_moves = env.get_legal_moves()
        self.assertGreater(len(black_moves), 0)
        self.assertNotEqual(initial_white_moves, black_moves)  # Ensure different legal moves for each side

    def test_get_legal_moves(self):
        """
        Tests the get_legal_moves method.

        This test case creates an instance of the Environ class, gets the initial legal moves, 
        and checks if the legal moves are returned as a list and if there are legal moves at the start.
        """
        env = Environ()
        legal_moves = env.get_legal_moves()
        self.assertIsInstance(legal_moves, list)  # Check if legal moves are returned as a list
        self.assertGreater(len(legal_moves), 0)  # Check if there are legal moves at the start

    def test_turn_index_at_boundaries(self):
        """
        Test updating the turn index when it is at the boundary values.

        This test sets the turn index to one less than the maximum allowed value,
        updates the state to reach the maximum value, and verifies the turn index.
        It then attempts to update the state beyond the maximum value and checks
        that an IndexError is raised.
        """
        env = Environ()
        env.turn_index = max_turn_index - 1
        env.update_curr_state()
        self.assertEqual(env.turn_index, max_turn_index)
        with self.assertRaises(IndexError):
            env.update_curr_state()

    def test_multiple_valid_moves(self):
        """
        Test applying multiple valid moves in succession.

        This test loads a series of valid chess moves onto the chessboard,
        verifies the moves are applied correctly, and checks the final board
        state using its FEN string representation.
        """
        env = Environ()
        moves = ['e4', 'e5', 'Nf3', 'Nc6']
        for move in moves:
            env.load_chessboard(move)
        self.assertEqual(env.board.fen(), 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3')

    def test_sequence_of_moves_with_resets(self):
        """
        Test the consistency of the environment after applying moves and resetting.

        This test applies a series of valid chess moves, resets the environment,
        and checks that the turn index and board state are reset to their initial values.
        """
        env = Environ()
        moves = ['e4', 'e5', 'Nf3', 'Nc6']
        for move in moves:
            env.load_chessboard(move)
        env.reset_environ()
        self.assertEqual(env.turn_index, 0)
        self.assertEqual(env.board.fen(), chess.STARTING_FEN)

    def test_stress_test(self):
        """
        Stress test the environment with a large number of moves and undos.

        This test repeatedly loads and undoes a move to ensure the environment
        remains consistent and does not encounter errors under extended use.
        """
        env = Environ()
        for _ in range(50):
            env.load_chessboard('e4')
            env.update_curr_state()
            env.undo_move()
        self.assertEqual(env.turn_index, 0)
        self.assertEqual(env.board.fen(), chess.STARTING_FEN)


if __name__ == '__main__':
    unittest.main()