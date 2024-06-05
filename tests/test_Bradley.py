import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
import sys
import os
import chess
from pathlib import Path

# Determine the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine the project root by going one level up from the script directory
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.Agent import Agent
from src import game_settings
from src.Environ import Environ
from src.Bradley import Bradley

chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_100, compression = 'zip')
chess_data = chess_data.head(5)

class TestBradley(unittest.TestCase):
    
    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_initialization(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        # Check if the attributes are initialized correctly
        self.assertEqual(bradley.chess_data.equals(chess_data), True)
        self.assertIsInstance(bradley.environ, Environ)
        self.assertIsInstance(bradley.W_rl_agent, Agent)
        self.assertIsInstance(bradley.B_rl_agent, Agent)
        self.assertTrue(mock_engine.called)
        self.assertTrue(mock_open.called)


    # @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    # @patch('builtins.open')
    # def test_file_openings(self, mock_open, mock_engine):
    #     mock_engine.return_value = MagicMock()
    #     mock_open.return_value = MagicMock()

    #     bradley = Bradley(chess_data)

    #     # Check if the errors file, initial training results file, and additional training results file are opened in append mode
    #     expected_calls = [
    #         call(Path(project_root, 'debug/bradley_errors_log.txt').resolve(), 'a'),
    #         call(Path(project_root, 'training_results/initial_training_results.txt').resolve(), 'a'),
    #         call(Path(project_root, 'training_results/additional_training_results.txt').resolve(), 'a')
    #     ]
    #     mock_open.assert_has_calls(expected_calls, any_order=True)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_receive_opp_move_valid(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        valid_move = 'e2e4'
        
        with patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state:
            result = bradley.receive_opp_move(valid_move)

            # Verify that load_chessboard and update_curr_state were called
            mock_load_chessboard.assert_called_once_with(valid_move)
            mock_update_curr_state.assert_called_once()
            self.assertTrue(result)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_receive_opp_move_invalid(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        invalid_move = 'invalid_move'
        
        with patch.object(bradley.environ, 'load_chessboard', side_effect=ValueError('Invalid move')) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state:
            with self.assertRaises(Exception):
                bradley.receive_opp_move(invalid_move)

            # Verify that load_chessboard was called and update_curr_state was not called
            mock_load_chessboard.assert_called_once_with(invalid_move)
            mock_update_curr_state.assert_not_called()
            mock_open().write.assert_called()  # Ensure an error was logged


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_receive_opp_move_state_update(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        valid_move = 'e2e4'
        
        with patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state:
            result = bradley.receive_opp_move(valid_move)

            # Verify that load_chessboard and update_curr_state were called
            mock_load_chessboard.assert_called_once_with(valid_move)
            mock_update_curr_state.assert_called_once()
            self.assertTrue(result)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_rl_agent_selects_chess_move_valid(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        rl_agent_color = 'W'
        
        with patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}) as mock_get_curr_state, \
            patch.object(bradley.W_rl_agent, 'choose_action', return_value='e2e4') as mock_choose_action, \
            patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state:
            
            selected_move = bradley.rl_agent_selects_chess_move(rl_agent_color)

            # Verify that get_curr_state, choose_action, load_chessboard, and update_curr_state were called
            mock_get_curr_state.assert_called_once()
            mock_choose_action.assert_called_once_with({'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}, 'Game 1')
            mock_load_chessboard.assert_called_once_with('e2e4')
            mock_update_curr_state.assert_called_once()
            self.assertEqual(selected_move, 'e2e4')


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_rl_agent_selects_chess_move_no_legal_moves(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        rl_agent_color = 'W'
        
        with patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': []}) as mock_get_curr_state:
            
            with self.assertRaises(Exception):
                bradley.rl_agent_selects_chess_move(rl_agent_color)

            # Verify that get_curr_state was called
            mock_get_curr_state.assert_called_once()


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_rl_agent_selects_chess_move_state_update(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        rl_agent_color = 'W'
        
        with patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}) as mock_get_curr_state, \
            patch.object(bradley.W_rl_agent, 'choose_action', return_value='e2e4') as mock_choose_action, \
            patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state:
            
            selected_move = bradley.rl_agent_selects_chess_move(rl_agent_color)

            # Verify that get_curr_state, choose_action, load_chessboard, and update_curr_state were called
            mock_get_curr_state.assert_called_once()
            mock_choose_action.assert_called_once_with({'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}, 'Game 1')
            mock_load_chessboard.assert_called_once_with('e2e4')
            mock_update_curr_state.assert_called_once()
            self.assertEqual(selected_move, 'e2e4')


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_is_game_over_chessboard(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        
        with patch.object(bradley.environ.board, 'is_game_over', return_value=True) as mock_is_game_over:
            result = bradley.is_game_over()

            # Verify that is_game_over was called and the method returns True
            mock_is_game_over.assert_called_once()
            self.assertTrue(result)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_is_game_over_turn_index(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        
        # Mocking the maximum turn index and setting the current turn index to maximum
        bradley.environ.turn_index = game_settings.max_turn_index
        
        with patch.object(bradley.environ.board, 'is_game_over', return_value=False) as mock_is_game_over:
            result = bradley.is_game_over()

            # Verify that is_game_over was called and the method returns True when turn index reaches maximum
            mock_is_game_over.assert_called_once()
            self.assertTrue(result)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_is_game_over_no_legal_moves(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        # Mocking the current state to have no legal moves
        with patch.object(bradley.environ, 'get_legal_moves', return_value=[]) as mock_get_legal_moves, \
            patch.object(bradley.environ.board, 'is_game_over', return_value=False) as mock_is_game_over:
            result = bradley.is_game_over()

            # Verify that get_legal_moves and is_game_over were called
            mock_get_legal_moves.assert_called_once()
            mock_is_game_over.assert_called_once()
            self.assertTrue(result)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_get_game_outcome_valid(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        
        # Mocking the outcome method to return a specific result
        mock_outcome = MagicMock()
        mock_outcome.result.return_value = '1-0'
        
        with patch.object(bradley.environ.board, 'outcome', return_value=mock_outcome) as mock_outcome_method:
            result = bradley.get_game_outcome()

            # Verify that outcome and result methods were called and the method returns the correct result
            mock_outcome_method.assert_called_once()
            mock_outcome.result.assert_called_once()
            self.assertEqual(result, '1-0')


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_get_game_outcome_invalid(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        # Mocking the outcome method to raise an AttributeError
        with patch.object(bradley.environ.board, 'outcome', side_effect=AttributeError('Cannot determine outcome')) as mock_outcome_method:
            result = bradley.get_game_outcome()

            # Verify that outcome was called and the method returns the correct error message
            mock_outcome_method.assert_called_once()
            self.assertTrue(result.startswith('error at get_game_outcome: '))


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_get_game_termination_reason_valid(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        # Mocking the outcome method to return a specific termination reason
        mock_outcome = MagicMock()
        mock_outcome.termination_reason = 'normal'
        
        with patch.object(bradley.environ.board, 'outcome', return_value=mock_outcome) as mock_outcome_method:
            result = bradley.get_game_termination_reason()

            # Verify that outcome and termination_reason were called and the method returns the correct reason
            mock_outcome_method.assert_called_once()
            self.assertEqual(result, 'normal')


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_get_game_termination_reason_invalid(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        # Mocking the outcome method to raise an AttributeError
        with patch.object(bradley.environ.board, 'outcome', side_effect=AttributeError('Cannot determine termination reason')) as mock_outcome_method:
            result = bradley.get_game_termination_reason()

            # Verify that outcome was called and the method returns the correct error message
            mock_outcome_method.assert_called_once()
            self.assertTrue(result.startswith('error at get_game_termination_reason: '))


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_train_rl_agents(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        with patch.object(bradley, 'assign_points_to_Q_table', return_value=None) as mock_assign_points, \
            patch.object(bradley.W_rl_agent, 'choose_action', return_value='e2e4') as mock_choose_action_w, \
            patch.object(bradley.B_rl_agent, 'choose_action', return_value='e7e5') as mock_choose_action_b, \
            patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}) as mock_get_curr_state, \
            patch.object(bradley, 'find_estimated_Q_value', return_value=0) as mock_find_est_q_value, \
            patch.object(bradley, 'find_next_Qval', return_value=0) as mock_find_next_qval, \
            patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state, \
            patch.object(bradley.environ, 'reset_environ', return_value=None) as mock_reset_environ:

            bradley.train_rl_agents(pd.DataFrame())

            # Verify that the key methods were called
            self.assertTrue(mock_assign_points.called)
            self.assertTrue(mock_choose_action_w.called)
            self.assertTrue(mock_choose_action_b.called)
            self.assertTrue(mock_get_curr_state.called)
            self.assertTrue(mock_find_est_q_value.called)
            self.assertTrue(mock_find_next_qval.called)
            self.assertTrue(mock_load_chessboard.called)
            self.assertTrue(mock_update_curr_state.called)
            self.assertTrue(mock_reset_environ.called)
            self.assertTrue(bradley.W_rl_agent.is_trained)
            self.assertTrue(bradley.B_rl_agent.is_trained)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_train_rl_agents_Q_value_updates(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        with patch.object(bradley, 'assign_points_to_Q_table', return_value=None) as mock_assign_points, \
            patch.object(bradley.W_rl_agent, 'choose_action', return_value='e2e4') as mock_choose_action_w, \
            patch.object(bradley.B_rl_agent, 'choose_action', return_value='e7e5') as mock_choose_action_b, \
            patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}) as mock_get_curr_state, \
            patch.object(bradley, 'find_estimated_Q_value', return_value=1) as mock_find_est_q_value, \
            patch.object(bradley, 'find_next_Qval', return_value=2) as mock_find_next_qval, \
            patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state, \
            patch.object(bradley.environ, 'reset_environ', return_value=None) as mock_reset_environ:
            
            bradley.train_rl_agents(pd.DataFrame())

            # Verify that the Q-value related methods were called with correct parameters
            self.assertTrue(mock_find_est_q_value.called)
            self.assertTrue(mock_find_next_qval.called)
            self.assertTrue(mock_assign_points.called)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_train_rl_agents_error_handling(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        with patch.object(bradley, 'assign_points_to_Q_table', side_effect=Exception('Test error')) as mock_assign_points, \
            patch.object(bradley.W_rl_agent, 'choose_action', return_value='e2e4') as mock_choose_action_w, \
            patch.object(bradley.B_rl_agent, 'choose_action', return_value='e7e5') as mock_choose_action_b, \
            patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}) as mock_get_curr_state, \
            patch.object(bradley, 'find_estimated_Q_value', return_value=1) as mock_find_est_q_value, \
            patch.object(bradley, 'find_next_Qval', return_value=2) as mock_find_next_qval, \
            patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state, \
            patch.object(bradley.environ, 'reset_environ', return_value=None) as mock_reset_environ:

            with self.assertRaises(Exception):
                bradley.train_rl_agents(pd.DataFrame())

            # Verify that the error was logged
            mock_open().write.assert_called()


    # @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    # @patch('builtins.open')
    # def test_continue_training_rl_agents(self, mock_open, mock_engine):
    #     mock_engine.return_value = MagicMock()
    #     mock_open.return_value = MagicMock()
    #     bradley = Bradley(chess_data)
        
    #     with self.assertRaises(NotImplementedError):
    #         bradley.continue_training_rl_agents(10)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_assign_points_to_Q_table(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        chess_move = 'e2e4'
        curr_turn = 'W1'
        curr_Qval = 10
        rl_agent_color = 'W'
        
        with patch.object(bradley.W_rl_agent, 'change_Q_table_pts', return_value=None) as mock_change_Q_table_pts:
            bradley.assign_points_to_Q_table(chess_move, curr_turn, curr_Qval, rl_agent_color)
            # Verify that change_Q_table_pts was called with correct parameters
            mock_change_Q_table_pts.assert_called_once_with(chess_move, curr_turn, curr_Qval)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_assign_points_to_Q_table_key_error(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()

        bradley = Bradley(chess_data)
        chess_move = 'e2e4'
        curr_turn = 'W1'
        curr_Qval = 10
        rl_agent_color = 'W'
        
        with patch.object(bradley.W_rl_agent, 'change_Q_table_pts', side_effect=[KeyError('Test KeyError'), None]) as mock_change_Q_table_pts, \
            patch.object(bradley.W_rl_agent, 'update_Q_table', return_value=None) as mock_update_Q_table:
            
            bradley.assign_points_to_Q_table(chess_move, curr_turn, curr_Qval, rl_agent_color)

            # Verify that change_Q_table_pts was called twice and update_Q_table was called once
            self.assertEqual(mock_change_Q_table_pts.call_count, 2)
            mock_update_Q_table.assert_called_once_with([chess_move])
            mock_open().write.assert_called()  # Ensure an error was logged


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_rl_agent_plays_move(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        chess_move = 'e2e4'
        curr_game = 'Game 1'
        
        with patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state:
            bradley.rl_agent_plays_move(chess_move, curr_game)
            # Verify that load_chessboard and update_curr_state were called with correct parameters
            mock_load_chessboard.assert_called_once_with(chess_move, curr_game)
            mock_update_curr_state.assert_called_once()


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_rl_agent_plays_move_error_handling(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        chess_move = 'e2e4'
        curr_game = 'Game 1'
        
        with patch.object(bradley.environ, 'load_chessboard', side_effect=Exception('Test error')) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state:
            
            with self.assertRaises(Exception):
                bradley.rl_agent_plays_move(chess_move, curr_game)

            # Verify that load_chessboard was called and the error was logged
            mock_load_chessboard.assert_called_once_with(chess_move, curr_game)
            mock_update_curr_state.assert_not_called()
            mock_open().write.assert_called()  # Ensure an error was logged


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_find_estimated_Q_value(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        with patch.object(bradley, 'analyze_board_state', return_value={'centipawn_score': 20, 'anticipated_next_move': 'e7e5'}) as mock_analyze_board, \
            patch.object(bradley.environ.board, 'push_uci', return_value=None) as mock_push_uci, \
            patch.object(bradley.environ.board, 'pop', return_value=None) as mock_pop:
            estimated_Q_value = bradley.find_estimated_Q_value()

            # Verify that analyze_board_state, push_uci, and pop were called
            mock_analyze_board.assert_called_once_with(bradley.environ.board)
            mock_push_uci.assert_called_once_with('e7e5')
            mock_pop.assert_called_once()
            self.assertEqual(estimated_Q_value, 20)  # centipawn score used for Q-value estimation


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_find_estimated_Q_value_error_handling(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)
        
        with patch.object(bradley, 'analyze_board_state', side_effect=Exception('Test error')) as mock_analyze_board, \
            patch.object(bradley.environ.board, 'push_uci', return_value=None) as mock_push_uci, \
            patch.object(bradley.environ.board, 'pop', return_value=None) as mock_pop:
            
            with self.assertRaises(Exception):
                bradley.find_estimated_Q_value()

            # Verify that analyze_board_state was called and the error was logged
            mock_analyze_board.assert_called_once_with(bradley.environ.board)
            mock_push_uci.assert_not_called()
            mock_pop.assert_not_called()
            mock_open().write.assert_called()  # Ensure an error was logged


    def test_find_next_Qval(self):
        bradley = Bradley(chess_data)
        curr_Qval = 10
        learn_rate = 0.6
        reward = 5
        discount_factor = 0.9
        est_Qval = 20

        next_Qval = bradley.find_next_Qval(curr_Qval, learn_rate, reward, discount_factor, est_Qval)

        # Calculate the expected Q-value using the SARSA formula
        expected_Qval = curr_Qval + learn_rate * (reward + (discount_factor * est_Qval) - curr_Qval)
        self.assertEqual(next_Qval, expected_Qval)


    # @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    # @patch('builtins.open')
    # def test_analyze_board_state(self, mock_open, mock_engine):
    #     mock_engine_instance = MagicMock()
    #     mock_engine.return_value = mock_engine_instance
    #     mock_open.return_value = MagicMock()

    #     bradley = Bradley(chess_data)
    #     board = bradley.environ.board

    #     # Mocking the analysis results from the engine
    #     analysis_results = [{
    #         'score': chess.engine.Cp(20),
    #         'pv': ['e7e5']
    #     }]
        
    #     with patch.object(mock_engine_instance, 'analyze', return_value=analysis_results):
    #         result = bradley.analyze_board_state(board)
            
    #         # Verify the analysis result is correctly interpreted
    #         self.assertEqual(result['centipawn_score'], 20)
    #         self.assertEqual(result['anticipated_next_move'], 'e7e5')


    # @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    # @patch('builtins.open')
    # def test_analyze_board_state_error_handling(self, mock_open, mock_engine):
    #     mock_engine_instance = MagicMock()
    #     mock_engine.return_value = mock_engine_instance
    #     mock_open.return_value = MagicMock()
    #     bradley = Bradley(chess_data)
    #     board = bradley.environ.board

    #     # Mocking the engine to raise an exception during analysis
    #     with patch.object(mock_engine_instance, 'analyze', side_effect=Exception('Test error')):
    #         with self.assertRaises(Exception):
    #             bradley.analyze_board_state(board)

    #         # Verify that the error was logged
    #         mock_open().write.assert_called()  # Ensure an error was logged


    def test_get_reward(self):
        bradley = Bradley(chess_data)
        # Define game settings for rewards
        game_settings.piece_development = 5
        game_settings.capture = 10
        game_settings.promotion = 20
        game_settings.promotion_queen = 25
        # Test different move types and their corresponding rewards
        move_development = 'Nf3'
        move_capture = 'Nxf7'
        move_promotion = 'e8=Q'
        move_promotion_queen = 'e8=Q'
    
        reward_development = bradley.get_reward(move_development)
        reward_capture = bradley.get_reward(move_capture)
        reward_promotion = bradley.get_reward(move_promotion)
        reward_promotion_queen = bradley.get_reward(move_promotion_queen)

        self.assertEqual(reward_development, game_settings.piece_development)
        self.assertEqual(reward_capture, game_settings.capture)
        self.assertEqual(reward_promotion, game_settings.promotion + game_settings.promotion_queen)
        self.assertEqual(reward_promotion_queen, game_settings.promotion + game_settings.promotion_queen)


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_identify_corrupted_games(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)

        # Mocking methods to simulate errors for corrupted games
        with patch.object(bradley.environ, 'get_curr_state', side_effect=[{}, Exception('Test error')]), \
            patch.object(bradley.W_rl_agent, 'choose_action', return_value='e2e4'), \
            patch.object(bradley.B_rl_agent, 'choose_action', return_value='e7e5'), \
            patch.object(bradley.environ, 'load_chessboard', side_effect=[None, Exception('Test error')]), \
            patch.object(bradley.environ, 'update_curr_state', return_value=None), \
            patch.object(bradley.environ, 'reset_environ', return_value=None):
            bradley.identify_corrupted_games()

            # Verify that the corrupted game was added to the list and the error was logged
            self.assertTrue(mock_open().write.called)  # Ensure an error was logged
            self.assertIn(1, bradley.corrupted_games_list)
        

    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_generate_Q_est_df(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)

        # Mocking methods to generate Q-values
        with patch.object(bradley, 'find_estimated_Q_value', return_value=50) as mock_find_estimated_Q_value, \
            patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}) as mock_get_curr_state, \
            patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state, \
            patch.object(bradley.environ, 'reset_environ', return_value=None) as mock_reset_environ:
            q_est_vals_file_path = 'q_est_vals_test.txt'
            bradley.generate_Q_est_df(q_est_vals_file_path)

            # Verify that find_estimated_Q_value and other methods were called
            self.assertTrue(mock_find_estimated_Q_value.called)
            self.assertTrue(mock_get_curr_state.called)
            self.assertTrue(mock_load_chessboard.called)
            self.assertTrue(mock_update_curr_state.called)
            self.assertTrue(mock_reset_environ.called)
            mock_open.assert_called_with(q_est_vals_file_path, 'w')  # Ensure the file was opened for writing
            mock_open().write.assert_called()  # Ensure data was written to the file


    @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    @patch('builtins.open')
    def test_generate_Q_est_df_error_handling(self, mock_open, mock_engine):
        mock_engine.return_value = MagicMock()
        mock_open.return_value = MagicMock()
        bradley = Bradley(chess_data)

        # Mocking methods to simulate errors during Q-value generation
        with patch.object(bradley, 'find_estimated_Q_value', side_effect=Exception('Test error')) as mock_find_estimated_Q_value, \
            patch.object(bradley.environ, 'get_curr_state', return_value={'turn_index': 0, 'curr_turn': 'W1', 'legal_moves': ['e2e4']}) as mock_get_curr_state, \
            patch.object(bradley.environ, 'load_chessboard', return_value=None) as mock_load_chessboard, \
            patch.object(bradley.environ, 'update_curr_state', return_value=None) as mock_update_curr_state, \
            patch.object(bradley.environ, 'reset_environ', return_value=None) as mock_reset_environ:
            q_est_vals_file_path = 'q_est_vals_test.txt'
            
            with self.assertRaises(Exception):
                bradley.generate_Q_est_df(q_est_vals_file_path)

            # Verify that find_estimated_Q_value was called and the error was logged
            mock_find_estimated_Q_value.assert_called()
            mock_open().write.assert_called()  # Ensure an error was logged


    # @patch('src.Bradley.chess.engine.SimpleEngine.popen_uci')
    # @patch('builtins.open')
    # def test_analyze_board_state_mate_score(self, mock_open, mock_engine):
    #     mock_engine_instance = MagicMock()
    #     mock_engine.return_value = mock_engine_instance
    #     mock_open.return_value = MagicMock()

    #     bradley = Bradley(chess_data)
    #     board = bradley.environ.board

    #     # Mocking the analysis results from the engine
    #     analysis_result = [{
    #         'score': chess.engine.Mate(3),
    #         'pv': ['e7e5']
    #     }]

    #     with patch.object(mock_engine_instance, 'analyse', return_value=analysis_result):
    #         result = bradley.analyze_board_state(board)

    #         # Verify the analysis result is correctly interpreted
    #         self.assertEqual(result['mate_score'], 3)
    #         self.assertIsNone(result['centipawn_score'])
    #         self.assertEqual(result['anticipated_next_move'], 'e7e5')


if __name__ == '__main__':
    unittest.main()