import unittest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.Environ import Environ
from src.Agent import Agent
from src import game_settings
import sys
import os

# Determine the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine the project root by going one level up from the script directory
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Insert the project root and 'src' directory into the system path
chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_100, compression = 'zip')
chess_data = chess_data.head(5)

class TestAgent(unittest.TestCase):
    
    def setUp(self):
        self.chess_data = chess_data        
        self.agent = Agent(color='W', chess_data=self.chess_data)        
        self.environ_state = {
            'turn_index': 0,
            'curr_turn': 'W1',
            'legal_moves': [
                            "a3", "a4",
                            "b3", "b4",
                            "c3", "c4",
                            "d3", "d4",
                            "e3", "e4",
                            "f3", "f4",
                            "g3", "g4",
                            "h3", "h4",
                            "Nf3", "Nh3",
                            "Nc3", "Na3"
                        ]
        }

    def test_initialization(self):
        # Test agent initialization
        self.assertEqual(self.agent.color, 'W')
        self.assertEqual(self.agent.chess_data.equals(self.chess_data), True)
        self.assertEqual(self.agent.learn_rate, 0.6)
        self.assertEqual(self.agent.discount_factor, 0.35)
        self.assertEqual(self.agent.is_trained, False)

    def test_invalid_move_handling(self):
        # Test handling of invalid moves in Q-table methods
        with self.assertRaises(KeyError):
            self.agent.change_Q_table_pts('invalid_move', 'W1', 10)
        
    def test_choose_action_with_legal_moves(self):
        # Simulate the agent choosing a move when there are legal moves
        chosen_move = self.agent.choose_action(self.environ_state)
        self.assertIn(chosen_move, self.environ_state['legal_moves'])

    def test_choose_action_with_no_legal_moves(self):
        # Simulate the agent's behavior when no legal moves are available
        self.environ_state['legal_moves'] = []
        chosen_move = self.agent.choose_action(self.environ_state)
        self.assertEqual(chosen_move, '')

    def test_choose_action_updates_Q_table_with_new_moves(self):
        # Simulate the agent encountering new legal moves not in Q-table
        new_legal_moves = ['Nf3', 'Nc3']
        self.environ_state['legal_moves'] = new_legal_moves
        with patch.object(self.agent, 'update_Q_table', wraps=self.agent.update_Q_table) as mock_update:
            self.agent.choose_action(self.environ_state)
            mock_update.assert_called_once_with(new_legal_moves)
    
    def test_choose_action_trained_vs_untrained(self):
        # Generate random integers between 1 and 1000
        random_integers = np.random.randint(1, 1001, size=self.agent.Q_table.shape)
        # Assign the random integers to the q table
        self.agent.Q_table.iloc[:, :] = random_integers

        # Ensure behavior changes based on the agent being trained or not
        self.agent.is_trained = False
        chosen_move_training = self.agent.choose_action(self.environ_state)
        
        self.agent.is_trained = True
        chosen_move_game = self.agent.choose_action(self.environ_state)
        
        self.assertNotEqual(chosen_move_training, chosen_move_game)

    def test_change_Q_table_pts(self):
        # Test adding points to a Q-table cell
        move = 'e4'
        turn = 'W1'
        points = 10
        initial_value = self.agent.Q_table.at[move, turn]
        self.agent.change_Q_table_pts(move, turn, points)
        updated_value = self.agent.Q_table.at[move, turn]
        self.assertEqual(updated_value, initial_value + points)

    def test_invalid_move_handling_in_change_Q_table_pts(self):
        # Test handling of invalid moves in Q-table methods
        with self.assertRaises(KeyError):
            self.agent.change_Q_table_pts('invalid_move', 'W1', 10)

    def test_update_Q_table(self):
        # Test updating the Q-table with new moves
        new_moves = ['g4', 'h4']
        self.agent.update_Q_table(new_moves)
        for move in new_moves:
            self.assertTrue(move in self.agent.Q_table.index)
            self.assertTrue((self.agent.Q_table.loc[move] == 0).all())

    def test_state_retrieval_error(self):
        # Test how agent handles state retrieval errors
        with patch('src.Environ.get_curr_state', side_effect=Exception("State retrieval error")):
            with self.assertRaises(Exception) as context:
                self.agent.choose_action(self.environ_state)
            self.assertTrue('State retrieval error' in str(context.exception))
    
    
if __name__ == '__main__':
    unittest.main()
