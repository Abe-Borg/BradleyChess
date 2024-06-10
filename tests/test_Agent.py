import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Determine the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Determine the project root by going one level up from the script directory
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.Agent import Agent
from src import game_settings

class TestAgent(unittest.TestCase):
    """
    The TestAgent class is a unit test class for the Agent class in the src module.
    This class inherits from unittest.TestCase which is a standard test case class in Python's unittest framework.

    Attributes:
        agent (src.Agent): An instance of the Agent class that is being tested.
        environ_state (dict): A dictionary representing the environment state for the agent.
    """
    def setUp(self):
        """
        The setUp method is a special method in unittest.TestCase.
        It is run before each test method (i.e., methods starting with 'test') in the class. It is used to set up any state that is common across multiple test methods.
        In this case, it sets up a subset of chess game data, an instance of the Agent class, and a dictionary representing the environment state for the agent.
        """        
        self.agent = Agent(color='W')
        self.agent.Q_table = pd.read_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
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
        """
            This method tests the initialization of the Agent class.
            It checks if the agent's color is 'W', if the learning rate is 0.6, if the discount factor is 0.35, and if the agent is not trained initially.
            The method uses the assertEqual method from unittest.TestCase to check if the actual values are equal to the expected values.

            Raises:
                AssertionError: If any of the actual values do not match the expected values.
        """
        self.assertEqual(self.agent.color, 'W')
        self.assertEqual(self.agent.learn_rate, 0.6)
        self.assertEqual(self.agent.discount_factor, 0.35)
        self.assertEqual(self.agent.is_trained, False)
        # assert that q_table was assigned and is not None
        self.assertIsNotNone(self.agent.Q_table)

    def test_invalid_move_handling(self):
        """
            This method tests the handling of invalid moves in the change_Q_table_pts method of the Agent class.
            It checks if the agent correctly raises a KeyError when an invalid move is attempted to be changed in the Q-table.

            Raises:
                AssertionError: If the agent does not raise a KeyError when an invalid move is attempted to be changed in the Q-table.
        """
        # Test handling of invalid moves in Q-table methods
        with self.assertRaises(KeyError):
            self.agent.change_Q_table_pts('invalid_move', 'W1', 10)
        
    def test_choose_action_with_legal_moves(self):
        """
            This method tests the choose_action method of the Agent class when there are legal moves available.
            It checks if the move chosen by the agent is in the list of legal moves.

            Raises:
                AssertionError: If the move chosen by the agent is not in the list of legal moves.
        """
        chosen_move = self.agent.choose_action(self.environ_state)
        self.assertIn(chosen_move, self.environ_state['legal_moves'])

    def test_choose_action_with_no_legal_moves(self):
        """
            This method tests the choose_action method of the Agent class when there are no legal moves available.
            It checks if the agent correctly returns an empty string when there are no legal moves available.

            Raises:
                AssertionError: If the agent does not return an empty string when there are no legal moves available.
        """
        self.environ_state['legal_moves'] = []
        chosen_move = self.agent.choose_action(self.environ_state)
        self.assertEqual(chosen_move, '')

    def test_choose_action_updates_Q_table_with_new_moves(self):
        """
            This method tests the choose_action method of the Agent class when it encounters new legal moves not in the Q-table.
            It checks if the agent correctly updates the Q-table with the new moves.

            Raises:
                AssertionError: If the agent does not update the Q-table with the new moves.
        """
        new_legal_moves = ['Nk3', 'Nk3']
        self.environ_state['legal_moves'] = new_legal_moves
        with patch.object(self.agent, 'update_Q_table', wraps=self.agent.update_Q_table) as mock_update:
            self.agent.choose_action(self.environ_state)
            mock_update.assert_called_once_with(new_legal_moves)
    
    def test_choose_action_trained_vs_untrained(self):
        """
            This method tests the choose_action method of the Agent class when the agent is trained versus when it is not trained.
            It checks if the agent's behavior changes based on whether it is trained or not. Specifically, it checks if the move chosen by the agent when it is trained is different from the move chosen when it is not trained.

            Raises:
                AssertionError: If the move chosen by the agent when it is trained is the same as the move chosen when it is not trained.
        """
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
        """
            This method tests the change_Q_table_pts method of the Agent class.
            It checks if the agent correctly updates the Q-table with the specified points for a given move and turn.

            Raises:
                AssertionError: If the updated value in the Q-table is not equal to the initial value plus the specified points.
        """
        # Test adding points to a Q-table cell
        move = 'e4'
        turn = 'W1'
        points = 10
        initial_value = self.agent.Q_table.at[move, turn]
        self.agent.change_Q_table_pts(move, turn, points)
        updated_value = self.agent.Q_table.at[move, turn]
        self.assertEqual(updated_value, initial_value + points)

    def test_invalid_move_handling_in_change_Q_table_pts(self):
        """
            This method tests the handling of invalid moves in the change_Q_table_pts method of the Agent class.
            It checks if the agent correctly raises a KeyError when an invalid move is attempted to be changed in the Q-table.

            Raises:
                AssertionError: If the agent does not raise a KeyError when an invalid move is attempted to be changed in the Q-table.
        """
        # Test handling of invalid moves in Q-table methods
        with self.assertRaises(KeyError):
            self.agent.change_Q_table_pts('invalid_move', 'W1', 10)

    def test_update_Q_table(self):
        """
            This method tests the update_Q_table method of the Agent class.
            It checks if the agent correctly updates the Q-table with new moves and if the initial Q-values for these new moves are all zero.

            Raises:
                AssertionError: If the new moves are not in the Q-table or if the initial Q-values for these new moves are not all zero.
        """
        new_moves = ['gk4', 'hk4']
        self.agent.update_Q_table(new_moves)
        for move in new_moves:
            self.assertTrue(move in self.agent.Q_table.index)
            self.assertTrue((self.agent.Q_table.loc[move] == 0).all())
                
    
if __name__ == '__main__':
    unittest.main()
