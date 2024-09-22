<agents/Agent.py> 

import game_settings
import pandas as pd
import numpy as np
import helper_methods
import logging
from typing import Union
from typing import Dict
from typing import List
import custom_exceptions
from utils.logging_config import setup_logger
from typing import Optional

agent_logger = setup_logger(__name__, game_settings.agent_errors_filepath)

class Agent:
    def __init__(self, color: str, learn_rate: float = 0.6, discount_factor: float = 0.35, q_table: Optional[pd.DataFrame] = None):
        try:
            self.color = color
            self.learn_rate = learn_rate
            self.discount_factor = discount_factor
            self.is_trained: bool = False
            self.q_table = q_table if q_table is not None else pd.DataFrame()
        except Exception as e:
            agent_logger.error(f'at __init__: failed to initialize agent. Error: {e}\n', exc_info=True)
            raise custom_exceptions.AgentInitializationError(f'failed to initialize agent due to error: {e}') from e
    ### end of __init__ ###

    def choose_action(self, chess_data, environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        if chess_data is None:
            chess_data = {}

        if environ_state['legal_moves'] == []:
            agent_logger.info(f'Agent.choose_action: legal_moves is empty. curr_game: {curr_game}, curr_turn: {environ_state['curr_turn']}\n')
            return ''
        
        try:
            self.update_q_table(environ_state['legal_moves']) # this func also checks if there are any new unique move strings
        except Exception as e:
            error_message = f'Failed to update Q-table. curr_game: {curr_game}, curr_turn: {environ_state["curr_turn"]} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e

        try:
            if self.is_trained:
                return self.policy_game_mode(environ_state['legal_moves'], environ_state['curr_turn'])
            else:
                return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
        except Exception as e:
            error_message = f'Failed to choose action. curr_game: {curr_game}, curr_turn: {environ_state["curr_turn"]} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e
    ### end of choose_action ###
    
    def policy_training_mode(self, chess_data, curr_game: str, curr_turn: str) -> str:
        try:
            chess_move = chess_data.at[curr_game, curr_turn]
            return chess_move
        except Exception as e:
            error_message = f'Failed to choose action at policy_training_mode. curr_game: {curr_game}, curr_turn: {curr_turn} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e
    ### end of policy_training_mode ###

    def policy_game_mode(self, legal_moves: list[str], curr_turn: str) -> str:
        dice_roll = helper_methods.get_number_with_probability(game_settings.chance_for_random_move)
        
        try:
            legal_moves_in_q_table = self.q_table[curr_turn].loc[self.q_table[curr_turn].index.intersection(legal_moves)]
        except Exception as e:
            error_message = f'at policy_game_mode: legal moves not found in q_table or legal_moves is empty. curr_turn: {curr_turn} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e

        if dice_roll == 1:
            chess_move = legal_moves_in_q_table.sample().index[0]
        else:
            chess_move = legal_moves_in_q_table.idxmax()
        return chess_move
    ### end of policy_game_mode ###

    def change_q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        try:    
            self.q_table.at[chess_move, curr_turn] += pts
        except Exception as e:
            error_message = f'@ change_q_table_pts(). Failed to change q_table points. chess_move: {chess_move}, curr_turn: {curr_turn}, pts: {pts} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e
    ### end of change_q_table_pts ###

    def update_q_table(self, new_chess_moves: Union[str, list[str]]) -> None:
        if isinstance(new_chess_moves, str):
            new_chess_moves = [new_chess_moves]
        
        # Convert to set for efficient lookup
        new_moves_set = set(new_chess_moves)

        # Check if all moves are already in the Q-table
        existing_moves = set(self.q_table.index)
        truly_new_moves = new_moves_set - existing_moves

        # If no new moves, return early
        if not truly_new_moves:
            return

        try:
            q_table_new_values: pd.DataFrame = pd.DataFrame(
                0, 
                index = list(truly_new_moves), 
                columns = self.q_table.columns, 
                dtype = np.int64
            )

            self.q_table = pd.concat([self.q_table, q_table_new_values])
        except Exception as e:
            error_message = f'@ update_q_table(). Failed to update q_table. new_chess_moves: {new_chess_moves}, dur to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e
    ### update_q_table ###

<end of agents/Agent.py> 

<environment/Environ.py>

import game_settings
import chess
import logging
import custom_exceptions
from utils.logging_config import setup_logger 
from typing import Dict 
from typing import Union

environ_logger = setup_logger(__name__, game_settings.environ_errors_filepath)


class Environ:
    def __init__(self):
        try: 
            self.board: chess.Board = chess.Board()
            
            # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
            max_turns = game_settings.max_num_turns_per_player * 2 # 2 players
            self.turn_list: list[str] = [f'{"W" if i % 2 == 0 else "B"}{i // 2 + 1}' for i in range(max_turns)]
            self.turn_index: int = 0
        except Exception as e:
            environ_logger.error(f'at __init__: failed to initialize environ. Error: {e}\n', exc_info=True)
            raise custom_exceptions.EnvironInitializationError(f'failed to initialize environ due to error: {e}') from e
    ### end of constructor

    def get_curr_state(self) -> Dict[str, Union[int, str, list[str]]]:
        if not (0 <= self.turn_index < len(self.turn_list)):
            environ_logger.error(f'ERROR: Turn index out of range: {self.turn_index}\n')
            raise custom_exceptions.TurnIndexError(f'Turn index out of range: {self.turn_index}')
    
        try:
            curr_turn = self.get_curr_turn()
            legal_moves = self.get_legal_moves()
        except Exception as e:
            error_message = f'An error occurred at get_curr_state or get_legal_moves due to error: {str(e)}'
            environ_logger.error()
            raise custom_exceptions.StateRetrievalError(error_message) from e
        
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    ### end of get_curr_state
    
    def update_curr_state(self) -> None:
        if self.turn_index >= game_settings.max_turn_index:
            environ_logger.error(f'ERROR: max_turn_index reached: {self.turn_index} >= {game_settings.max_turn_index}\n')
            raise IndexError(f"Maximum turn index ({game_settings.max_turn_index}) reached!")
    
        if self.turn_index >= len(self.turn_list):
            environ_logger.error(f'ERROR: turn index out of bounds: {self.turn_index} >= {len(self.turn_list)}\n')
            raise IndexError(f"Turn index out of bounds: {self.turn_index}")
    
        self.turn_index += 1
    ### end of update_curr_state
    
    def get_curr_turn(self) -> str:                        
        if not (0 <= self.turn_index < len(self.turn_list)):
            environ_logger.error(f'ERROR: Turn index out of range: {self.turn_index}\n')
            raise custom_exceptions.TurnIndexError(f'Turn index out of range: {self.turn_index}')
        
        return self.turn_list[self.turn_index]
        ### end of get_curr_turn
    
    def load_chessboard(self, chess_move: str, curr_game = 'Game 1') -> None:
        try:
            self.board.push_san(chess_move)
        except Exception as e:
            error_message = f'An error occurred at load_chessboard: {str(e)}, unable to load chessboard with {chess_move} in {curr_game}'
            environ_logger.error(error_message)
            raise custom_exceptions.InvalidMoveError(error_message) from e
    ### end of load_chessboard    

    def pop_chessboard(self) -> None:
        try:
            self.board.pop()
        except Exception as e:
            error_message = f'An error occurred at pop_chessboard. unable to pop chessboard, due to error: {str(e)}'
            environ_logger.error(error_message)
            raise custom_exceptions.ChessboardPopError(error_message) from e
    ### end of pop_chessboard

    def undo_move(self) -> None:
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
    
    def get_legal_moves(self) -> list[str]:   
        try:
            return [self.board.san(move) for move in self.board.legal_moves]
        except Exception as e:
            error_message = f'An error occurred at get_legal_moves: {str(e)}, legal moves could not be retrieved, at turn index: {self.turn_index}, current turn: {self.get_curr_turn()}, current board state: {self.board}, current legal moves: {self.board.legal_moves}'
            environ_logger.error(error_message)
            raise custom_exceptions.NoLegalMovesError(error_message) from e
    ### end of get_legal_moves
    
<end of environment/Environ.py>

<main/agent_vs_agent.py>

import helper_methods
import game_settings
import time
import Environ
import Agent
import logging
import custom_exceptions
from utils.logging_config import setup_logger 

agent_vs_agent_logger = setup_logger(__name__, game_settings.agent_vs_agent_logger_filepath)

def agent_vs_agent(environ, w_agent, b_agent, print_to_screen = False, current_game = 0) -> None:
    try:
        # play all moves in a single game
        print(f'Playing game {current_game}\n')
        while helper_methods.is_game_over(environ) == False:
            if print_to_screen:
                print(f'\nCurrent turn: {environ.get_curr_turn()}')
                chess_move = helper_methods.agent_selects_and_plays_chess_move(w_agent, environ)
                time.sleep(3)
                print(f'White agent played {chess_move}')
            else:
                agent_vs_agent_logger.info(f'Current turn is: {environ.get_current_turn()}. \nWhite agent played {chess_move}\n')
            
            # sometimes game ends after white's turn
            if helper_methods.is_game_over(environ) == False:
                if print_to_screen:
                    chess_move = helper_methods.agent_selects_and_plays_chess_move(b_agent, environ)
                    time.sleep(3)
                    print(f'Black agent played {chess_move} curr board is:\n{environ.board}\n')
                else:
                    agent_vs_agent_logger.info(f'Black agent played {chess_move} curr board is:\n{environ.board}\n')
    except Exception as e:
        error_message = f'An error occurred at agent_vs_agent: {e}'
        agent_vs_agent_logger.error(error_message)
        raise custom_exceptions.GamePlayError(error_message) from e

    # game is over, reset environ
    agent_vs_agent_logger.info('Game is over\n')
    agent_vs_agent_logger.info(f'Final board is:\n{environ.board}\n')
    agent_vs_agent_logger.info(f'game result is: {environ.get_game_result()}\n')
    environ.reset_environ()
### end of agent_vs_agent

if __name__ == '__main__':
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    try:
        bradley = helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
        imman = helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)

        # ask user to input number of games to play
        number_of_games = int(input('How many games do you want the agents to play? '))
        print_to_screen = (input('Do you want to print the games to the screen? (y/n) ')).lower()[0]

        # while there are games still to play, call agent_vs_agent
        for current_game in range(int(number_of_games)):
            if print_to_screen == 'y':
                print(f'Game {current_game + 1}')

            agent_vs_agent(environ, bradley, imman, print_to_screen, current_game)

    except Exception as e:
        print(f'agent vs agent interrupted because of:  {e}')
        agent_vs_agent_logger.error(f'An error occurred: {e}\n')        
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print('single agent vs agent game is complete')
    print(f'it took: {total_time}')
    quit()

<end of main/agent_vs_agent.py>

<main/agent_vs_human.py>

import helper_methods
import game_settings
import time
import Environ
import Agent
import logging
import custom_exceptions
from utils.logging_config import setup_logger

agent_vs_human_logger = setup_logger(__name__, game_settings.agent_vs_human_logger_filepath)


def play_game_vs_human(environ: Environ.Environ, chess_agent: Agent.Agent) -> None:
    player_turn = 'W'
    try:
        while helper_methods.is_game_over(environ) == False:
            print(f'\nCurrent turn is :  {environ.get_curr_turn()}\n')
            chess_move = handle_move(player_turn, chess_agent)
            print(f'{player_turn} played {chess_move}\n')
            player_turn = 'B' if player_turn == 'W' else 'W'

        print(f'Game is over, result is: {helper_methods.get_game_outcome(environ)}')
        print(f'The game ended because of: {helper_methods.get_game_termination_reason(environ)}')
    except Exception as e:
        print(f'An error occurred at play_game_vs_human: {e}')
        error_message = f'An error occurred at play_game_vs_human: {str(e)}'
        agent_vs_human_logger(error_message, exc_info=True)
        raise custom_exceptions.GamePlayError(error_message) from e
    
    finally:
        environ.reset_environ()
### end of play_game

def handle_move(player_color: str, chess_agent: Agent.Agent) -> str:
    if player_color == chess_agent.color:
        print('=== RL AGENT\'S TURN ===\n')
        chess_move = helper_methods.agent_selects_and_plays_chess_move(chess_agent, environ)
    else:
        print('=== OPPONENT\'S TURN ===')
        chess_move = input('Enter chess move: ')
        
        try:
            while not helper_methods.receive_opponent_move(chess_move, environ):
                print('Invalid move, try again.')
                chess_move = input('Enter chess move: ')
            return chess_move
        except Exception as e:
            error_message = f'An error occurred at handle_move: {e}'
            agent_vs_human_logger.error(error_message)
            raise custom_exceptions.GamePlayError(error_message) from e
### end of handle_move

if __name__ == '__main__':    
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    try:
        bradley = helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
        imman = helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)
        
        rl_agent_color = input('Enter color for agent to be , \'W\' or \'B\': ')
    
        if rl_agent_color == 'W':
            play_game_vs_human(environ, bradley)
        else: 
            play_game_vs_human(environ, imman)
        
    except Exception as e:
        print(f'agent vs human interrupted because of:  {e}')
        agent_vs_human_logger.error(f'An error occurred: {e}\n')
        quit()

    end_time = time.time()
    total_time = end_time - start_time
    print('agent vs human game is complete')
    print(f'it took: {total_time}')
    quit() 

<end of main/agent_vs_human.py>

<main/continue_training_agents.py>

import helper_methods
import game_settings
import time
import training_functions
import Agent
import Environ
from utils.logging_config import setup_logger

agent_vs_agent_logger = setup_logger(__name__, game_settings.agent_vs_agent_logger_filepath)

if __name__ == '__main__':
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')
    
    helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)
    num_games_to_play = game_settings.agent_vs_agent_num_games

    try:
        training_functions.continue_training_rl_agents(num_games_to_play, bradley, imman, environ)
        helper_methods.pikl_q_table(bradley, game_settings.bradley_agent_q_table_path)
        helper_methods.pikl_q_table(imman, game_settings.imman_agent_q_table_path)

    except Exception as e:
        print(f'training interrupted because of:  {e}')
        quit()
        
    end_time = time.time()
    total_time = end_time - start_time

    print('agent v agent training round is complete')
    print(f'it took: {total_time}')
    quit() 

<end of main/continue_training_agents.py> 

<main/train_new_agents.py>

import helper_methods
import game_settings
import pandas as pd
import time
import training_functions
import Agent
import chess
from utils.logging_config import setup_logger

train_new_agents_logger = setup_logger(__name__, game_settings.train_new_agents_logger_filepath)

if __name__ == '__main__':
    start_time = time.time()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    # !!!!!!!!!!!!!!!!! change this each time for new section of the database  !!!!!!!!!!!!!!!!!
    estimated_q_values_table = pd.read_pickle(game_settings.est_q_vals_filepath_part_1, compression = 'zip')
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression = 'zip')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    try:
        training_functions.train_rl_agents(chess_data, estimated_q_values_table, bradley, imman)
    except Exception as e:
        print(f'training interrupted because of:  {e}')
        quit()
        
    end_time = time.time()
    helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time} for {game_settings.training_sample_size} games\n')
    quit() 

<end of main/train_new_agents.py>

<training/training_functions.py>

from utils import helper_methods
import chess
from utils import game_settings
from environment import Environ
import pandas as pd
import copy
from utils import custom_exceptions
import re
from utils.logging_config import setup_logger

training_functions_logger = setup_logger(__name__, game_settings.training_functions_logger_filepath)

def train_rl_agents(chess_data, est_q_val_table, w_agent, b_agent):
    ### FOR EACH GAME IN THE TRAINING SET ###
    for game_number in chess_data.index:
        w_curr_q_value: int = game_settings.initial_q_val
        b_curr_q_value: int = game_settings.initial_q_val

        try: 
            train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at train_one_game: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            raise Exception from e
    
    w_agent.is_trained = True
    b_agent.is_trained = True
    return w_agent, b_agent
### end of train_rl_agents

def train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value) -> None:
    num_chess_moves_curr_training_game: int = chess_data.at[game_number, 'PlyCount']
    environ = Environ.Environ()

    try:
        curr_state = environ.get_curr_state()
    except Exception as e:
        training_functions_logger.error(f'An error occurred environ.get_curr_state: {e}\n')
        training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
        training_functions_logger.error(f'at game: {game_number}\n')
        training_functions_logger.error(f'at turn: {curr_state['turn_index']}')
        raise Exception from e

    ### THIS WHILE LOOP PLAYS THROUGH ONE GAME ###
    while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
        ##################### WHITE'S TURN ####################
        # choose action a from state s, using policy
        try:
            w_chess_move = w_agent.choose_action(chess_data, curr_state, game_number)
        except Exception as e:
            training_functions_logger.error(f'Hi from train_one_game. An error occurred at w_agent.choose_action: {e}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            raise Exception from e

        if not w_chess_move:
            training_functions_logger.error(f'An error occurred at w_agent.choose_action\n')
            training_functions_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
            raise Exception("w_chess_move is empty")

        ### ASSIGN POINTS TO q TABLE FOR WHITE AGENT ###
        # on the first turn for white, this would assign to W1 col at chess_move row.
        # on W's second turn, this would be q_next which is calculated on the first loop.                
        assign_points_to_q_table(w_chess_move, curr_state['curr_turn'], w_curr_q_value, w_agent)

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ### WHITE AGENT PLAYS THE SELECTED MOVE ###
        # take action a, observe r, s', and load chessboard
        try:
            apply_move_and_update_state(w_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at apply_move_and_update_state: {e}\n')
            training_functions_logger.error(f'at game_number: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        try:
            w_reward = get_reward(w_chess_move)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at get_reward for white: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # get latest curr_state since apply_move_and_update_state updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            training_functions_logger.error(f'An error occurred at get_curr_state: {e}\n')
            training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
            training_functions_logger.error(f'At game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # check if game ended
        try: 
            if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                break # game is over, exit function.

            else: # current game continues
                # the var curr_turn_for_q_values is here because we previously moved to next turn (after move was played)
                # but we want to assign the q est based on turn just before the curr turn was incremented.
                w_est_q_value: int = est_q_val_table.at[game_number, curr_turn_for_q_est]
        except Exception as e:
            training_functions_logger.error(f'error when determining if game ended after white\'s move: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        ##################### BLACK'S TURN ####################
        # choose action a from state s, using policy
        try: 
            b_chess_move = b_agent.choose_action(chess_data, curr_state, game_number)
        except Exception as e:
            training_functions_logger.error(f'Hi from train_one_game. An error occurred at b_agent.choose_action: {e}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            raise Exception from e

        if not b_chess_move:
            training_functions_logger.error(f'An error occurred at w_agent.choose_action\n')
            training_functions_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            raise Exception from e

        # assign points to q table
        assign_points_to_q_table(b_chess_move, curr_state['curr_turn'], b_curr_q_value, b_agent)

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ##### BLACK AGENT PLAYS SELECTED MOVE #####
        # take action a, observe r, s', and load chessboard
        try:
            apply_move_and_update_state(b_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at apply_move_and_update_state: {e}\n')
            training_functions_logger.error(f'at game_number: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        try:
            b_reward = get_reward(b_chess_move)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at get_reward for black: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # get latest curr_state since apply_move_and_update_state updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            training_functions_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
            training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
            training_functions_logger.error(f'At game: {game_number}\n')
            raise Exception from e

        # find the estimated q value for Black, but first check if game ended
        try: 
            if environ.board.is_game_over() or not curr_state['legal_moves']:
                break # game is over, exit function
            else: # current game continues
                b_est_q_value: int = est_q_val_table.at[game_number, curr_turn_for_q_est]
        except Exception as e:
            training_functions_logger.error(f'error when determining if game ended after black\'s move: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        training_functions_logger.info(f'b_est_q_value: {b_est_q_value}\n')
        training_functions_logger.info(f'about to calc next q values\n')
        training_functions_logger.info(f'w_curr_q_value: {w_curr_q_value}\n')
        training_functions_logger.info(f'b_curr_q_value: {b_curr_q_value}\n')
        training_functions_logger.info(f'w_reward: {w_reward}\n')
        training_functions_logger.info(f'b_reward: {b_reward}\n')
        training_functions_logger.info(f'w_est_q_value: {w_est_q_value}\n')
        training_functions_logger.info(f'b_est_q_value: {b_est_q_value}\n\n')

        # ***CRITICAL STEP***, this is the main part of the SARSA algorithm.
        try:
            w_next_q_value: int = find_next_q_value(w_curr_q_value, w_agent.learn_rate, w_reward, w_agent.discount_factor, w_est_q_value)
            b_next_q_value: int = find_next_q_value(b_curr_q_value, b_agent.learn_rate, b_reward, b_agent.discount_factor, b_est_q_value)
        except Exception as e:
            training_functions_logger.error(f'An error occurred while calculating next q values: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            raise Exception from e

        training_functions_logger.info(f'sarsa calc complete\n')
        training_functions_logger.info(f'w_next_q_value: {w_next_q_value}\n')
        training_functions_logger.info(f'b_next_q_value: {b_next_q_value}\n')

        # on the next turn, w_next_q_value and b_next_q_value will be added to the q table. so if this is the end of the first round,
        # next round it will be W2 and then we assign the q value at W2 col
        w_curr_q_value = w_next_q_value
        b_curr_q_value = b_next_q_value

        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            training_functions_logger.error(f'An error occurred just after finding qnext: {e}\n')
            training_functions_logger.error("failed to get_curr_state\n") 
            training_functions_logger.error(f'At game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e
    ### END OF CURRENT GAME LOOP ###

    training_functions_logger.info(f'{game_number} is over.\n')
    training_functions_logger.info(f'\nThe Chessboard looks like this:\n')
    training_functions_logger.info(f'\n{environ.board}\n\n')
    training_functions_logger.info(f'Game result is: {helper_methods.get_game_outcome(environ)}\n')    
    training_functions_logger.info(f'The game ended because of: {helper_methods.get_game_termination_reason()}\n')
    training_functions_logger.info(f'DB shows game ended b/c: {chess_data.at[game_number, "Result"]}\n')

    environ.reset_environ()
### end of train_one_game

def generate_q_est_df(chess_data, w_agent, b_agent) -> pd.DataFrame:
    estimated_q_values = chess_data.copy(deep = True)
    estimated_q_values = estimated_q_values.astype('int64')
    estimated_q_values.iloc[:, 1:] = 0

    for game_number in chess_data.index:
        try: 
            generate_q_est_df_one_game(chess_data, game_number, w_agent, b_agent)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at generate_q_est_df_one_game: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            raise Exception from e

    return estimated_q_values
# end of generate_q_est_df

def generate_q_est_df_one_game(chess_data, game_number, w_agent, b_agent) -> None:
    num_chess_moves_curr_training_game: int = chess_data.at[game_number, 'PlyCount']
    environ = Environ.Environ()
    engine = start_chess_engine()
    
    try:
        curr_state = environ.get_curr_state()
    except Exception as e:
        training_functions_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
        training_functions_logger.error(f'at: {game_number}\n')
        raise Exception from e
    
    ### LOOP PLAYS THROUGH ONE GAME ###
    while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
        ##################### WHITE'S TURN ####################
        # choose action a from state s, using policy

        try:
            w_chess_move = w_agent.choose_action(chess_data, curr_state, game_number)
        except Exception as e:
            training_functions_logger.error(f'Hi from train_one_game. An error occurred at w_agent.choose_action: {e}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            raise Exception from e

        if not w_chess_move:
            training_functions_logger.error(f'An error occurred at w_agent.choose_action\n')
            training_functions_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
            raise Exception("w_chess_move is empty")

        # assign curr turn to new var for now. once agent plays move, turn will be updated, but we need 
        # to track the turn before so that the est q value can be assigned to the correct column.
        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ### WHITE AGENT PLAYS THE SELECTED MOVE ###
        # take action a, observe r, s', and load chessboard
        try:
            apply_move_and_update_state(w_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at apply_move_and_update_state: {e}\n')
            training_functions_logger.error(f'at game_number: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # get latest curr_state since apply_move_and_update_state updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            training_functions_logger.error(f'hi from generate_q_est_df_one_game. An error occurred at get_curr_state: {e}\n')
            training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
            training_functions_logger.error(f'At game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e
        
        # check if game ended
        try: 
            if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                break # game is over, exit function.

            else: # current game continues
                try: 
                    w_est_q_value: int = find_estimated_q_value(environ, engine)
                except Exception as e:
                    training_functions_logger.error(f"at White turn, failed to find_estimated_q_valueue because error: {e}\n")
                    training_functions_logger.error(f'curr state is :{curr_state}\n')
                    training_functions_logger.error(f'at : {game_number}\n')
                    raise Exception from e
        except Exception as e:
            training_functions_logger.error(f'error when determining if game ended after white\'s move: {e}\n')
            training_functions_logger.error(f'could also be that the find_estimated_q_value func failed')
            training_functions_logger.error(f'at game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        ##################### BLACK'S TURN ####################
        # choose action a from state s, using policy
        try:
            b_chess_move = b_agent.choose_action(chess_data, curr_state, game_number)
        except Exception as e:
            training_functions_logger.error(f'Hi from train_one_game. An error occurred at b_agent.choose_action: {e}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            raise Exception from e

        if not b_chess_move:
            training_functions_logger.error(f'An error occurred at b_agent.choose_action\n')
            training_functions_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            raise Exception("b_chess_move is empty")

        # assign curr turn to new var for now. once agent plays move, turn will be updated, but we need 
        # to track the turn before so that the est q value can be assigned to the correct column.
        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])
        
        ##### BLACK AGENT PLAYS SELECTED MOVE #####
        # take action a, observe r, s', and load chessboard
        try:
            apply_move_and_update_state(b_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at apply_move_and_update_state: {e}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # get latest curr_state since apply_move_and_update_state updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            training_functions_logger.error(f'hi from generate_q_est_df_one_game. An error occurred at get_curr_state: {e}\n')
            training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
            training_functions_logger.error(f'At game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # check if game ended
        try: 
            if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                break # game is over, exit function.
            else: # current game continues
                try: 
                    b_est_q_value: int = find_estimated_q_value(environ, engine)
                except Exception as e:
                    training_functions_logger.error(f"at Black's turn, failed to find_estimated_q_valueue because error: {e}\n")
                    training_functions_logger.error(f'curr state is :{curr_state}\n')
                    training_functions_logger.error(f'at : {game_number}\n')
                    raise Exception from e
        except Exception as e:
            training_functions_logger.error(f'error when determining if game ended after black\'s move: {e}\n')
            training_functions_logger.error(f'could also be that the find_estimated_q_value func failed')
            training_functions_logger.error(f'at game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            training_functions_logger.error(f'hi from generate_q_est_df_one_game. An error occurred at get_curr_state: {e}\n')
            training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
            training_functions_logger.error(f'At game: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e
    ### END OF CURRENT GAME LOOP ###
    environ.reset_environ()
    engine.quit()
# end of generate_q_est_df_one_game

def find_estimated_q_value(environ, engine) -> int:
    # RL agent just played a move. the board has changed, if stockfish analyzes the board, 
    # it will give points for the agent, based on the agent's latest move.
    # We also need the points for the ANTICIPATED next state, 
    # given the ACTICIPATED next action. In this case, the anticipated response from opposing agent.
    try:
        anticipated_next_move = analyze_board_state(environ.board, engine)
    except custom_exceptions.BoardAnalysisError as e:
        training_functions_logger.error(f'at find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error(f'failed to analyze_board_state\n')
        raise Exception from e
    
    # load up the chess board with opponent's anticipated chess move 
    try:
        environ.load_chessboard_for_q_est(anticipated_next_move)
    except custom_exceptions.ChessboardLoadError as e:
        training_functions_logger.error(f'at find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error(f'failed to load_chessboard_for_q_est\n')
        raise Exception from e
    
    # check if the game would be over with the anticipated next move, like unstopable checkmate.
    if environ.board.is_game_over() or not environ.get_legal_moves():
        try:
            environ.pop_chessboard()
        except custom_exceptions.ChessboardPopError as e:
            training_functions_logger.error(f'at find_estimated_q_value. An error occurred: {e}\n')
            training_functions_logger.error(f'failed at environ.pop_chessboard\n')
            raise Exception from e

    # this is the q estimated value due to what the opposing agent is likely to play in response to our move.    
    try:
        est_qval_analysis = analyze_board_state(environ.board, engine)
    except custom_exceptions.QValueEstimationError as e:
        training_functions_logger.error(f'at find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error(f'failed at analyze_board_state\n')
        raise Exception from e

    # get pts for est_qval 
    if est_qval_analysis['mate_score'] is None:
        est_qval = est_qval_analysis['centipawn_score']
    else: # there is an impending checkmate
        est_qval = game_settings.CHESS_MOVE_VALUES['mate_score']

    # IMPORTANT STEP, pop the chessboard of last move, we are estimating board states, not
    # playing a move.
    try:
        environ.pop_chessboard()
    except Exception as e:
        training_functions_logger.error(f'@ find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error("failed to pop_chessboard after est q val analysis values found\n")
        raise Exception from e

    return est_qval
# end of find_estimated_q_value

def find_next_q_value(curr_qval: int, learn_rate: float, reward: int, discount_factor: float, est_qval: int) -> int:
    try:
        next_qval = int(curr_qval + learn_rate * (reward + ((discount_factor * est_qval) - curr_qval)))
        return next_qval
    except OverflowError:
        training_functions_logger.error(f'@ find_next_q_value. An error occurred: OverflowError\n')
        training_functions_logger.error(f'curr_qval: {curr_qval}\n')
        training_functions_logger.error(f'learn_rate: {learn_rate}\n')
        training_functions_logger.error(f'reward: {reward}\n')
        training_functions_logger.error(f'discount_factor: {discount_factor}\n')
        training_functions_logger.error(f'est_qval: {est_qval}\n')
        raise custom_exceptions.QValueCalculationError("Overflow occurred during q-value calculation") from OverflowError
# end of find_next_q_value

def analyze_board_state(board, engine) -> dict:
    if not board.is_valid():
        training_functions_logger.error(f'at analyze_board_state. Board is in invalid state\n')
        raise custom_exceptions.InvalidBoardStateError(f'at analyze_board_state. Board is in invalid state\n')

    try: 
        analysis_result = engine.analyse(board, game_settings.search_limit, multipv = game_settings.num_moves_to_return)
    except Exception as e:
        training_functions_logger.error(f'@ Bradley_analyze_board_state. An error occurred during analysis: {e}\n')
        training_functions_logger.error(f"Chessboard is:\n{board}\n")
        raise custom_exceptions.EngineAnalysisError("error occured during stockfish analysis") from e

    mate_score = None
    centipawn_score = None
    anticipated_next_move = None

    try:
        # Get score from analysis_result and normalize for player perspective
        pov_score = analysis_result[0]['score'].white() if board.turn == chess.WHITE else analysis_result[0]['score'].black()

        if pov_score.is_mate():
            mate_score = pov_score.mate()
        else:
            centipawn_score = pov_score.score()
    except Exception as e:
        training_functions_logger.error(f'An error occurred while extracting scores: {e}\n')
        raise custom_exceptions.ScoreExtractionError("Error occurred while extracting scores from analysis") from e

    try:
        anticipated_next_move = analysis_result[0]['pv'][0]
    except Exception as e:
        training_functions_logger.error(f'An error occurred while extracting the anticipated next move: {e}\n')
        raise custom_exceptions.MoveExtractionError("Error occurred while extracting the anticipated next move") from e
    
    return {
        'mate_score': mate_score,
        'centipawn_score': centipawn_score,
        'anticipated_next_move': anticipated_next_move
    }
### end of analyze_board_state

def apply_move_and_update_state(chess_move: str, game_number, environ: Environ.Environ) -> None:
    try:
        environ.load_chessboard(chess_move, game_number)
    except custom_exceptions.ChessboardLoadError as e:
        training_functions_logger.error(f'at apply_move_and_update_state. An error occurred at {game_number}: {e}\n')
        training_functions_logger.error(f"failed to load_chessboard with move {chess_move}\n")
        raise Exception from e

    try:
        environ.update_curr_state()
    except custom_exceptions.StateUpdateError as e:
        training_functions_logger.error(f'at apply_move_and_update_state. update_curr_state() failed to increment turn_index, Caught exception: {e}\n')
        training_functions_logger.error(f'Current state is: {environ.get_curr_state()}\n')
        raise Exception from e
# end of apply_move_and_update_state

def get_reward(chess_move: str) -> int:
    if not chess_move or not isinstance(chess_move, str):
        raise custom_exceptions.RewardCalculationError("Invalid chess move input")

    total_reward = 0
    # Check for piece development (N, R, B, Q)
    if re.search(r'[NRBQ]', chess_move):
        total_reward += game_settings.CHESS_MOVE_VALUES['piece_development']
    # Check for capture
    if 'x' in chess_move:
        total_reward += game_settings.CHESS_MOVE_VALUES['capture']
    # Check for promotion (with additional reward for queen promotion)
    if '=' in chess_move:
        total_reward += game_settings.CHESS_MOVE_VALUES['promotion']
        if '=Q' in chess_move:
            total_reward += game_settings.CHESS_MOVE_VALUES['promotion_queen']
    return total_reward
## end of get_reward

def start_chess_engine(): 
    try:
        chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
        return chess_engine
    except custom_exceptions.EngineStartError as e:
        training_functions_logger.error(f'An error occurred at start_chess_engine: {e}\n')
        raise Exception from e
# end of start_chess_engine

def assign_points_to_q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
    try:
        chess_agent.update_q_table([chess_move])
        chess_agent.change_q_table_pts(chess_move, curr_turn, curr_q_val)
    except custom_exceptions.QTableUpdateError as e: 
        training_functions_logger.error(f'caught exception: {e} at assign_points_to_q_table\n')
        training_functions_logger.error(f'chess_move: {chess_move}\n')
        training_functions_logger.error(f'curr_turn: {curr_turn}\n')
        training_functions_logger.error(f'curr_q_val: {curr_q_val}\n')
        training_functions_logger.error(f'chess_agent: {chess_agent}\n')
        raise Exception from e
# enf of assign_points_to_q_table 

<end of training/training_functions.py> 

<utils/custom_exceptions.py>

# custom_exceptions.py

class ChessError(Exception):
    """Base class for exceptions in the chess application."""
    def __init__(self, message="An error occurred in the chess application"):
        self.message = message
        super().__init__(self.message)

class ChessboardError(ChessError):
    """Base class for exceptions related to the chessboard."""
    pass

class ChessboardLoadError(ChessboardError):
    """Exception raised when there's an error loading a move onto the chessboard."""
    def __init__(self, move, message=None):
        self.move = move
        self.message = message or f"Error loading move '{move}' onto chessboard"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move})"

class ChessboardPopError(ChessboardError):
    """Exception raised when there's an error removing a move from the chessboard."""
    def __init__(self, move, message=None):
        self.move = move
        self.message = message or f"Error removing {move} from chessboard"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move})"

class ChessboardManipulationError(ChessboardError):
    """Exception raised when there's an error manipulating the chessboard."""
    def __init__(self, action, message=None):
        self.action = action
        self.message = message or f"Error during chessboard manipulation: {action}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Action: {self.action})"

class StateError(ChessError):
    """Base class for exceptions related to the game state."""
    pass

class StateUpdateError(StateError):
    """Exception raised when there's an error updating the current state of the environment."""
    def __init__(self, current_state, message=None):
        self.current_state = current_state
        self.message = message or "Error updating current state"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Current State: {self.current_state})"

class StateRetrievalError(StateError):
    """Exception raised when there's an error retrieving the current state of the environment."""
    def __init__(self, message="Error retrieving current state"):
        super().__init__(message)

class IllegalMoveError(ChessError):
    """Exception raised when an illegal move is attempted."""
    def __init__(self, move, message=None):
        self.move = move
        self.message = message or f"Illegal move: {move}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move})"

class GameError(ChessError):
    """Base class for exceptions related to game flow."""
    pass

class GameOverError(GameError):
    """Exception raised when trying to make a move in a finished game."""
    def __init__(self, message="The game is already over"):
        super().__init__(message)

class NoLegalMovesError(GameError):
    """Exception raised when there are no legal moves available."""
    def __init__(self, message="No legal moves available"):
        super().__init__(message)

class GameOutcomeError(GameError):
    """Exception raised when the game outcome cannot be determined."""
    def __init__(self, message="Unable to determine game outcome"):
        super().__init__(message)

class GameTerminationError(GameError):
    """Exception raised when the game termination reason cannot be determined."""
    def __init__(self, message="Unable to determine game termination reason"):
        super().__init__(message)

class TrainingError(ChessError):
    """Exception raised for errors during the training process."""
    def __init__(self, stage, message=None):
        self.stage = stage
        self.message = message or f"An error occurred during the training process: {stage}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Training Stage: {self.stage})"

class QTableError(ChessError):
    """Base class for exceptions related to Q-table operations."""
    pass

class QTableUpdateError(QTableError):
    """Exception raised when there's an error updating the Q-table."""
    def __init__(self, move, turn, message=None):
        self.move = move
        self.turn = turn
        self.message = message or f"Error updating Q-table for move '{move}' at turn {turn}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move}, Turn: {self.turn})"

class QValueCalculationError(QTableError):
    """Exception raised when there's an error calculating the Q-value."""
    def __init__(self, params, message=None):
        self.params = params
        self.message = message or "Error calculating Q-value"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Parameters: {self.params})"

class AnalysisError(ChessError):
    """Base class for exceptions related to board analysis."""
    pass

class BoardAnalysisError(AnalysisError):
    """Exception raised when there's an error analyzing the board state."""
    def __init__(self, board_state, message=None):
        self.board_state = board_state
        self.message = message or "Error analyzing board state"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Board State: {self.board_state})"

class InvalidBoardStateError(AnalysisError):
    """Exception raised when the chess board is in an invalid state."""
    def __init__(self, board_state, message=None):
        self.board_state = board_state
        self.message = message or "Chess board is in an invalid state"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Board State: {self.board_state})"

class EngineAnalysisError(AnalysisError):
    """Exception raised when there's an error during engine analysis."""
    def __init__(self, engine, message=None):
        self.engine = engine
        self.message = message or f"Error occurred during chess engine analysis with {engine}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Engine: {self.engine})"

class ScoreExtractionError(AnalysisError):
    """Exception raised when there's an error extracting scores from analysis."""
    def __init__(self, analysis_result, message=None):
        self.analysis_result = analysis_result
        self.message = message or "Error extracting scores from analysis"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Analysis Result: {self.analysis_result})"

class MoveExtractionError(AnalysisError):
    """Exception raised when there's an error extracting the anticipated move from analysis."""
    def __init__(self, analysis_result, message=None):
        self.analysis_result = analysis_result
        self.message = message or "Error extracting anticipated move from analysis"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Analysis Result: {self.analysis_result})"



class AgentError(ChessError):
    """Base class for exceptions in the Agent class."""
    pass

class InvalidActionError(AgentError):
    """Exception raised when an invalid action is chosen by agent."""
    def __init__(self, action, message=None):
        self.action = action
        self.message = message or f"Invalid action chosen: {action}"
        super().__init__(self.message)

class QTableAccessError(AgentError):
    """Exception raised when there's an error accessing the Q-table."""
    pass

class FailureToChooseActionError(AgentError):
    """Exception raised when the agent fails to choose an action."""
    def __init__(self, message="Agent failed to choose an action"):
        super().__init__(message)

class AgentInitializationError(AgentError):
    """Exception raised when there's an error initializing the agent."""
    def __init__(self, message="Error initializing agent"):
        super().__init__(message)

class EnvironError(ChessError):
    """Base class for exceptions in the Environ class."""
    pass

class TurnIndexError(EnvironError):
    """Exception raised when there's an issue with the turn index."""
    pass

class InvalidMoveError(EnvironError):
    """Exception raised when an invalid move is attempted."""
    pass

class HelperMethodError(ChessError):
    """Base class for exceptions in helper methods."""
    pass

class EngineStartError(HelperMethodError):
    """Exception raised when there's an error starting the chess engine."""
    pass

class RewardCalculationError(HelperMethodError):
    """Exception raised when there's an error calculating rewards."""
    pass

class TrainingFunctionError(ChessError):
    """Base class for exceptions in training functions."""
    pass

class QValueEstimationError(TrainingFunctionError):
    """Exception raised when there's an error estimating Q-values."""
    pass

class GameSimulationError(TrainingFunctionError):
    """Exception raised when there's an error simulating games during training."""
    pass

class GamePlayError(ChessError):
    """Exception raised when there's an error playing games between agents and human v agents."""
    pass 

<end of utils/custom_exceptions.py>

<utils/game_settings.py>

import chess.engine
import pandas as pd
from pathlib import Path
import chess

base_directory = Path(__file__).parent

pd.set_option('display.max_columns', None)

PRINT_TRAINING_RESULTS = False
PRINT_STEP_BY_STEP = False

agent_vs_agent_num_games = 100 # number of games that agents will play against each other

# the following numbers are based on centipawn scores
CHESS_MOVE_VALUES: dict[str, int] = {
        'new_move': 100, # a move that has never been made before
        'capture': 150,
        'piece_development': 200,
        'check': 300,
        'promotion': 500,
        'promotion_queen': 900,
        'mate_score': 1_000
    }

training_sample_size = 50_000
max_num_turns_per_player = 200
max_turn_index = max_num_turns_per_player * 2 - 1

initial_q_val = 1 # this is relevant when training an agent. SARSA algorithm requires an initial value
chance_for_random_move = 0.05 # 10% chance that RL agent selects random chess move
        
# The following values are for the chess engine analysis of moves.
# we only want to look ahead one move, that's the anticipated q value at next state, and next action
num_moves_to_return = 1
depth_limit = 1
time_limit = None
search_limit = chess.engine.Limit(depth = depth_limit, time = time_limit)

stockfish_filepath = base_directory / ".." / "stockfish_15_win_x64_avx2" / "stockfish_15_x64_avx2.exe"
absolute_stockfish_filepath = stockfish_filepath.resolve()

bradley_agent_q_table_path = base_directory / ".." / "Q_Tables" / "bradley_agent_q_table.pkl"
imman_agent_q_table_path = base_directory / ".." / "Q_Tables" / "imman_agent_q_table.pkl"
unique_chess_moves_list_path = base_directory / ".." / "Q_Tables" / "unique_chess_moves_list.pkl"

helper_methods_logger_filepath = base_directory / ".." / "debug" / "helper_methods_logger_file.txt"
agent_logger_filepath = base_directory / ".." / "debug" / "agent_logger_file.txt"
environ_logger_filepath = base_directory / ".." / "debug" / "environ_logger_file.txt"
training_functions_logger_filepath = base_directory / ".." / "debug" / "training_functions_logger_file.txt"
agent_vs_agent_logger_filepath = base_directory / ".." / "debug" / "agent_vs_agent_logger_file.txt"

initial_training_results_filepath = base_directory / ".." / "training_results" / "initial_training_results.txt"
additional_training_results_filepath = base_directory / ".." / "training_results" / "additional_training_results.txt"
agent_vs_agent_filepath = base_directory / ".." / "training_results" / "agent_vs_agent_games.txt"
agent_vs_human_logger_filepath = base_directory / ".." / "training_results" / "agent_vs_human_logger_file.txt" 

<end of utils/game_settings.py>

<utils/helper_methods.py>

import pandas as pd
import game_settings
import random
import chess.engine
import logging
import Agent
import custom_exceptions
from utils.logging_config import setup_logger

helper_methods_logger = setup_logger(__name__, game_settings.helper_methods_errors_filepath)

def agent_selects_and_plays_chess_move(chess_agent, environ) -> str:
    try:
        curr_state = environ.get_curr_state()
    except Exception as e:
        error_message = f'error at agent_selects_and_plays_chess_move: {str(e)}, unable to retrieve current state\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.StateRetrievalError(error_message) from e
    
    chess_move: str = chess_agent.choose_action(curr_state) # we're not training, so we don't need to pass current_game

    try:
        environ.load_chessboard(chess_move)
    except Exception as e:
        error_message = f'error at agent_selects_and_plays_chess_move: {str(e)}, failed to load chessboard with move: {chess_move}\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.ChessboardLoadError(error_message) from e

    try:
        environ.update_curr_state()
        return chess_move
    except Exception as e:
        error_message = f'error at agent_selects_and_plays_chess_move: {str(e)}, failed to update current state\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.StateUpdateError(error_message) from e
### end of agent_selects_and_plays_chess_move

def receive_opponent_move(chess_move: str, environ) -> bool:                                                                                 
    try:
        environ.load_chessboard(chess_move)
    except Exception as e:
        error_message = f'error at receive_opp_move: {str(e)}, failed to load chessboard with move: {chess_move}\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.ChessboardLoadError(error_message) from e

    try:
        environ.update_curr_state()
        return True
    except Exception as e:
        error_message = f'error at receive_opp_move: {str(e)}, failed to update current state\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.StateUpdateError(error_message) from e
### end of receive_opp_move

def pikl_q_table(chess_agent, q_table_path: str) -> None:
    chess_agent.q_table.to_pickle(q_table_path, compression = 'zip')
### end of pikl_q_table

def bootstrap_agent(chess_agent, existing_q_table_path: str) -> Agent.Agent:
    chess_agent.q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    chess_agent.is_trained = True
    return chess_agent
### end of bootstrap_agent

def get_number_with_probability(probability: float) -> int:
    if random.random() < probability:
        return 1
    else:
        return 0
### end of get_number_with_probability

def reset_q_table(q_table) -> None:
    q_table.iloc[:, :] = 0
    return q_table    
### end of reset_q_table ###

def is_game_over(environ) -> bool:
    try:
        return (
            environ.board.is_game_over() or
            environ.turn_index >= game_settings.max_turn_index or
            (len(environ.get_legal_moves()) == 0)
        )
    except Exception as e:
        error_message = f'error at is_game_over: {str(e)}, failed to determine if game is over\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameOverError(error_message) from e
### end of is_game_over

def get_game_outcome(environ) -> str:
    try:
        return environ.board.outcome().result()
    except Exception as e:
        error_message = f'error at get_game_outcome: {str(e)}, failed to get game outcome\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameOutcomeError(error_message) from e
### end of get_game_outcome

def get_game_termination_reason(environ) -> str:
    try:
        return str(environ.board.outcome().termination)
    except Exception as e:
        error_message = f'error at get_game_termination_reason: {str(e)}, failed to get game end reason\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameTerminationError(error_message) from e
### end of get_game_termination_reason 

<end of utils/helper_methods.py> 

<utils/logging_config.py> 

import logging
from pathlib import Path
import game_settings

def setup_logger(name: str, log_file: str, level=logging.ERROR) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

<end of utils/logging_config.py>