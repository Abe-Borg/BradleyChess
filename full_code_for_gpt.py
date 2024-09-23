<agents/Agent.py> 

from utils import game_settings, helper_methods, custom_exceptions, constants
import pandas as pd
import numpy as np
from typing import Union, Dict, List, Optional
from utils.logging_config import setup_logger
agent_logger = setup_logger(__name__, game_settings.agent_errors_filepath)

class Agent:
    def __init__(self, color: str, learn_rate: float = constants.default_learning_rate, discount_factor: float = constants.default_discount_factor, q_table: Optional[pd.DataFrame] = None):
        self.color = color
        self.learn_rate = learn_rate
        self.discount_factor = discount_factor
        self.is_trained: bool = False
        self.q_table = q_table if q_table is not None else pd.DataFrame()

    def choose_action(self, chess_data, environ_state: Dict[str, Union[int, str, List[str]]], curr_game: str = 'Game 1') -> str:
        if not chess_data:
            chess_data = {}
        legal_moves = environ_state['legal_moves']
        if not legal_moves:
            agent_logger.info(f'Agent.choose_action: legal_moves is empty. curr_game: {curr_game}, curr_turn: {environ_state['curr_turn']}\n')
            return ''
        self.update_q_table(legal_moves)
        if self.is_trained:
            return self.policy_game_mode(legal_moves, environ_state['curr_turn'])
        else:
            return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
    
    def policy_training_mode(self, chess_data, curr_game: str, curr_turn: str) -> str:
        try:
            chess_move = chess_data.at[curr_game, curr_turn]
            return chess_move
        except KeyError as e:
            error_message = f'Failed to choose action at policy_training_mode. curr_game: {curr_game}, curr_turn: {curr_turn} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message) from e

    def policy_game_mode(self, legal_moves: List[str], curr_turn: str) -> str:
        dice_roll = helper_methods.get_number_with_probability(game_settings.chance_for_random_move)
        legal_moves_in_q_table = self.q_table[curr_turn].loc[self.q_table[curr_turn].index.intersection(legal_moves)]
        if legal_moves_in_q_table.empty:
            error_message = f'at policy_game_mode: legal moves not found in q_table or legal_moves is empty.'
            agent_logger.error(error_message)
            raise custom_exceptions.FailureToChooseActionError(error_message)

        if dice_roll == 1:
            chess_move = legal_moves_in_q_table.sample().index[0]
        else:
            chess_move = legal_moves_in_q_table.idxmax()
        return chess_move

    def change_q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        try:    
            self.q_table.at[chess_move, curr_turn] += pts
        except KeyError as e:
            error_message = f'@ change_q_table_pts(). Failed to change q_table points. chess_move: {chess_move}, curr_turn: {curr_turn}, pts: {pts} due to error: {str(e)}'
            agent_logger.error(error_message)
            raise custom_exceptions.QTableUpdateError(error_message) from e

    def update_q_table(self, new_chess_moves: Union[str, List[str]]) -> None:
        if isinstance(new_chess_moves, str):
            new_chess_moves = [new_chess_moves]
        truly_new_moves = set(new_chess_moves) - set(self.q_table.index)
        if not truly_new_moves:
            return
        q_table_new_values: pd.DataFrame = pd.DataFrame(
            0, 
            index = truly_new_moves,
            columns = self.q_table.columns, 
            dtype = np.int64
        )
        self.q_table = pd.concat([self.q_table, q_table_new_values])

<end of agents/Agent.py> 

<environment/Environ.py>

import chess
from utils import custom_exceptions, game_settings, constants
from utils.logging_config import setup_logger 
from typing import Union, Dict, List
environ_logger = setup_logger(__name__, game_settings.environ_errors_filepath)

class Environ:
    def __init__(self):

        try: 
            self.board: chess.Board = chess.Board()
            
            # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
            max_turns = constants.max_num_turns_per_player * 2 # 2 players
            self.turn_list: List[str] = [f'{"W" if i % 2 == 0 else "B"}{i // 2 + 1}' for i in range(max_turns)]
            self.turn_index: int = 0
        except Exception as e:
            environ_logger.error(f'at __init__: failed to initialize environ. Error: {e}\n', exc_info=True)
            raise custom_exceptions.EnvironInitializationError(f'failed to initialize environ due to error: {e}') from e

    def get_curr_state(self) -> Dict[str, Union[int, str, List[str]]]:
        if not (0 <= self.turn_index < len(self.turn_list)):
            message = f'Turn index out of range: {self.turn_index}'
            environ_logger.error(message)
            raise custom_exceptions.TurnIndexError(message)
    
        curr_turn = self.get_curr_turn()
        legal_moves = self.get_legal_moves()     
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    
    def update_curr_state(self) -> None:
        if self.turn_index >= constants.max_turn_index:
            message = f'ERROR: max_turn_index reached: {self.turn_index} >= {constants.max_turn_index}\n'
            environ_logger.error(message)
            raise IndexError(message)
        if self.turn_index >= len(self.turn_list):
            message = f'ERROR: turn index out of bounds: {self.turn_index} >= {len(self.turn_list)}\n'
            environ_logger.error(message)
            raise IndexError(message)
        self.turn_index += 1
    
    def get_curr_turn(self) -> str:                        
        if not (0 <= self.turn_index < len(self.turn_list)):
            environ_logger.error(f'ERROR: Turn index out of range: {self.turn_index}\n')
            raise custom_exceptions.TurnIndexError(f'Turn index out of range: {self.turn_index}')
        
        return self.turn_list[self.turn_index]
    
    def load_chessboard(self, chess_move: str, curr_game = 'Game 1') -> None:
        try:
            self.board.push_san(chess_move)
        except Exception as e:
            error_message = f'An error occurred at load_chessboard: {str(e)}, unable to load chessboard with {chess_move} in {curr_game}'
            environ_logger.error(error_message)
            raise custom_exceptions.InvalidMoveError(error_message) from e

    def pop_chessboard(self) -> None:
        try:
            self.board.pop()
        except Exception as e:
            error_message = f'An error occurred at pop_chessboard. unable to pop chessboard, due to error: {str(e)}'
            environ_logger.error(error_message)
            raise custom_exceptions.ChessboardPopError(error_message) from e

    def undo_move(self) -> None:
        try:
            self.board.pop()
            if self.turn_index > 0:
                self.turn_index -= 1
        except Exception as e:
            error_message = f'An error occurred at undo_move, unable to undo move due to error: {str(e)}, at turn index: {self.turn_index}'
            environ_logger.error(error_message)
            raise custom_exceptions.ChessboardPopError(error_message) from e

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

    def reset_environ(self) -> None:
        self.board.reset()
        self.turn_index = 0
    
    def get_legal_moves(self) -> List[str]:   
        try:
            return [self.board.san(move) for move in self.board.legal_moves]
        except Exception as e:
            error_message = f'An error occurred at get_legal_moves: {str(e)}, legal moves could not be retrieved, at turn index: {self.turn_index}, current turn: {self.get_curr_turn()}, current board state: {self.board}, current legal moves: {self.board.legal_moves}'
            environ_logger.error(error_message)
            raise custom_exceptions.NoLegalMovesError(error_message) from e
    
<end of environment/Environ.py>

<main/agent_vs_agent.py>

from utils import helper_methods, game_settings, custom_exceptions
import time
from environment import Environ
from agents import Agent
from utils.logging_config import setup_logger 
agent_vs_agent_logger = setup_logger(__name__, game_settings.agent_vs_agent_logger_filepath)

def agent_vs_agent(environ, w_agent, b_agent, print_to_screen = False, current_game = 0) -> None:
    agent_vs_agent_logger.info(f'Playing game {current_game}\n')
    try:    
        while helper_methods.is_game_over(environ) == False:
            chess_move = helper_methods.agent_selects_and_plays_chess_move(w_agent, environ)
            agent_vs_agent_logger.info(f'\nCurrent turn: {environ.get_curr_turn()}')
            agent_vs_agent_logger.info(f'White agent played {chess_move}')
            agent_vs_agent_logger.info(f'Current turn is: {environ.get_current_turn()}. \nWhite agent played {chess_move}\n')
            if helper_methods.is_game_over(environ) == False:
                chess_move = helper_methods.agent_selects_and_plays_chess_move(b_agent, environ)
                agent_vs_agent_logger.info(f'Black agent played {chess_move} curr board is:\n{environ.board}\n')
    except custom_exceptions.GamePlayError as e:
        agent_vs_agent_logger.error(f'An error occurred at agent_vs_agent: {e}')
        raise

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
        number_of_games = int(input('How many games do you want the agents to play? '))
        print_to_screen = (input('Do you want to print the games to the screen? (y/n) ')).lower()[0]
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

from utils import helper_methods, game_settings, custom_exceptions
import time
from environment import Environ
from agents import Agent
from utils.logging_config import setup_logger
agent_vs_human_logger = setup_logger(__name__, game_settings.agent_vs_human_logger_filepath)

def play_game_vs_human(environ: Environ.Environ, chess_agent: Agent.Agent) -> None:
    player_turn = 'W'
    try:
        while not helper_methods.is_game_over(environ):
            print(f'\nCurrent turn is :  {environ.get_curr_turn()}\n')
            chess_move = handle_move(player_turn, chess_agent, environ)
            print(f'{player_turn} played {chess_move}\n')
            player_turn = 'B' if player_turn == 'W' else 'W'

        print(f'Game is over, result is: {helper_methods.get_game_outcome(environ)}')
        print(f'The game ended because of: {helper_methods.get_game_termination_reason(environ)}')
    except custom_exceptions.GamePlayError as e:
        print(f'An error occurred at play_game_vs_human: {e}')
        error_message = f'An error occurred at play_game_vs_human: {str(e)}'
        agent_vs_human_logger.error(error_message)
        raise 
    
    finally:
        environ.reset_environ()

def handle_move(player_color: str, chess_agent: Agent.Agent, environ: Environ.Environ) -> str:
    if player_color == chess_agent.color:
        print('=== RL AGENT\'S TURN ===\n')
        chess_move = helper_methods.agent_selects_and_plays_chess_move(chess_agent, environ)
    else:
        print('=== OPPONENT\'S TURN ===')
        while True:
            chess_move = input('Enter chess move: or type \'exit\' to quit: ')
            try:
                if helper_methods.receive_opponent_move(chess_move, environ):
                    return chess_move
                if chess_move == 'exit':
                    print('Exiting game...')
                    quit()
            except custom_exceptions.ChessboardLoadError as e:
                agent_vs_human_logger.error(e)
                print('Failed to load move. Try again.')
            except custom_exceptions.StateUpdateError as e:
                agent_vs_human_logger.error(e)
                print('Failed to update state. Try again.')

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
        exit(1)

    end_time = time.time()
    total_time = end_time - start_time
    print('agent vs human game is complete')
    print(f'it took: {total_time}')

<end of main/agent_vs_human.py>

<main/continue_training_agents.py>

from utils import helper_methods, game_settings, custom_exceptions, constants
import time
from training import training_functions
from agents import Agent
from environment import Environ
from utils.logging_config import setup_logger
agent_vs_agent_logger = setup_logger(__name__, game_settings.agent_vs_agent_logger_filepath)

if __name__ == '__main__':
    start_time = time.time()
    environ = Environ.Environ()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')
    helper_methods.bootstrap_agent(bradley, game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent(imman, game_settings.imman_agent_q_table_path)
    num_games_to_play = constants.agent_vs_agent_num_games

    try:
        training_functions.continue_training_rl_agents(num_games_to_play, bradley, imman, environ)
        helper_methods.pikl_q_table(bradley, game_settings.bradley_agent_q_table_path)
        helper_methods.pikl_q_table(imman, game_settings.imman_agent_q_table_path)
    except custom_exceptions.TrainingError as e:
        print(f'training interrupted because of:  {e}')
        agent_vs_agent_logger.error(f'An error occurred: {e}')
        exit(1)
    end_time = time.time()
    total_time = end_time - start_time
    print('agent v agent training round is complete')
    print(f'it took: {total_time}') 

<end of main/continue_training_agents.py> 

<main/train_new_agents.py>

from utils import helper_methods, game_settings, custom_exceptions, constants
import pandas as pd
import time
from training import training_functions
from agents import Agent
from utils.logging_config import setup_logger
train_new_agents_logger = setup_logger(__name__, game_settings.train_new_agents_logger_filepath)

if __name__ == '__main__':
    start_time = time.time()
    bradley = Agent.Agent('W')
    imman = Agent.Agent('B')

    # change this each time for new section of the database
    # estimated q table number must match chess_data number
    estimated_q_values_table = pd.read_pickle(game_settings.est_q_vals_filepath_part_1, compression = 'zip')
    chess_data = pd.read_pickle(game_settings.chess_games_filepath_part_1, compression = 'zip')

    try:
        training_functions.train_rl_agents(chess_data, estimated_q_values_table, bradley, imman)
    except custom_exceptions.TrainingError as e:
        print(f'training interrupted because of:  {e}')
        train_new_agents_logger.error(f'An error occurred: {e}')
        exit(1)
        
    end_time = time.time()
    helper_methods.pikl_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.pikl_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)
    total_time = end_time - start_time
    print('training is complete')
    print(f'it took: {total_time} for {constants.training_sample_size} games\n') 

<end of main/train_new_agents.py>

<training/training_functions.py>

from typing import Tuple
from agents import Agent
import chess
from utils import game_settings, custom_exceptions, constants
from environment import Environ
import pandas as pd
import copy
import re
from utils.logging_config import setup_logger
training_functions_logger = setup_logger(__name__, game_settings.training_functions_logger_filepath)

def train_rl_agents(chess_data, est_q_val_table, w_agent, b_agent) -> Tuple[Agent.Agent, Agent.Agent]:
    for game_number in chess_data.index:
        w_curr_q_value: int = game_settings.initial_q_val
        b_curr_q_value: int = game_settings.initial_q_val

        try: 
            train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at train_one_game: {e}\nat game: {game_number}')
            raise
    
    w_agent.is_trained = True
    b_agent.is_trained = True
    return w_agent, b_agent

def train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value) -> None:
    num_moves: int = chess_data.at[game_number, 'PlyCount']
    environ = Environ.Environ()
    curr_state = environ.get_curr_state()

    while curr_state['turn_index'] < (num_moves):
        ##################### WHITE'S TURN ####################
        # choose action a from state s, using policy
        w_chess_move = w_agent.choose_action(chess_data, curr_state, game_number)

        if not w_chess_move:
            training_functions_logger.error(f'An error occurred at w_agent.choose_action\n w_chess_move is empty at state: {curr_state}\n')
            raise custom_exceptions.EmptyChessMoveError(f"w_chess_move is empty at state: {curr_state}")

        ### ASSIGN POINTS TO q TABLE FOR WHITE AGENT ###
        # on the first turn for white, this would assign to W1 col at chess_move row.
        # on W's second turn, this would be q_next which is calculated on the first loop.                
        assign_points_to_q_table(w_chess_move, curr_state['curr_turn'], w_curr_q_value, w_agent)
        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ### WHITE AGENT PLAYS THE SELECTED MOVE ###
        apply_move_and_update_state(w_chess_move, game_number, environ)
        w_reward = get_reward(w_chess_move)
        curr_state = environ.get_curr_state()

        if environ.board.is_game_over() or curr_state['turn_index'] >= (num_moves) or not curr_state['legal_moves']:
            break
        else:
            # the var curr_turn_for_q_values is here because we previously moved to next turn (after move was played)
            # but we want to assign the q est based on turn just before the curr turn was incremented.
            w_est_q_value: int = est_q_val_table.at[game_number, curr_turn_for_q_est]

        ##################### BLACK'S TURN ####################
        # choose action a from state s, using policy
        b_chess_move = b_agent.choose_action(chess_data, curr_state, game_number)

        if not b_chess_move:
            training_functions_logger.error(f'An error occurred at b_agent.choose_action\n w_chess_move is empty at state: {curr_state}\n')
            raise custom_exceptions.EmptyChessMoveError(f"b_chess_move is empty at state: {curr_state}")

        # assign points to q table
        assign_points_to_q_table(b_chess_move, curr_state['curr_turn'], b_curr_q_value, b_agent)

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ##### BLACK AGENT PLAYS SELECTED MOVE #####
        apply_move_and_update_state(b_chess_move, game_number, environ) 
        b_reward = get_reward(b_chess_move)
        curr_state = environ.get_curr_state()

        if environ.board.is_game_over() or not curr_state['legal_moves']:
            break 
        else:
            b_est_q_value: int = est_q_val_table.at[game_number, curr_turn_for_q_est]

        # SARSA Update
        w_next_q_value: int = find_next_q_value(w_curr_q_value, w_agent.learn_rate, w_reward, w_agent.discount_factor, w_est_q_value)
        b_next_q_value: int = find_next_q_value(b_curr_q_value, b_agent.learn_rate, b_reward, b_agent.discount_factor, b_est_q_value)
    
        # on the next turn, w_next_q_value and b_next_q_value will be added to the q table. so if this is the end of the first round,
        # next round it will be W2 and then we assign the q value at W2 col
        w_curr_q_value = w_next_q_value
        b_curr_q_value = b_next_q_value
        curr_state = environ.get_curr_state()
    environ.reset_environ()

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

def generate_q_est_df_one_game(chess_data, game_number, w_agent, b_agent) -> None:
    num_moves: int = chess_data.at[game_number, 'PlyCount']
    environ = Environ.Environ()
    engine = start_chess_engine()
    
    try:
        curr_state = environ.get_curr_state()
    except Exception as e:
        training_functions_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
        training_functions_logger.error(f'at: {game_number}\n')
        raise Exception from e
    
    ### LOOP PLAYS THROUGH ONE GAME ###
    while curr_state['turn_index'] < (num_moves):
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
            if environ.board.is_game_over() or curr_state['turn_index'] >= (num_moves) or not curr_state['legal_moves']:
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
            if environ.board.is_game_over() or curr_state['turn_index'] >= (num_moves) or not curr_state['legal_moves']:
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
        est_qval = constants.CHESS_MOVE_VALUES['mate_score']

    # IMPORTANT STEP, pop the chessboard of last move, we are estimating board states, not
    # playing a move.
    try:
        environ.pop_chessboard()
    except Exception as e:
        training_functions_logger.error(f'@ find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error("failed to pop_chessboard after est q val analysis values found\n")
        raise Exception from e

    return est_qval

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

def analyze_board_state(board, engine) -> dict:
    if not board.is_valid():
        training_functions_logger.error(f'at analyze_board_state. Board is in invalid state\n')
        raise custom_exceptions.InvalidBoardStateError(f'at analyze_board_state. Board is in invalid state\n')
    try: 
        analysis_result = engine.analyse(board, game_settings.search_limit, multipv = constants.chess_engine_num_moves_to_return)
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

def get_reward(chess_move: str) -> int:
    if not chess_move or not isinstance(chess_move, str):
        raise custom_exceptions.RewardCalculationError("Invalid chess move input")

    total_reward = 0
    if re.search(r'[NRBQ]', chess_move):
        total_reward += constants.CHESS_MOVE_VALUES['piece_development']
    if 'x' in chess_move:
        total_reward += constants.CHESS_MOVE_VALUES['capture']
    if '=' in chess_move:
        total_reward += constants.CHESS_MOVE_VALUES['promotion']
        if '=Q' in chess_move:
            total_reward += constants.CHESS_MOVE_VALUES['promotion_queen']
    return total_reward

def start_chess_engine(): 
    try:
        chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
        return chess_engine
    except custom_exceptions.EngineStartError as e:
        training_functions_logger.error(f'An error occurred at start_chess_engine: {e}\n')
        raise Exception from e

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

<end of training/training_functions.py> 

<utils/helper_methods.py>

import pandas as pd
from utils import game_settings, constants, custom_exceptions
import random
from agents import Agent
from utils.logging_config import setup_logger
helper_methods_logger = setup_logger(__name__, game_settings.helper_methods_errors_filepath)

def agent_selects_and_plays_chess_move(chess_agent, environ) -> str:
    curr_state = environ.get_curr_state() 
    chess_move: str = chess_agent.choose_action(curr_state)
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return chess_move

def receive_opponent_move(chess_move: str, environ) -> bool:                                                                                 
    environ.load_chessboard(chess_move)
    environ.update_curr_state()
    return True

def pikl_q_table(chess_agent, q_table_path: str) -> None:
    chess_agent.q_table.to_pickle(q_table_path, compression = 'zip')

def bootstrap_agent(chess_agent, existing_q_table_path: str) -> Agent.Agent:
    chess_agent.q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    chess_agent.is_trained = True
    return chess_agent

def get_number_with_probability(probability: float) -> int:
    if random.random() < probability:
        return 1
    else:
        return 0

def reset_q_table(q_table) -> None:
    q_table.iloc[:, :] = 0
    return q_table    

def is_game_over(environ) -> bool:
    try:
        return (
            environ.board.is_game_over() or
            environ.turn_index >= constants.max_turn_index or
            (len(environ.get_legal_moves()) == 0)
        )
    except Exception as e:
        error_message = f'error at is_game_over: {str(e)}, failed to determine if game is over\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameOverError(error_message) from e

def get_game_outcome(environ) -> str:
    try:
        return environ.board.outcome().result()
    except Exception as e:
        error_message = f'error at get_game_outcome: {str(e)}, failed to get game outcome\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameOutcomeError(error_message) from e

def get_game_termination_reason(environ) -> str:
    try:
        return str(environ.board.outcome().termination)
    except Exception as e:
        error_message = f'error at get_game_termination_reason: {str(e)}, failed to get game end reason\n'
        helper_methods_logger.error(error_message)
        raise custom_exceptions.GameTerminationError(error_message) from e

<end of utils/helper_methods.py> 

<utils/logging_config.py> 

import logging
def setup_logger(name: str, log_file: str, level=logging.ERROR) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

<end of utils/logging_config.py> 




I need you to review my latest code. It is a program that is meant to train chess playing agents. I use a database of 9 million games. There are two stages of training. In the first stage, the agents play out the moves in the database exactly as shown, learning through the SARSA algorithm and utilizing a Q table. In the second stage, the agents play against each other, while still modifying their Q table. The moves in the second stage of training are selected based on the values in the Q tables for each agent. Here is what I think is the most important and relevant parts of my  directory structure (not everything is included in this because I feel that some things don't need to be analyzed at this time). 

BradleyChess/
│
├── agents/
│   └── Agent.py
│
├── environment/
│   └── Environ.py
│
├── main/
│   ├── agent_vs_agent.py
│   ├── agent_vs_human.py
│   ├── continue_training_agents.py
│   └── train_new_agents.py
│
├── training/
│   └── training_functions.py
│
└── utils/
    ├──  helper_methods.py
    ├── logging_config.py 

I am using pandas dataframes to store the chess database games. the dataframes are pkld right now. 

this is what the dataframes look like: 
         PlyCount   W1   B1    W2    B2
Game 1          4   e4   e5   Nf3   Nc6
Game 2          4   d4   d5   Nc3   Nf6

the row indices are strings. All column indices are strings. The cell values for PlyCount are ints. The other cells are strings which represent chess moves in algebraic notation.

I will need to implement parallel processing for the training. 

Your purpose in this chat is to review my code and help me implement parallel processing for training. Since I have so many chess games to process, I need to implement parallel processing. I am going to share all of my code first. Review it to get familiar with the project. Then wait for me to share a plan for parallel processing that I developed before this chat. You will need to review that closely as well. After you have closely examined both, you need to formulate a plan on how we are going to implement parallel processing to train the agents. When you formulate your own plan you will need to have incorporated anything of value from the plan I developed. I am leaving it up to your judgement. Throughout this process I need you to be very opinionated. You will tell me what the best way to do things is. You will also avoid being overly verbose; get to the point. You will need to tell me exactly what to change and you shall show me the full code. You will need to walk me through this step by step. Before we start, summarize what I've just told you. I want to make sure you understand the task.