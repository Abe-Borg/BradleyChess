
class Agent:
    def __init__(self, color: str, learn_rate = 0.6, discount_factor = 0.35, q_table: pd.DataFrame = None):        
        self.learn_rate = learn_rate
        self.discount_factor = discount_factor
        self.color = color
        self.is_trained: bool = False
        self.q_table: pd.DataFrame = q_table # q table will be assigned at program execution.
    ### end of __init__ ###

    def choose_action(self, chess_data, environ_state: dict[str, str, list[str]], curr_game: str = 'Game 1') -> str:
        if environ_state['legal_moves'] == []:
            return ''
        
        self.update_q_table(environ_state['legal_moves']) # this func also checks if there are any new unique move strings

        if self.is_trained:
            return self.policy_game_mode(environ_state['legal_moves'], environ_state['curr_turn'])
        else:
            return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
    ### end of choose_action ###
    
    def policy_training_mode(self, chess_data, curr_game: str, curr_turn: str) -> str:
        try:
            chess_move = chess_data.at[curr_game, curr_turn]
            return chess_move
        except Exception as e:
            raise Exception from e
    ### end of policy_training_mode ###

    def policy_game_mode(self, legal_moves: list[str], curr_turn: str) -> str:
        dice_roll = helper_methods.get_number_with_probability(game_settings.chance_for_random_move)
        
        try:
            legal_moves_in_q_table = self.q_table[curr_turn].loc[self.q_table[curr_turn].index.intersection(legal_moves)]
        except Exception as e:
            raise Exception from e

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
            raise Exception from e
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

        q_table_new_values: pd.DataFrame = pd.DataFrame(
            0, 
            index = list(truly_new_moves), 
            columns = self.q_table.columns, 
            dtype = np.int64
        )

        self.q_table = pd.concat([self.q_table, q_table_new_values])
    ### update_q_table ###

class Environ:
    def __init__(self):
        self.board: chess.Board = chess.Board()
        
        # turn_list and turn_index work together to track the current turn (a string like this, 'W1')
        max_turns = game_settings.max_num_turns_per_player * 2 # 2 players
        self.turn_list: list[str] = [f'{"W" if i % 2 == 0 else "B"}{i // 2 + 1}' for i in range(max_turns)]
        self.turn_index: int = 0
    ### end of constructor

    def get_curr_state(self) -> dict[str, str, list[str]]:
        if not (0 <= self.turn_index < len(self.turn_list)):
            raise IndexError(f'Turn index out of range: {self.turn_index}')
    
        curr_turn = self.get_curr_turn()
        legal_moves = self.get_legal_moves()
        return {'turn_index': self.turn_index, 'curr_turn': curr_turn, 'legal_moves': legal_moves}
    ### end of get_curr_state
    
    def update_curr_state(self) -> None:
        if self.turn_index >= game_settings.max_turn_index:
            # self.environ_logger.error(f'ERROR: max_turn_index reached: {self.turn_index} >= {game_settings.max_turn_index}\n')
            raise IndexError(f"Maximum turn index ({game_settings.max_turn_index}) reached!")
    
        if self.turn_index >= len(self.turn_list):
            raise IndexError(f"Turn index out of bounds: {self.turn_index}")
    
        self.turn_index += 1
    ### end of update_curr_state
    
    def get_curr_turn(self) -> str:                        
        if not (0 <= self.turn_index < len(self.turn_list)):
            raise IndexError(f'Turn index out of range: {self.turn_index}')
        
        return self.turn_list[self.turn_index]
        ### end of get_curr_turn
    
    def load_chessboard(self, chess_move: str, curr_game = 'Game 1') -> None:
        try:
            self.board.push_san(chess_move)
        except ValueError as e:
            raise ValueError(e) from e
    ### end of load_chessboard    

    def pop_chessboard(self) -> None:
        try:
            self.board.pop()
        except IndexError as e:
            raise IndexError(f"An error occurred: {e}, unable to pop chessboard'")
    ### end of pop_chessboard

    def undo_move(self) -> None:
        try:
            self.board.pop()

            if self.turn_index > 0:
                self.turn_index -= 1
        except IndexError as e:
            raise IndexError(e) from e
    ### end of undo_move

    def load_chessboard_for_Q_est(self, analysis_results: list[dict]) -> None:
        anticipated_chess_move = analysis_results['anticipated_next_move']  # this has the form like this, Move.from_uci('e4f6')
        self.environ_logger.debug(f'anticipated_chess_move: {anticipated_chess_move}. This should have the form like this, Move.from_uci(\'e4f6\')\n')
        try:
            move = chess.Move.from_uci(anticipated_chess_move)
            self.board.push(move)    
        except ValueError as e:
            raise ValueError(e) from e
    ### end of load_chessboard_for_Q_est

    def reset_environ(self) -> None:
        self.board.reset()
        self.turn_index = 0
    ### end of reset_environ
    
    def get_legal_moves(self) -> list[str]:   
        return [self.board.san(move) for move in self.board.legal_moves]
    ### end of get_legal_moves

import helper_methods
import chess
import logging
import game_settings
import Environ
import pandas as pd
import copy
import custom_exceptions

# Logger Initialization
training_functions_logger = logging.getLogger(__name__)
training_functions_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler(game_settings.training_functions_logger_filepath)   ### <<< ====== create this filepath
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
error_handler.setFormatter(formatter)
training_functions_logger.addHandler(error_handler)

def train_rl_agents(chess_data, est_q_val_table, w_agent, b_agent):
    """
        Trains the RL agents using the SARSA algorithm and sets their `is_trained` flag to True.
        This method trains two RL agents by having them play games from a database exactly as shown, and learning from that. 
        The agents learn from these games using the SARSA (State-Action-Reward-State-Action) algorithm.
        
        Args:
            est_q_val_table (pd.DataFrame): A DataFrame containing the estimated q values for each game in the training set.
        Raises:
            Exception: A TrainingError is raised if an error occurs while getting the current state, choosing an action, playing a move, or getting the latest current state. The exception is written to the errors file.
        Side Effects:
            Modifies the q tables of the RL agents and sets their `is_trained` flag to True.
            Writes the start and end of each game, any errors that occur, and the final state of the chessboard to the initial training results file.
            Writes any errors that occur to the errors file.
            Resets the environment at the end of each game.
    """
    ### FOR EACH GAME IN THE TRAINING SET ###
    for game_number in chess_data.index:
        w_curr_q_value: int = game_settings.initial_q_val
        b_curr_q_value: int = game_settings.initial_q_val

        train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value)

    # training is complete, all games in database have been processed
    ### I will need to use a pool.Queue to collect all the values to be input to the q table at the end of training.
    
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
        # training_functions_logger.error(f'An error occurred environ.get_curr_state: {e}\n')
        # training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
        # training_functions_logger.error(f'at game: {game_number}\n')
        # training_functions_logger.error(f'at turn: {curr_state['turn_index']}')
        return

    ### THIS WHILE LOOP PLAYS THROUGH ONE GAME ###
    while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
        ##################### WHITE'S TURN ####################
        # choose action a from state s, using policy
        w_chess_move = w_agent.choose_action(chess_data, curr_state, game_number)

        if not w_chess_move:
            # training_functions_logger.error(f'An error occurred at w_agent.choose_action\n')
            # training_functions_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
            break # game is over, exit function.

        ### ASSIGN POINTS TO q TABLE FOR WHITE AGENT ###
        # on the first turn for white, this would assign to W1 col at chess_move row.
        # on W's second turn, this would be q_next which is calculated on the first loop.                
        assign_points_to_q_table(w_chess_move, curr_state['curr_turn'], w_curr_q_value, w_agent)

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ### WHITE AGENT PLAYS THE SELECTED MOVE ###
        # take action a, observe r, s', and load chessboard
        try:
            rl_agent_plays_move_during_training(w_chess_move, game_number, environ)
        except Exception as e:
            # training_functions_logger.error(f'An error occurred at rl_agent_plays_move_during_training: {e}\n')
            # training_functions_logger.error(f'at game_number: {game_number}\n')
            # training_functions_logger.error(f'at state: {curr_state}\n')
            break # game is over, exit function.

        w_reward = get_reward(w_chess_move)

        # get latest curr_state since rl_agent_plays_move_during_training updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            # training_functions_logger.error(f'An error occurred at get_curr_state: {e}\n')
            # training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
            # training_functions_logger.error(f'At game: {game_number}\n')
            # training_functions_logger.error(f'at state: {curr_state}\n')
            break # game is over, exit function.
        
        # find the estimated q value for White, but first check if game ended
        if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
            break # game is over, exit function.

        else: # current game continues
            # the var curr_turn_for_q_values is here because we previously moved to next turn (after move was played)
            # but we want to assign the q est based on turn just before the curr turn was incremented.
            w_est_q_value: int = est_q_val_table.at[game_number, curr_turn_for_q_est]

        ##################### BLACK'S TURN ####################
        # choose action a from state s, using policy
        b_chess_move = b_agent.choose_action(chess_data, curr_state, game_number)

        if not b_chess_move:
            # training_functions_logger.error(f'An error occurred at w_agent.choose_action\n')
            # training_functions_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
            # training_functions_logger.error(f'at: {game_number}\n')
            break # game is over, exit function

        # assign points to q table
        assign_points_to_q_table(b_chess_move, curr_state['curr_turn'], b_curr_q_value, b_agent)

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ##### BLACK AGENT PLAYS SELECTED MOVE #####
        # take action a, observe r, s', and load chessboard
        try:
            rl_agent_plays_move_during_training(b_chess_move, game_number, environ)
        except Exception as e:
            # training_functions_logger.error(f'An error occurred at rl_agent_plays_move_during_training: {e}\n')
            # training_functions_logger.error(f'at game_number: {game_number}\n')
            # training_functions_logger.error(f'at state: {curr_state}\n')
            break # game is over, exit function

        b_reward = get_reward(b_chess_move)

        # get latest curr_state since rl_agent_plays_move_during_training updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            # training_functions_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
            # training_functions_logger.error(f'curr board is:\n{environ.board}\n\n')
            # training_functions_logger.error(f'At game: {game_number}\n')
            break # game is over, exit function

        # find the estimated q value for Black, but first check if game ended
        if environ.board.is_game_over() or not curr_state['legal_moves']:
            break # game is over, exit function
        else: # current game continues
            b_est_q_value: int = est_q_val_table.at[game_number, curr_turn_for_q_est]

        # training_functions_logger.info(f'b_est_q_value: {b_est_q_value}\n')
        # training_functions_logger.info(f'about to calc next q values\n')
        # training_functions_logger.info(f'w_curr_q_value: {w_curr_q_value}\n')
        # training_functions_logger.info(f'b_curr_q_value: {b_curr_q_value}\n')
        # training_functions_logger.info(f'w_reward: {w_reward}\n')
        # training_functions_logger.info(f'b_reward: {b_reward}\n')
        # training_functions_logger.info(f'w_est_q_value: {w_est_q_value}\n')
        # training_functions_logger.info(f'b_est_q_value: {b_est_q_value}\n\n')

        # ***CRITICAL STEP***, this is the main part of the SARSA algorithm.
        w_next_q_value: int = find_next_q_value(w_curr_q_value, w_agent.learn_rate, w_reward, w_agent.discount_factor, w_est_q_value)
        b_next_q_value: int = find_next_q_value(b_curr_q_value, b_agent.learn_rate, b_reward, b_agent.discount_factor, b_est_q_value)
    
        # training_functions_logger.info(f'sarsa calc complete\n')
        # training_functions_logger.info(f'w_next_q_value: {w_next_q_value}\n')
        # training_functions_logger.info(f'b_next_q_value: {b_next_q_value}\n')

        # on the next turn, w_next_q_value and b_next_q_value will be added to the q table. so if this is the end of the first round,
        # next round it will be W2 and then we assign the q value at W2 col
        w_curr_q_value = w_next_q_value
        b_curr_q_value = b_next_q_value

        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            # training_functions_logger.error(f'An error occurred: {e}\n')
            # training_functions_logger.error("failed to get_curr_state\n") 
            # training_functions_logger.error(f'At game: {game_number}\n')
            # training_functions_logger.error(f'at state: {curr_state}\n')
            break
    ### END OF CURRENT GAME LOOP ###

    # training_functions_logger.info(f'{game_number} is over.\n')
    # training_functions_logger.info(f'\nThe Chessboard looks like this:\n')
    # training_functions_logger.info(f'\n{environ.board}\n\n')
    # training_functions_logger.info(f'Game result is: {helper_methods.get_game_outcome(environ)}\n')    
    # training_functions_logger.info(f'The game ended because of: {helper_methods.get_game_termination_reason()}\n')
    # training_functions_logger.info(f'DB shows game ended b/c: {chess_data.at[game_number, "Result"]}\n')

    environ.reset_environ()
### end of train_one_game

def generate_q_est_df(chess_data, w_agent, b_agent) -> pd.DataFrame:
    """
        Generates a dataframe containing the estimated q-values for each chess move in the chess database.

        This method iterates over each game in the chess database and plays through the game using the reinforcement 
        learning agents. For each move, it calculates the estimated q-value and writes it to a file.

        The method first tries to get the current state of the game. If an error occurs, it logs the error and the 
        current board state in the errors file and moves on to the next game.

        The method then enters a loop where it alternates between the white and black agents choosing and playing 
        moves. If an error occurs while choosing or playing a move, the method logs the error and the current state 
        in the errors file and breaks out of the loop to move on to the next game.

        After each move, the method tries to get the latest state of the game. If an error occurs, it logs the error 
        and the current board state in the errors file and breaks out of the loop to move on to the next game.

        If the game is not over and there are still legal moves, the method tries to find the estimated q-value for 
        the current move and writes it to the file. If an error occurs while finding the estimated q-value, the 
        method logs the error and the current state in the errors file and breaks out of the loop to move on to the 
        next game.

        The loop continues until the game is over, there are no more legal moves, or the maximum number of moves for 
        the current training game has been reached.

        After each game, the method resets the environment to prepare for the next game.

        Args:
            chess_data (pd.DataFrame): A DataFrame containing the chess database.
        Returns:
            estimated_q_values (pd.DataFrame): A DataFrame containing the estimated q-values for each chess move.
    """
    estimated_q_values = chess_data.copy(deep = True)
    estimated_q_values = estimated_q_values.astype('int64')
    estimated_q_values.iloc[:, 1:] = 0

    for game_number in chess_data.index:
        generate_q_est_df_one_game(chess_data, game_number, w_agent, b_agent)
    
    # i will need to use a pool.Queue to collect all the values to be input to the q est table.
    return estimated_q_values
# end of generate_q_est_df

def generate_q_est_df_one_game(chess_data, game_number, w_agent, b_agent) -> None:
    num_chess_moves_curr_training_game: int = chess_data.at[game_number, 'PlyCount']
    environ = Environ.Environ()
    engine = start_chess_engine()
    
    try:
        curr_state = environ.get_curr_state()
    except Exception as e:
        # training_functions_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
        # training_functions_logger.error(f'at: {game_number}\n')
        break
    
    ### LOOP PLAYS THROUGH ONE GAME ###
    while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
        w_chess_move = w_agent.choose_action(chess_data, curr_state, game_number)
        if not w_chess_move:
            break

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        try:
            rl_agent_plays_move_during_training(w_chess_move, game_number, environ)
        except Exception as e:
            raise Exception from e
            break

        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            raise Exception from e
            break
        
        if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
            break
        else: # current game continues
            try:
                w_est_q_value: int = find_estimated_q_value(environ, engine)
            except Exception as e:
                raise Exception from e
                break

        b_chess_move = b_agent.choose_action(chess_data, curr_state, game_number)
        if not b_chess_move:
            break

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])
        
        try:
            rl_agent_plays_move_during_training(b_chess_move, game_number, environ)
        except Exception as e:
            break

        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            raise Exception from e
            break

        if environ.board.is_game_over() or not curr_state['legal_moves']:
            break
        else:
            try:
                b_est_q_val: int = find_estimated_q_value(environ, engine)
            except Exception as e:
                raise Exception from e
                break

        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            raise Exception from e
            break
    environ.reset_environ()
    engine.quit()
    
def find_estimated_q_value(environ, engine) -> int:
    try:
        anticipated_next_move = analyze_board_state(environ.board, engine)
    except Exception as e:
        raise Exception from e
    
    try:
        environ.load_chessboard_for_q_est(anticipated_next_move)
    except Exception as e:
        raise Exception from e
    
    # check if the game would be over with the anticipated next move, like unstopable checkmate.
    if environ.board.is_game_over() or not environ.get_legal_moves():
        try:
            environ.pop_chessboard()
        except Exception as e:
            raise Exception from e

    # this is the q estimated value due to what the opposing agent is likely to play in response to our move.    
    try:
        est_qval_analysis = analyze_board_state(environ.board, engine)
    except Exception as e:
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
        raise Exception from e

    return est_qval
# end of find_estimated_q_value

def find_next_q_value(curr_qval: int, learn_rate: float, reward: int, discount_factor: float, est_qval: int) -> int:
    try:
        next_qval = int(curr_qval + learn_rate * (reward + ((discount_factor * est_qval) - curr_qval)))
        return next_qval
    except OverflowError:
        raise custom_exceptions.QValueCalculationError("Overflow occurred during q-value calculation") from OverflowError
# end of find_next_q_value

def analyze_board_state(board, engine) -> dict:
    if not board.is_valid():
        raise custom_exceptions.InvalidBoardStateError(f'at Bradley.analyze_board_state. Board is in invalid state\n')

    try: 
        analysis_result = engine.analyse(board, game_settings.search_limit, multipv = game_settings.num_moves_to_return)
    except Exception as e:
        raise custom_exceptions.EngineAnalysisError("error occured during stockfish analysis") from e

    mate_score = None
    centipawn_score = None
    anticipated_next_move = None

    try:
        # Get score from analysis_result and normalize for player perspective
        pov_score = analysis_result[0]['score'].white() if board.turn == chess.WHITE else analysis_result[0]['score'].black()

        # Check if the score is a mate score and get the mate score, otherwise get the centipawn score
        if pov_score.is_mate():
            mate_score = pov_score.mate()
        else:
            centipawn_score = pov_score.score()
    except Exception as e:
        raise custom_exceptions.ScoreExtractionError("Error occurred while extracting scores from analysis") from e

    try:
        # Extract the anticipated next move from the analysis
        anticipated_next_move = analysis_result[0]['pv'][0]
    except Exception as e:
        raise custom_exceptions.MoveExtractionError("Error occurred while extracting the anticipated next move") from e
    
    return {
        'mate_score': mate_score,
        'centipawn_score': centipawn_score,
        'anticipated_next_move': anticipated_next_move
    }
### end of analyze_board_state

def rl_agent_plays_move_during_training(chess_move: str, game_number, environ: Environ.Environ) -> None:
    try:
        environ.load_chessboard(chess_move, game_number)
    except custom_exceptions.ChessboardLoadError as e:
        raise Exception from e

    try:
        environ.update_curr_state()
    except custom_exceptions.StateUpdateError as e:
        raise Exception from e
# end of rl_agent_plays_move_during_training

def get_reward(chess_move: str) -> int:
    if not chess_move or not isinstance(chess_move, str):
        raise ValueError("Invalid chess move input")    # <<<<<<<< !!!!!! put custom exception here later

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
    except Exception as e:
        raise Exception from e
# end of start_chess_engine


def assign_points_to_q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
    try:
        chess_agent.update_q_table([chess_move])
        chess_agent.change_q_table_pts(chess_move, curr_turn, curr_q_val)
    except custom_exceptions.QTableUpdateError as e: 
        raise Exception from e
# enf of assign_points_to_q_table    