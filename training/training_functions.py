from typing import Tuple
from agents import Agent
import chess
from utils import game_settings, custom_exceptions, constants
from environment import Environ
import pandas as pd
import copy
import re
from utils.logging_config import setup_logger
from multiprocessing import Pool, cpu_count

training_functions_logger = setup_logger(__name__, game_settings.training_functions_logger_filepath)

def process_games_in_parallel(game_indices, worker_function, *args):
    """
    Processes games in parallel using the specified worker function.
    Args:
        game_indices (list): List of game indices to process.
        worker_function (callable): The function to execute in parallel.
        *args: Additional arguments to pass to the worker function.
    Returns:
        list: Results from each process.
    """
    num_processes = min(cpu_count(), len(game_indices))  # Avoid more processes than games
    chunks = chunkify(game_indices, num_processes)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_function, [(chunk, *args) for chunk in chunks])
    return results

def train_rl_agents(chess_data, est_q_val_table, w_agent, b_agent):
    """
        Trains the RL agents using the SARSA algorithm         
        Args:
            chess_data (pd.DataFrame): A DataFrame containing the chess database.
            est_q_val_table (pd.DataFrame): A DataFrame containing the estimated 
            q values for each game in the training set.
            w_agent: The white agent.
            b_agent: The black agent.
        Returns:
            Tuple[Agent, Agent]: A tuple containing the trained white and black agents.
    """
    num_processes = cpu_count()
    game_indices = list(chess_data.index)
    chunks = chunkify(game_indices, num_processes)

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_train_games, [(chunk, chess_data, est_q_val_table) for chunk in chunks])

    # Merge Q-tables from all processes
    w_agent_q_tables = [result[0] for result in results]
    b_agent_q_tables = [result[1] for result in results]

    w_agent.q_table = merge_q_tables(w_agent_q_tables)
    b_agent.q_table = merge_q_tables(b_agent_q_tables)

    w_agent.is_trained = True
    b_agent.is_trained = True

    return w_agent, b_agent
### end of train_rl_agents

def train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ, engine) -> None:
    num_moves: int = chess_data.at[game_number, 'PlyCount']
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
            # curr_turn_for_q_est is here because we previously moved to next turn (after move was played)
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
### end of train_one_game

def generate_q_est_df(chess_data, w_agent, b_agent) -> pd.DataFrame:
    """
        Generates a dataframe containing the estimated q-values for each chess move in the chess database.
        This method iterates over each game in the chess database and plays through the game using the reinforcement 
        learning agents. For each move, it calculates the estimated q-value.
        After each move, the method tries to get the latest state of the game.         
        The loop continues until the game is over, there are no more legal moves, or the maximum number of moves for 
        the current training game has been reached.
        Args:
            chess_data (pd.DataFrame): A DataFrame containing the chess database.
            w_agent: The white agent.
            b_agent: The black agent.
        Returns:
            estimated_q_values (pd.DataFrame): A DataFrame containing the estimated q-values for each chess move.
    """
    # Create a copy of the chess data to store the estimated q-values
    # the cells will be reassigned to be ints instead of strings.
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
    """
        Generates the estimated q-values for each chess move in a single game.
        This method plays through a single game in the chess database using the reinforcement learning agents.
        For each move, it calculates the estimated q-value based on the current state of the board.
        The method then updates the estimated q-values DataFrame with the calculated q-values.
        The loop continues until the game is over, there are no more legal moves, or the maximum number of moves
        for the current training game has been reached.
        Args:
            chess_data (pd.DataFrame): A DataFrame containing the chess database.
            game_number (int): The index of the game in the chess database.
            w_agent: The white agent.
            b_agent: The black agent.
        Raises:
            Exception: An exception is raised if an error occurs during the training process.
        Side Effects:
            Modifies the estimated q-values DataFrame with the calculated q-values.
    """
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
# end of generate_q_est_df_one_game

# def continue_training_rl_agents
#     ### placeholder, will implement this function later.
# ### end of continue_training_rl_agents

def find_estimated_q_value(environ, engine) -> int:
    """
        Estimates the q-value for the RL agent's next action without actually playing the move.
        This method simulates the agent's next action and the anticipated response from the opposing agent 
        to estimate the q-value. The steps are as follows:

        1. Observes the next state of the chessboard after the agent's move.
        2. Analyzes the current state of the board to predict the opposing agent's response.
        3. Loads the board with the anticipated move of the opposing agent.
        4. Estimates the q-value based on the anticipated state of the board.

        The estimation of the q-value is derived from analyzing the board state with the help of a chess engine 
        (like Stockfish). If there's no impending checkmate, the estimated q-value is the centipawn score of 
        the board state. Otherwise, it's computed based on the impending checkmate turns multiplied by a predefined 
        mate score reward.

        After estimating the q-value, the method reverts the board state to its original state before the simulation.

        Returns:
            int: The estimated q-value for the agent's next action.
        Raises:
            BoardAnalysisError: An exception is raised if an error occurs while analyzing the board state for the estimated q-value
            ChessboardManipulationError: if an error occurs loading the chessboard, popping the chessboard.

        Side Effects:
            Temporarily modifies the state of the chessboard by loading and popping moves.
            Writes to the errors file if an error occurs.
    """
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
    
    try:
        environ.load_chessboard_for_q_est(anticipated_next_move)
    except custom_exceptions.ChessboardLoadError as e:
        training_functions_logger.error(f'at find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error(f'failed to load_chessboard_for_q_est\n')
        raise Exception from e
    
    if environ.board.is_game_over() or not environ.get_legal_moves():
        try:
            environ.board.pop()
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

    if est_qval_analysis['mate_score'] is None:
        est_qval = est_qval_analysis['centipawn_score']
    else: # there is an impending checkmate
        est_qval = constants.CHESS_MOVE_VALUES['mate_score']

    # IMPORTANT STEP, pop the chessboard of last move, we are estimating board states, not
    # playing a move.
    try:
        environ.board.pop()
    except Exception as e:
        training_functions_logger.error(f'@ find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error("failed to pop_chessboard after est q val analysis values found\n")
        raise Exception from e

    return est_qval
# end of find_estimated_q_value

def find_next_q_value(curr_qval: int, learn_rate: float, reward: int, discount_factor: float, est_qval: int) -> int:
    """
        calculates the next q-value based on the current q-value, the learning rate, the reward, the 
        discount factor, and the estimated q-value for the next state-action pair. 
            next_qval = curr_qval + learn_rate * (reward + (discount_factor * est_qval) - curr_qval)
        This formula is derived from the SARSA algorithm, which is a model-free reinforcement learning method used 
        to estimate the q-values for state-action pairs in an environment.
        Args:
            curr_qval (int): The current q-value for the state-action pair.
            learn_rate (float): The learning rate, a value between 0 and 1. This parameter controls how much the 
            q-value is updated on each iteration.
            reward (int): The reward obtained from the current action.
            discount_factor (float): The discount factor, a value between 0 and 1. This parameter determines the 
            importance of future rewards.
            est_qval (int): The estimated q-value for the next state-action pair.
        Returns:
            int: The next q-value, calculated using the SARSA algorithm.
        Raises:
            QValueCalculationError: If an error or overflow occurs during the calculation of the next q-value.
    """
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
    """
        Analyzes the current state of the chessboard using the Stockfish engine and returns the analysis results.
        This method uses the Stockfish engine to analyze the current state of the chessboard. The analysis results 
        include the mate score, the centipawn score, and the anticipated next move.
        Args:
            board (chess.Board): The current state of the chessboard to analyze.
            engine (Stockfish): The Stockfish engine used to analyze the board.
        Returns:
            dict: A dictionary containing the analysis results:
            - 'mate_score': Number of moves to mate (None if not a mate position)
            - 'centipawn_score': Centipawn score (None if mate position)
            - 'anticipated_next_move': The best move suggested by the engine
        Raises:
            InvalidBoardStateError: If the board is in an invalid state.
            EngineAnalysisError: If an error occurs during the Stockfish analysis.
            ScoreExtractionError: If an error occurs while extracting scores from the analysis.
            MoveExtractionError: If an error occurs while extracting the anticipated next move.

        Side Effects:
            Writes to the errors file if an error occurs.
    """
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
### end of analyze_board_state

def apply_move_and_update_state(chess_move: str, game_number: str, environ) -> None:
    """
        Loads the chessboard with the given move and updates the current state of the environment.
        This method is used during training. 
        Args: 
            chess_move (str): A string representing the chess move in standard algebraic notation.
            game_number (str): The current game being played during training.
            environ (Environ): The environment object representing the current state of the game.
        Raises:
            Exception: An exception is raised if an error occurs while loading the chessboard or updating the 
            current state.
        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
            Writes to the errors file if an error occurs.
    """
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
    """
        calculates the reward for a given chess move by checking for specific patterns in the move string 
        that correspond to different types of moves. 
        Args:
            chess_move (str): A string representing the selected chess move in standard algebraic notation.
        Returns:
            int: The total reward for the given chess move, calculated based on the type of move.
        Raises:
            ValueError: If the chess_move string is empty or invalid.
    """
    if not chess_move or not isinstance(chess_move, str):
        raise custom_exceptions.RewardCalculationError("Invalid chess move input")

    total_reward = 0
    # Check for piece development (N, R, B, Q)
    if re.search(r'[NRBQ]', chess_move):
        total_reward += constants.CHESS_MOVE_VALUES['piece_development']
    # Check for capture
    if 'x' in chess_move:
        total_reward += constants.CHESS_MOVE_VALUES['capture']
    # Check for promotion (with additional reward for queen promotion)
    if '=' in chess_move:
        total_reward += constants.CHESS_MOVE_VALUES['promotion']
        if '=Q' in chess_move:
            total_reward += constants.CHESS_MOVE_VALUES['promotion_queen']
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
    """
        Assigns points to the q table for the given chess move, current turn, current q value, and RL agent color.
        Args:
            chess_move (str): The chess move to assign points to in the q table.
            curr_turn (str): The current turn of the game.
            curr_qval (int): The current q value for the given chess move.
            chess_agent: The RL agent to assign points to in the q table.
        Raises:
            QTableUpdateError: is raised if the chess move is not represented in the q table. The exception is 
            written to the errors file.
        Side Effects:
            Modifies the q table of the RL agent by assigning points to the given chess move.
            Writes to the errors file if a exception is raised.
    """
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

def chunkify(lst, n):
    # utility function to split the game indices into chunks.
    return [lst[i::n] for i in range(n)]

def worker_train_games(game_indices_chunk, chess_data, est_q_val_table):
    # Each process will run this function to train on its chunk of games.
    w_curr_q_value: int = copy.copy(constants.initial_q_val)
    b_curr_q_value: int = copy.copy(constants.initial_q_val)
    w_agent = Agent.Agent('W')
    b_agent = Agent.Agent('B')
    environ = Environ.Environ()
    engine = start_chess_engine()

    for game_number in game_indices_chunk:
        try:
            train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ, engine)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at train_one_game: {e}\nat game: {game_number}')
            continue

    engine.quit()
    return w_agent.q_table, b_agent.q_table

def merge_q_tables(q_tables_list):
    """
        Merges Q-tables from multiple processes, handling unique moves and duplicates.
        Args:
            q_tables_list (list): List of Q-tables to merge.
        Returns:
            pd.DataFrame: Merged Q-table.
    """
    merged_q_table = pd.concat(q_tables_list, axis = 0)
    merged_q_table = merged_q_table.groupby(merged_q_table.index).sum(min_count = 1)
    merged_q_table.fillna(0, inplace = True)
    return merged_q_table
