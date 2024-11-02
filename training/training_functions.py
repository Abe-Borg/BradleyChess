from typing import Tuple
from agents import Agent
import chess
from utils import game_settings, constants
from environment import Environ
import pandas as pd
import copy
import re
from multiprocessing import Pool, cpu_count

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

def train_rl_agents(chess_data, est_q_val_table):
    """
        Trains the RL agents using the SARSA algorithm in parallel.

        Args:
            chess_data (pd.DataFrame): DataFrame containing chess games.
            est_q_val_table (pd.DataFrame): DataFrame containing estimated Q-values.

        Returns:
            w_agent (Agent): Trained white agent.
            b_agent (Agent): Trained black agent.
    """
    game_indices = list(chess_data.index)
    results = process_games_in_parallel(game_indices, worker_train_games, chess_data, est_q_val_table)

    # Collect and merge Q-tables from all processes
    w_agent_q_tables = [result[0] for result in results]
    b_agent_q_tables = [result[1] for result in results]

    w_agent = Agent.Agent('W')
    b_agent = Agent.Agent('B')

    w_agent.q_table = merge_q_tables(w_agent_q_tables)
    b_agent.q_table = merge_q_tables(b_agent_q_tables)

    w_agent.is_trained = True
    b_agent.is_trained = True

    return w_agent, b_agent
### end of train_rl_agents

def train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ, engine) -> None:
    num_moves: int = chess_data.at[game_number, 'PlyCount']
    curr_state = environ.get_curr_state()

    while curr_state['turn_index'] < num_moves:
        ##################### WHITE'S TURN ####################
        w_next_q_value, w_est_q_value = handle_agent_turn(
            agent=w_agent,
            chess_data=chess_data,
            curr_state=curr_state,
            game_number=game_number,
            environ=environ,
            engine=engine,
            curr_q_value=w_curr_q_value,
            est_q_val_table=est_q_val_table
        )
        w_curr_q_value = w_next_q_value
        curr_state = environ.get_curr_state()

        if environ.board.is_game_over():
            break

        ##################### BLACK'S TURN ####################
        b_next_q_value, b_est_q_value = handle_agent_turn(
            agent=b_agent,
            chess_data=chess_data,
            curr_state=curr_state,
            game_number=game_number,
            environ=environ,
            engine=engine,
            curr_q_value=b_curr_q_value,
            est_q_val_table=est_q_val_table
        )
        b_curr_q_value = b_next_q_value
        curr_state = environ.get_curr_state()

        if environ.board.is_game_over():
            break

### end of train_one_game

def generate_q_est_df(chess_data) -> pd.DataFrame:
    """
        Generates a DataFrame containing the estimated Q-values for each chess move.

        Args:
            chess_data (pd.DataFrame): DataFrame containing chess games.

        Returns:
            estimated_q_values (pd.DataFrame): DataFrame with estimated Q-values.
    """
    game_indices = list(chess_data.index)
    results = process_games_in_parallel(game_indices, worker_generate_q_est, chess_data)

    # Combine estimated Q-values from all processes
    estimated_q_values_list = results  # Each process returns a DataFrame

    estimated_q_values = pd.concat(estimated_q_values_list)
    estimated_q_values.sort_index(inplace=True)

    return estimated_q_values
# end of generate_q_est_df

def generate_q_est_df_one_game(chess_data, game_number, environ, engine) -> pd.DataFrame:
    """
        Generates the estimated Q-values for each move in a single game.

        Args:
            chess_data (pd.DataFrame): DataFrame containing chess games.
            game_number (str): The identifier for the current game.
            environ (Environ): The environment object.
            engine: The chess engine object.

        Returns:
            pd.DataFrame: Estimated Q-values for this game.
    """
    num_moves: int = chess_data.at[game_number, 'PlyCount']
    estimated_q_values_game = pd.DataFrame(index=[game_number], columns=chess_data.columns)
    estimated_q_values_game.iloc[0] = 0  # Initialize Q-values to zero

    curr_state = environ.get_curr_state()

    while curr_state['turn_index'] < num_moves:
        # White's turn
        curr_turn = curr_state['curr_turn']
        if curr_turn.startswith('W'):
            # Get move from chess_data
            w_chess_move = chess_data.at[game_number, curr_turn]
            apply_move_and_update_state(w_chess_move, game_number, environ)
            curr_state = environ.get_curr_state()

            # Estimate Q-value
            est_qval = find_estimated_q_value(environ, engine)
            estimated_q_values_game.at[game_number, curr_turn] = est_qval

        # Black's turn
        elif curr_turn.startswith('B'):
            b_chess_move = chess_data.at[game_number, curr_turn]
            apply_move_and_update_state(b_chess_move, game_number, environ)
            curr_state = environ.get_curr_state()

            # Estimate Q-value
            est_qval = find_estimated_q_value(environ, engine)
            estimated_q_values_game.at[game_number, curr_turn] = est_qval

        else:
            # Handle unexpected turn label
            training_functions_logger.error(f'Unexpected turn label: {curr_turn}')
            break

        # Check for game over
        if environ.board.is_game_over() or not curr_state['legal_moves']:
            break

    return estimated_q_values_game

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
        Side Effects:
            Temporarily modifies the state of the chessboard by loading and popping moves.
            Writes to the errors file if an error occurs.
    """
    # RL agent just played a move. the board has changed, if stockfish analyzes the board, 
    # it will give points for the agent, based on the agent's latest move.
    # We also need the points for the ANTICIPATED next state, 
    # given the ACTICIPATED next action. In this case, the anticipated response from opposing agent.
    anticipated_next_move = analyze_board_state(environ.board, engine)
    environ.load_chessboard_for_q_est(anticipated_next_move)
    
    if environ.board.is_game_over() or not environ.get_legal_moves():
        environ.board.pop()

    # this is the q estimated value due to what the opposing agent is likely to play in response to our move.    
    est_qval_analysis = analyze_board_state(environ.board, engine)

    if est_qval_analysis['mate_score'] is None:
        est_qval = est_qval_analysis['centipawn_score']
    else: # there is an impending checkmate
        est_qval = constants.CHESS_MOVE_VALUES['mate_score']

    # IMPORTANT STEP, pop the chessboard of last move, we are estimating board states, not
    # playing a move.
    environ.board.pop()
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
    """
    return int(curr_qval + learn_rate * (reward + ((discount_factor * est_qval) - curr_qval)))
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
    analysis_result = engine.analyse(board, game_settings.search_limit, multipv = constants.chess_engine_num_moves_to_return)
    mate_score = None
    centipawn_score = None
    anticipated_next_move = None

    # Get score from analysis_result and normalize for player perspective
    pov_score = analysis_result[0]['score'].white() if board.turn == chess.WHITE else analysis_result[0]['score'].black()
    if pov_score.is_mate():
        mate_score = pov_score.mate()
    else:
        centipawn_score = pov_score.score()

    anticipated_next_move = analysis_result[0]['pv'][0]
    return {
        'mate_score': mate_score,
        'centipawn_score': centipawn_score,
        'anticipated_next_move': anticipated_next_move
    }
### end of analyze_board_state

def apply_move_and_update_state(chess_move: str, game_number: str, environ) -> None:
    environ.board.push_san(chess_move)
    environ.update_curr_state()
# end of apply_move_and_update_state

def get_reward(chess_move: str) -> int:
    """
        calculates the reward for a given chess move by checking for specific patterns in the move string 
        that correspond to different types of moves. 
        Args:
            chess_move (str): A string representing the selected chess move in standard algebraic notation.
        Returns:
            int: The total reward for the given chess move, calculated based on the type of move.
    """
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
    chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    return chess_engine
# end of start_chess_engine

def assign_points_to_q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
    chess_agent.update_q_table([chess_move])
    chess_agent.change_q_table_pts(chess_move, curr_turn, curr_q_val)
# enf of assign_points_to_q_table 

def chunkify(lst, n):
    # utility function to split the game indices into chunks.
    return [lst[i::n] for i in range(n)]

def worker_train_games(game_indices_chunk, chess_data, est_q_val_table):
    """
        Worker function for training agents on a chunk of games.

        Args:
            game_indices_chunk (list): List of game indices for this process.
            chess_data (pd.DataFrame): DataFrame containing chess games.
            est_q_val_table (pd.DataFrame): DataFrame containing estimated Q-values.

        Returns:
            tuple: (w_agent.q_table, b_agent.q_table)
    """
    w_agent = Agent.Agent('W')
    b_agent = Agent.Agent('B')
    environ = Environ()
    engine = start_chess_engine()

    for game_number in game_indices_chunk:
        try:
            w_curr_q_value = constants.initial_q_val
            b_curr_q_value = constants.initial_q_val
            train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ, engine)
        except Exception as e:
            continue

        environ.reset_environ()
    engine.quit()
    return w_agent.q_table, b_agent.q_table

def worker_generate_q_est(game_indices_chunk, chess_data):
    """
        Worker function for generating estimated Q-values for a chunk of games.

        Args:
            game_indices_chunk (list): List of game indices for this process.
            chess_data (pd.DataFrame): DataFrame containing chess games.

        Returns:
            pd.DataFrame: Estimated Q-values for the processed games.
    """
    estimated_q_values_list = []

    environ = Environ()
    engine = start_chess_engine()

    for game_number in game_indices_chunk:
        try:
            estimated_q_values_game = generate_q_est_df_one_game(chess_data, game_number, environ, engine)
            estimated_q_values_list.append(estimated_q_values_game)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at generate_q_est_df_one_game: {e}\nat game: {game_number}')
            continue

        environ.reset_environ()

    engine.quit()
    # Combine estimated Q-values for this chunk
    estimated_q_values_chunk = pd.concat(estimated_q_values_list)
    return estimated_q_values_chunk

def merge_q_tables(q_tables_list):
    """
        Merges Q-tables from multiple processes, handling unique moves and duplicates.

        Args:
            q_tables_list (list): List of Q-tables to merge.

        Returns:
            pd.DataFrame: Merged Q-table.
    """
    merged_q_table = pd.concat(q_tables_list, axis=0)
    merged_q_table = merged_q_table.groupby(merged_q_table.index).sum(min_count=1)
    # Fill NaN values with zeros
    merged_q_table.fillna(0, inplace=True)
    return merged_q_table

def handle_agent_turn(agent, chess_data, curr_state, game_number, environ, engine, curr_q_value, est_q_val_table):
    """
    Handles a single agent's turn during training.

    Args:
        agent (Agent): The agent (white or black).
        chess_data (pd.DataFrame): DataFrame containing chess games.
        curr_state (dict): Current state of the environment.
        game_number (str): The identifier for the current game.
        environ (Environ): The environment object.
        engine: The chess engine object.
        curr_q_value (int): The current Q-value.
        est_q_val_table (pd.DataFrame): DataFrame containing estimated Q-values.

    Returns:
        tuple: (next_q_value, est_q_value)
    """
    curr_turn = curr_state['curr_turn']
    chess_move = agent.choose_action(chess_data, curr_state, game_number)
    if not chess_move:
        raise custom_exceptions.EmptyChessMoveError(f"{agent.color}_chess_move is empty at state: {curr_state}")

    # Assign current Q-value to Q-table
    assign_points_to_q_table(chess_move, curr_turn, curr_q_value, agent)

    # Apply move and update environment
    apply_move_and_update_state(chess_move, game_number, environ)
    reward = get_reward(chess_move)

    curr_state = environ.get_curr_state()

    # Check if game is over
    if environ.board.is_game_over() or not curr_state['legal_moves']:
        est_q_value = 0
    else:
        # Get estimated Q-value for next state
        next_turn = curr_state['curr_turn']
        est_q_value = est_q_val_table.at[game_number, next_turn]

    # SARSA update
    next_q_value = find_next_q_value(curr_q_value, agent.learn_rate, reward, agent.discount_factor, est_q_value)

    return next_q_value, est_q_value
