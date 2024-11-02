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
        generate_q_est_df_one_game(chess_data, game_number, w_agent, b_agent)
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
        Side Effects:
            Modifies the estimated q-values DataFrame with the calculated q-values.
    """
    num_moves: int = chess_data.at[game_number, 'PlyCount']
    environ = Environ.Environ()
    engine = start_chess_engine()
    
    curr_state = environ.get_curr_state()
    
    ### LOOP PLAYS THROUGH ONE GAME ###
    while curr_state['turn_index'] < (num_moves):
        ##################### WHITE'S TURN ####################
        # choose action a from state s, using policy
        w_chess_move = w_agent.choose_action(chess_data, curr_state, game_number)
        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        ### WHITE AGENT PLAYS THE SELECTED MOVE ###
        # take action a, observe r, s', and load chessboard
        apply_move_and_update_state(w_chess_move, game_number, environ)

        # get latest curr_state since apply_move_and_update_state updated the chessboard
        curr_state = environ.get_curr_state()
        
        # check if game ended
        if environ.board.is_game_over() or curr_state['turn_index'] >= (num_moves) or not curr_state['legal_moves']:
            break # game is over, exit function.
        else: # current game continues
            w_est_q_value: int = find_estimated_q_value(environ, engine)

        ##################### BLACK'S TURN ####################
        # choose action a from state s, using policy
        b_chess_move = b_agent.choose_action(chess_data, curr_state, game_number)
        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])
        
        ##### BLACK AGENT PLAYS SELECTED MOVE #####
        # take action a, observe r, s', and load chessboard
        apply_move_and_update_state(b_chess_move, game_number, environ)

        # get latest curr_state since apply_move_and_update_state updated the chessboard
        curr_state = environ.get_curr_state()

        # check if game ended
        if environ.board.is_game_over() or curr_state['turn_index'] >= (num_moves) or not curr_state['legal_moves']:
            break # game is over, exit function.
        else: # current game continues
            b_est_q_value: int = find_estimated_q_value(environ, engine)
        
        curr_state = environ.get_curr_state()
    ### END OF CURRENT GAME LOOP ###
    environ.reset_environ()
    engine.quit()
# end of generate_q_est_df_one_game

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
            continue

    engine.quit()
    return w_agent.q_table, b_agent.q_table

def merge_q_tables(q_tables_list):
    merged_q_table = pd.concat(q_tables_list, axis = 0)
    merged_q_table = merged_q_table.groupby(merged_q_table.index).sum(min_count = 1)
    merged_q_table.fillna(0, inplace = True)
    return merged_q_table
