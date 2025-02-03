from typing import Callable
import pandas as pd
from agents import Agent
import chess
from utils import game_settings, constants
from environment.Environ import Environ
import re
from multiprocessing import Pool, cpu_count
import logging

# Set up file-based logging (critical items only)
logger = logging.getLogger("training_functions")
logger.setLevel(logging.CRITICAL)
if not logger.handlers:
    fh = logging.FileHandler(str(game_settings.training_functions_logger_filepath))
    fh.setLevel(logging.CRITICAL)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def process_games_in_parallel(game_indices: str, worker_function: Callable[..., pd.DataFrame], *args):
    num_processes = min(cpu_count(), len(game_indices))
    chunks = chunkify(game_indices, num_processes)
    
    logger.critical(f"Creating {len(chunks)} chunks for parallel processing")
    for i, chunk in enumerate(chunks):
        logger.critical(f"Chunk {i} size: {len(chunk)}")
    
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(worker_function, [(chunk, *args) for chunk in chunks])
    
    # Verify results
    logger.critical("Verifying results...")
    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, pd.DataFrame) or isinstance(result, tuple):
            valid_results.append(result)
            logger.critical(f"Result {i} is valid DataFrame with shape {result.shape}")
        else:
            logger.critical(f"Result {i} is invalid: {type(result)}")
            
    return valid_results

def train_rl_agents(chess_data, est_q_val_table, white_q_table, black_q_table):
    game_indices = list(chess_data.index)

    results = process_games_in_parallel(game_indices, worker_train_games, chess_data, est_q_val_table, white_q_table, black_q_table)
    
    w_agent_q_tables = [result[0] for result in results if isinstance(result, tuple)]
    b_agent_q_tables = [result[1] for result in results if isinstance(result, tuple)]
    w_agent = Agent('W', q_table=white_q_table.copy())
    b_agent = Agent('B', q_table=black_q_table.copy())
    w_agent.q_table = merge_q_tables(w_agent_q_tables)
    b_agent.q_table = merge_q_tables(b_agent_q_tables)
    w_agent.is_trained = True
    b_agent.is_trained = True
    return w_agent, b_agent

def train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ, engine):
    num_moves = chess_data.at[game_number, 'PlyCount']
    curr_state = environ.get_curr_state()
    while curr_state['turn_index'] < num_moves:
        try:
            w_next_q_value, w_est_q_value = handle_agent_turn(
                agent=w_agent,
                chess_data=chess_data,
                curr_state=curr_state,
                game_number=game_number,
                environ=environ,
                curr_q_value=w_curr_q_value,
                est_q_val_table=est_q_val_table
            )
            w_curr_q_value = w_next_q_value
            curr_state = environ.get_curr_state()
        except Exception as e:
            logger.critical(f'error during white agent turn in game {game_number}: {str(e)}')
            break

        if environ.board.is_game_over():
            break

        try:
            b_next_q_value, b_est_q_value = handle_agent_turn(
                agent=b_agent,
                chess_data=chess_data,
                curr_state=curr_state,
                game_number=game_number,
                environ=environ,
                curr_q_value=b_curr_q_value,
                est_q_val_table=est_q_val_table
            )

            b_curr_q_value = b_next_q_value
            curr_state = environ.get_curr_state()
        except Exception as e:
            logger.critical(f'error during black agent turn in game {game_number}: {str(e)}')
            break

        if environ.board.is_game_over():
            break

def generate_q_est_df(chess_data: pd.DataFrame) -> pd.DataFrame:
    logger.critical("\nInitial Validation:")
    logger.critical(f"Chess data shape: {chess_data.shape}")
    logger.critical(f"First few indices: {list(chess_data.index[:5])}")
    logger.critical(f"Index type: {type(chess_data.index[0])}")
    
    # Ensure all indices are strings and properly formatted
    game_indices = [str(idx) for idx in chess_data.index]
    logger.critical(f'Starting processing with {len(game_indices)} games')
    
    # Create master DataFrame with exact structure
    master_df = pd.DataFrame(
        index=chess_data.index,  # Use original index
        columns=chess_data.columns,
        dtype=object
    )

    master_df['PlyCount'] = chess_data['PlyCount']
    move_cols = [col for col in chess_data.columns if col.startswith(('W', 'B'))]
    master_df[move_cols] = 0.0
    
    num_processes = min(cpu_count(), len(game_indices))
    chunks = chunkify(game_indices, num_processes)
    logger.critical(f"\nCreated {len(chunks)} chunks")    

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            worker_generate_q_est, 
            [(chunk, chess_data.loc[chunk]) for chunk in chunks]
        )
        
    logger.critical(f'\nReceived {len(results)} results from parallel processing')   

    for chunk_df in results:
        if isinstance(chunk_df, pd.DataFrame):
            for game_idx in chunk_df.index:
                if game_idx in master_df.index:
                    master_df.loc[game_idx, move_cols] = chunk_df.loc[game_idx, move_cols]
    
    return master_df

def generate_q_est_df_one_game(chess_data, game_number, environ, engine) -> pd.DataFrame:
    game_df = pd.DataFrame(
        index=[game_number],
        columns=chess_data.columns,
        dtype=float
    )
    game_df.fillna(0.0, inplace=True)
    
    num_moves = chess_data.at[game_number, 'PlyCount']
    curr_state = environ.get_curr_state()
    moves_processed = 0
    
    while moves_processed < num_moves:
        curr_turn = curr_state['curr_turn']
        try:
            chess_move = chess_data.at[game_number, curr_turn]
            if pd.isna(chess_move):
                break
                
            if chess_move.endswith('#'):
                game_df.at[game_number, curr_turn] = constants.CHESS_MOVE_VALUES['mate_score']
                break
            
            try:
                apply_move_and_update_state(chess_move, environ)
                est_qval = find_estimated_q_value(environ, engine)
                game_df.at[game_number, curr_turn] = est_qval
                moves_processed += 1
                
                curr_state = environ.get_curr_state()
                if environ.board.is_game_over() or not curr_state['legal_moves']:
                    break
                    
            except chess.IllegalMoveError as e:
                logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}")
                game_df.at[game_number, curr_turn] = 0
                
        except Exception as e:
            logger.critical(f"Error processing game {game_number}, turn {curr_turn}: {str(e)}")
            break
            
    return game_df

def find_estimated_q_value(environ, engine) -> int:
    anticipated_next_move = analyze_board_state(environ.board, engine)
    environ.load_chessboard_for_q_est(anticipated_next_move)

    if environ.board.is_game_over() or not environ.get_legal_moves():
        environ.board.pop()

    est_qval_analysis = analyze_board_state(environ.board, engine)

    if est_qval_analysis['mate_score'] is None:
        est_qval = est_qval_analysis['centipawn_score']
    else:
        est_qval = constants.CHESS_MOVE_VALUES['mate_score']

    environ.board.pop()
    return est_qval

def find_next_q_value(curr_qval: int, learn_rate: float, reward: int, discount_factor: float, est_qval: int) -> int:
    return int(curr_qval + learn_rate * (reward + ((discount_factor * est_qval) - curr_qval)))

def analyze_board_state(board, engine) -> dict:
    analysis_result = engine.analyse(board, game_settings.search_limit, multipv = constants.chess_engine_num_moves_to_return)
    mate_score = None
    centipawn_score = None
    anticipated_next_move = None
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

def apply_move_and_update_state(chess_move: str, environ) -> None:
    environ.board.push_san(chess_move)
    environ.update_curr_state()

def get_reward(chess_move: str) -> int:
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
    chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    return chess_engine

def assign_points_to_q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
    chess_agent.update_q_table([chess_move])
    chess_agent.change_q_table_pts(chess_move, curr_turn, curr_q_val)

def chunkify(lst, n):
    size = len(lst) // n
    remainder = len(lst) % n
    chunks = []
    start = 0
    
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
        
    logger.critical("Chunk sizes: " + ", ".join([f"Chunk {i}: {len(chunk)} games" for i, chunk in enumerate(chunks)]))
    return chunks

def worker_train_games(game_indices_chunk, chess_data, est_q_val_table, white_q_table, black_q_table):
    w_agent = Agent('W', q_table=white_q_table.copy())
    b_agent = Agent('B', q_table=black_q_table.copy())
    environ = Environ()
    engine = start_chess_engine()

    for game_number in game_indices_chunk:
        try:
            w_curr_q_value = constants.initial_q_val
            b_curr_q_value = constants.initial_q_val
            train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ, engine)
        except Exception as e:
            logger.critical(f"Error processing game {game_number} in worker_train_games: {str(e)}")
            continue

        environ.reset_environ()
    engine.quit()
    return w_agent.q_table, b_agent.q_table

def worker_generate_q_est(game_indices_chunk, chunk_data):
    logger.critical(f"Starting worker for {len(game_indices_chunk)} games")
    
    chunk_df = pd.DataFrame(
        index=game_indices_chunk,
        columns=chunk_data.columns,
        dtype=object
    )

    chunk_df['PlyCount'] = chunk_data['PlyCount']
    move_cols = [col for col in chunk_data.columns if col.startswith(('W', 'B'))]
    chunk_df[move_cols] = 0.0
    
    environ = Environ()
    engine = start_chess_engine()
    
    try:
        for game_number in game_indices_chunk:
            try:
                # Process one game
                ply_count = chunk_data.at[game_number, 'PlyCount']
                curr_state = environ.get_curr_state()
                move_count = 0
                
                while move_count < ply_count:
                    curr_turn = curr_state['curr_turn']
                    chess_move = chunk_data.at[game_number, curr_turn]
                    
                    if isinstance(chess_move, str) and chess_move.endswith('#'):
                        chunk_df.at[game_number, curr_turn] = constants.CHESS_MOVE_VALUES['mate_score']
                        break
                        
                    try:
                        environ.board.push_san(chess_move)
                        environ.update_curr_state()
                        est_qval = find_estimated_q_value(environ, engine)
                        chunk_df.at[game_number, curr_turn] = est_qval
                        move_count += 1
                        curr_state = environ.get_curr_state()
                        
                        if environ.board.is_game_over():
                            break
                            
                    except chess.IllegalMoveError:
                        logger.critical(f"Invalid move '{chess_move}' in game {game_number}, turn {curr_turn}")
                        chunk_df.at[game_number, curr_turn] = 0
                        break
                        
            except Exception as e:
                logger.critical(f"Error processing game {game_number} in worker_generate_q_est: {str(e)}")
                continue
            finally:
                environ.reset_environ()
                
    finally:
        engine.quit()
        
    logger.critical(f"Completed chunk processing for {len(game_indices_chunk)} games")
    return chunk_df

def merge_q_tables(q_tables_list):
    merged_q_table = pd.concat(q_tables_list, axis=0)
    merged_q_table = merged_q_table.groupby(merged_q_table.index).sum(min_count=1)
    merged_q_table.fillna(0, inplace=True)
    return merged_q_table

def handle_agent_turn(agent, chess_data, curr_state, game_number, environ, curr_q_value, est_q_val_table):
    curr_turn = curr_state['curr_turn']
    chess_move = agent.choose_action(chess_data, curr_state, game_number)
    
    if chess_move not in environ.get_legal_moves():
        logger.critical(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}. Skipping.")
        return curr_q_value, 0

    apply_move_and_update_state(chess_move, environ)
    reward = get_reward(chess_move)
    curr_state = environ.get_curr_state()

    if environ.board.is_game_over() or not curr_state['legal_moves']:
        est_q_value = 0
    else:
        next_turn = curr_state['curr_turn']
        est_q_value = est_q_val_table.at[game_number, next_turn]
    
    next_q_value = find_next_q_value(curr_q_value, agent.learn_rate, reward, agent.discount_factor, est_q_value)
    agent.change_q_table_pts(chess_move, curr_turn, next_q_value - curr_q_value)
    return next_q_value, est_q_value

def validate_dataframe_alignment(chess_df: pd.DataFrame, q_est_df: pd.DataFrame) -> bool:
    """Validate that two DataFrames have identical structure"""
    
    if not chess_df.index.equals(q_est_df.index):
        print("Index mismatch between chess data and Q-value estimates")
        return False
        
    if not chess_df.columns.equals(q_est_df.columns):
        print("Column mismatch between chess data and Q-value estimates")
        print(f"Missing columns in Q-est: {set(chess_df.columns) - set(q_est_df.columns)}")
        print(f"Extra columns in Q-est: {set(q_est_df.columns) - set(chess_df.columns)}")
        return False
        
    # Check move columns specifically
    move_cols = [col for col in chess_df.columns if col.startswith(('W', 'B'))]
    for col in move_cols:
        if col not in q_est_df.columns:
            print(f"Missing move column in Q-value estimates: {col}")
            return False
            
    return True

def print_dataframe_differences(df1: pd.DataFrame, df2: pd.DataFrame):
    """Print detailed differences between two DataFrames"""
    print("\nDataFrame Comparison:")
    print(f"DF1 shape: {df1.shape}, DF2 shape: {df2.shape}")
    print(f"DF1 index: {df1.index[:5]}")
    print(f"DF2 index: {df2.index[:5]}")
    print(f"DF1 columns: {df1.columns[:5]}")
    print(f"DF2 columns: {df2.columns[:5]}")
    print(f"DF1 dtypes:\n{df1.dtypes[:5]}")
    print(f"DF2 dtypes:\n{df2.dtypes[:5]}")