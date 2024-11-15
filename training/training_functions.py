import pandas as pd
from agents import Agent
import chess
from utils import game_settings, constants
from environment.Environ import Environ
import re
from multiprocessing import Pool, cpu_count

def process_games_in_parallel(game_indices, worker_function, *args):
    num_processes = min(cpu_count(), len(game_indices))
    chunks = chunkify(game_indices, num_processes)

    # Track which chunks were processed
    processed_chunks = set()

    with Pool(processes=num_processes) as pool:
        try:
            results = pool.starmap(worker_function, [(chunk, *args) for chunk in chunks])
            for i, result in enumerate(results):
                processed_chunks.add(i)
        except Exception as e:
            print(f"Error in parallel processing: {str(e)}")
            # Return processed results so far
            return results, len(processed_chunks) == len(chunks)
            
    return results, True

def train_rl_agents(chess_data, est_q_val_table, white_q_table, black_q_table):
    game_indices = list(chess_data.index)
    results = process_games_in_parallel(game_indices, worker_train_games, chess_data, est_q_val_table, white_q_table, black_q_table)
    w_agent_q_tables = [result[0] for result in results]
    b_agent_q_tables = [result[1] for result in results]
    w_agent = Agent('W')
    b_agent = Agent('B')
    w_agent.q_table = merge_q_tables(w_agent_q_tables)
    b_agent.q_table = merge_q_tables(b_agent_q_tables)
    w_agent.is_trained = True
    b_agent.is_trained = True
    return w_agent, b_agent

def train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value, environ, engine):
    num_moves = chess_data.at[game_number, 'PlyCount']
    curr_state = environ.get_curr_state()
    while curr_state['turn_index'] < num_moves:
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

def generate_q_est_df(chess_data: pd.DataFrame) -> pd.DataFrame:
    game_indices = list(chess_data.index)
    print(f'Starting processing with {len(game_indices)} games')
    
    results, is_complete = process_games_in_parallel(game_indices, worker_generate_q_est, chess_data)
    
    if not results:
        raise ValueError("No results generated during processing")
        
    # Flatten results maintaining order
    estimated_q_values_list = []
    for chunk_result in results:
        estimated_q_values_list.extend(chunk_result)
        
    # Concatenate while preserving order
    estimated_q_values = pd.concat(estimated_q_values_list)
    estimated_q_values.sort_index(inplace=True)
    
    # Validate structure
    if not validate_dataframe_alignment(chess_data, estimated_q_values):
        print("Warning: DataFrames do not match in structure")
        raise ValueError("Generated Q-value estimates do not match chess data structure")
        
    # Check for completeness
    if not is_complete:
        print("Warning: Some games may not have been processed")
        
    return estimated_q_values

def generate_q_est_df_one_game(chess_data, game_number, environ, engine) -> pd.DataFrame:
    num_moves = chess_data.at[game_number, 'PlyCount']
    # move_columns = [col for col in chess_data.columns if re.match(r'^[WB]\d+$', col)]
    estimated_q_values_game = pd.DataFrame(
        index=[game_number], columns=chess_data.columns,
        dtype=float)

    estimated_q_values_game.iloc[0] = float('nan')
    
    curr_state = environ.get_curr_state()
    moves_processed = 0

    while moves_processed < num_moves:  
        curr_turn = curr_state['curr_turn']
    
        # if curr_turn not in move_columns:
        #     print(f'invalid turn identifier: {curr_turn} for game {game_number}, skipping.')
        #     break
        try:    
            chess_move = chess_data.at[game_number, curr_turn]

            # Handle possible NaN or missing moves
            if pd.isna(chess_move) or not isinstance(chess_move, str):
                print(f"Missing or invalid move '{chess_move}' for turn '{curr_turn}' in game {game_number}. Skipping turn.")
                environ.update_curr_state()
                continue

            apply_move_and_update_state(chess_move, game_number, environ)
            
            if moves_processed == num_moves - 1:
                if environ.board.is_checkmate():
                    est_qval = constants.CHESS_MOVE_VALUES['mate_score']
                else:
                    est_qval = find_estimated_q_value(environ, engine)
            else:
                est_qval = find_estimated_q_value(environ, engine)

            estimated_q_values_game.at[game_number, curr_turn] = est_qval
            moves_processed += 1
            curr_state = environ.get_curr_state()

            if environ.board.is_game_over() or not curr_state['legal_moves']:
                break
        except chess.IllegalMoveError:
            print(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}. Skipping.")
            environ.update_curr_state()
            continue
    return estimated_q_values_game

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

def apply_move_and_update_state(chess_move: str, game_number: str, environ) -> None:
    environ.board.push_san(chess_move)
    environ.update_curr_state()

def get_reward(chess_move: str) -> int:
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

def start_chess_engine(): 
    chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    return chess_engine

def assign_points_to_q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
    chess_agent.update_q_table([chess_move])
    chess_agent.change_q_table_pts(chess_move, curr_turn, curr_q_val)

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

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
            continue
        environ.reset_environ()
    engine.quit()
    return w_agent.q_table, b_agent.q_table

def worker_generate_q_est(game_indices_chunk, chess_data):
    estimated_q_values_list = []
    environ = Environ()
    engine = start_chess_engine()
    for game_number in game_indices_chunk:
        try:
            estimated_q_values_game = generate_q_est_df_one_game(chess_data, game_number, environ, engine)
            estimated_q_values_list.append(estimated_q_values_game)
        except Exception as e:
            print(f"Error processing game {game_number}: {str(e)}")
            continue
        environ.reset_environ()
    engine.quit()
    return estimated_q_values_list

def merge_q_tables(q_tables_list):
    merged_q_table = pd.concat(q_tables_list, axis=0)
    merged_q_table = merged_q_table.groupby(merged_q_table.index).sum(min_count=1)
    merged_q_table.fillna(0, inplace=True)
    return merged_q_table

def handle_agent_turn(agent, chess_data, curr_state, game_number, environ, engine, curr_q_value, est_q_val_table):
    curr_turn = curr_state['curr_turn']
    chess_move = agent.choose_action(chess_data, curr_state, game_number)
    
    if chess_move not in environ.get_legal_moves():
        print(f"Invalid move '{chess_move}' for game {game_number}, turn {curr_turn}. Skipping.")
        return curr_q_value, 0

    apply_move_and_update_state(chess_move, game_number, environ)
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
    
    # Check index alignment
    if not chess_df.index.equals(q_est_df.index):
        print("Index mismatch between chess data and Q-value estimates")
        return False
        
    # Check column alignment
    if not chess_df.columns.equals(q_est_df.columns):
        print("Column mismatch between chess data and Q-value estimates")
        return False
        
    # Check move columns specifically
    move_cols = [col for col in chess_df.columns if col.startswith(('W', 'B'))]
    for col in move_cols:
        if col not in q_est_df.columns:
            print(f"Missing move column in Q-value estimates: {col}")
            return False
            
    return True