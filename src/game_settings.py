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

initial_q_val = 50 # this is relevant when first training an agent. SARSA algorithm requires an initial value
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

helper_methods_errors_filepath = base_directory / ".." / "debug" / "helper_methods_errors_log.txt"
agent_errors_filepath = base_directory / ".." / "debug" / "agent_errors_log.txt"
bradley_errors_filepath = base_directory / ".." / "debug" / "bradley_errors_log.txt"
environ_errors_filepath = base_directory / ".." / "debug" / "environ_errors_log.txt"

agent_step_by_step_filepath = base_directory / ".." / "debug" / "agent_step_by_step_log.txt"
bradley_step_by_step_filepath = base_directory / ".." / "debug" / "bradley_step_by_step_log.txt"
environ_step_by_step_filepath = base_directory / ".." / "debug" / "environ_step_by_step_log.txt"

initial_training_results_filepath = base_directory / ".." / "training_results" / "initial_training_results.txt"
additional_training_results_filepath = base_directory / ".." / "training_results" / "additional_training_results.txt"
agent_vs_agent_filepath = base_directory / ".." / "training_results" / "agent_vs_agent_games.txt"

est_q_vals_filepath_part_1 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_1.txt"
est_q_vals_filepath_part_2 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_2.txt"
est_q_vals_filepath_part_3 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_3.txt"
est_q_vals_filepath_part_4 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_4.txt"
est_q_vals_filepath_part_5 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_5.txt"
est_q_vals_filepath_part_6 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_6.txt"
est_q_vals_filepath_part_7 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_7.txt"
est_q_vals_filepath_part_8 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_8.txt"
est_q_vals_filepath_part_9 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_9.txt"
est_q_vals_filepath_part_10 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_10.txt"

est_q_vals_filepath_part_11 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_11.txt"
est_q_vals_filepath_part_12 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_12.txt"
est_q_vals_filepath_part_13 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_13.txt"
est_q_vals_filepath_part_14 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_14.txt"
est_q_vals_filepath_part_15 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_15.txt"
est_q_vals_filepath_part_16 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_16.txt"
est_q_vals_filepath_part_17 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_17.txt"
est_q_vals_filepath_part_18 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_18.txt"
est_q_vals_filepath_part_19 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_19.txt"
est_q_vals_filepath_part_20 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_20.txt"

est_q_vals_filepath_part_21 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_21.txt"
est_q_vals_filepath_part_22 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_22.txt"
est_q_vals_filepath_part_23 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_23.txt"
est_q_vals_filepath_part_24 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_24.txt"
est_q_vals_filepath_part_25 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_25.txt"
est_q_vals_filepath_part_26 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_26.txt"
est_q_vals_filepath_part_27 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_27.txt"
est_q_vals_filepath_part_28 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_28.txt"
est_q_vals_filepath_part_29 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_29.txt"
est_q_vals_filepath_part_30 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_30.txt"

est_q_vals_filepath_part_31 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_31.txt"
est_q_vals_filepath_part_32 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_32.txt"
est_q_vals_filepath_part_33 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_33.txt"
est_q_vals_filepath_part_34 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_34.txt"
est_q_vals_filepath_part_35 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_35.txt"
est_q_vals_filepath_part_36 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_36.txt"
est_q_vals_filepath_part_37 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_37.txt"
est_q_vals_filepath_part_38 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_38.txt"
est_q_vals_filepath_part_39 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_39.txt"
est_q_vals_filepath_part_40 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_40.txt"

est_q_vals_filepath_part_41 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_41.txt"
est_q_vals_filepath_part_42 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_42.txt"
est_q_vals_filepath_part_43 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_43.txt"
est_q_vals_filepath_part_44 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_44.txt"
est_q_vals_filepath_part_45 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_45.txt"
est_q_vals_filepath_part_46 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_46.txt"
est_q_vals_filepath_part_47 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_47.txt"
est_q_vals_filepath_part_48 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_48.txt"
est_q_vals_filepath_part_49 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_49.txt"
est_q_vals_filepath_part_50 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_50.txt"

est_q_vals_filepath_part_51 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_51.txt"
est_q_vals_filepath_part_52 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_52.txt"
est_q_vals_filepath_part_53 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_53.txt"
est_q_vals_filepath_part_54 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_54.txt"
est_q_vals_filepath_part_55 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_55.txt"
est_q_vals_filepath_part_56 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_56.txt"
est_q_vals_filepath_part_57 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_57.txt"
est_q_vals_filepath_part_58 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_58.txt"
est_q_vals_filepath_part_59 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_59.txt"
est_q_vals_filepath_part_60 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_60.txt"

est_q_vals_filepath_part_61 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_61.txt"
est_q_vals_filepath_part_62 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_62.txt"
est_q_vals_filepath_part_63 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_63.txt"
est_q_vals_filepath_part_64 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_64.txt"
est_q_vals_filepath_part_65 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_65.txt"
est_q_vals_filepath_part_66 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_66.txt"
est_q_vals_filepath_part_67 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_67.txt"
est_q_vals_filepath_part_68 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_68.txt"
est_q_vals_filepath_part_69 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_69.txt"
est_q_vals_filepath_part_70 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_70.txt"

est_q_vals_filepath_part_71 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_71.txt"
est_q_vals_filepath_part_72 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_72.txt"
est_q_vals_filepath_part_73 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_73.txt"
est_q_vals_filepath_part_74 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_74.txt"
est_q_vals_filepath_part_75 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_75.txt"
est_q_vals_filepath_part_76 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_76.txt"
est_q_vals_filepath_part_77 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_77.txt"
est_q_vals_filepath_part_78 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_78.txt"
est_q_vals_filepath_part_79 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_79.txt"
est_q_vals_filepath_part_80 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_80.txt"

est_q_vals_filepath_part_81 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_81.txt"
est_q_vals_filepath_part_82 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_82.txt"
est_q_vals_filepath_part_83 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_83.txt"
est_q_vals_filepath_part_84 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_84.txt"
est_q_vals_filepath_part_85 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_85.txt"
est_q_vals_filepath_part_86 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_86.txt"
est_q_vals_filepath_part_87 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_87.txt"
est_q_vals_filepath_part_88 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_88.txt"
est_q_vals_filepath_part_89 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_89.txt"
est_q_vals_filepath_part_90 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_90.txt"

est_q_vals_filepath_part_91 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_91.txt"
est_q_vals_filepath_part_92 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_92.txt"
est_q_vals_filepath_part_93 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_93.txt"
est_q_vals_filepath_part_94 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_94.txt"
est_q_vals_filepath_part_95 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_95.txt"
est_q_vals_filepath_part_96 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_96.txt"
est_q_vals_filepath_part_97 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_97.txt"
est_q_vals_filepath_part_98 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_98.txt"
est_q_vals_filepath_part_99 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_99.txt"
est_q_vals_filepath_part_100 = base_directory / ".." / "Q_Tables" / "Estimated_Q_Values" / "est_q_vals_part_100.txt"

chess_games_filepath_part_1 = base_directory / ".." / "chess_data" / "chess_games_part_1.pkl"
chess_games_filepath_part_2 = base_directory / ".." / "chess_data" / "chess_games_part_2.pkl"
chess_games_filepath_part_3 = base_directory / ".." / "chess_data" / "chess_games_part_3.pkl"
chess_games_filepath_part_4 = base_directory / ".." / "chess_data" / "chess_games_part_4.pkl"
chess_games_filepath_part_5 = base_directory / ".." / "chess_data" / "chess_games_part_5.pkl"
chess_games_filepath_part_6 = base_directory / ".." / "chess_data" / "chess_games_part_6.pkl"
chess_games_filepath_part_7 = base_directory / ".." / "chess_data" / "chess_games_part_7.pkl"
chess_games_filepath_part_8 = base_directory / ".." / "chess_data" / "chess_games_part_8.pkl"
chess_games_filepath_part_9 = base_directory / ".." / "chess_data" / "chess_games_part_9.pkl"
chess_games_filepath_part_10 = base_directory / ".." / "chess_data" / "chess_games_part_10.pkl"

chess_games_filepath_part_11 = base_directory / ".." / "chess_data" / "chess_games_part_11.pkl"
chess_games_filepath_part_12 = base_directory / ".." / "chess_data" / "chess_games_part_12.pkl"
chess_games_filepath_part_13 = base_directory / ".." / "chess_data" / "chess_games_part_13.pkl"
chess_games_filepath_part_14 = base_directory / ".." / "chess_data" / "chess_games_part_14.pkl"
chess_games_filepath_part_15 = base_directory / ".." / "chess_data" / "chess_games_part_15.pkl"
chess_games_filepath_part_16 = base_directory / ".." / "chess_data" / "chess_games_part_16.pkl"
chess_games_filepath_part_17 = base_directory / ".." / "chess_data" / "chess_games_part_17.pkl"
chess_games_filepath_part_18 = base_directory / ".." / "chess_data" / "chess_games_part_18.pkl"
chess_games_filepath_part_19 = base_directory / ".." / "chess_data" / "chess_games_part_19.pkl"
chess_games_filepath_part_20 = base_directory / ".." / "chess_data" / "chess_games_part_20.pkl"

chess_games_filepath_part_21 = base_directory / ".." / "chess_data" / "chess_games_part_21.pkl"
chess_games_filepath_part_22 = base_directory / ".." / "chess_data" / "chess_games_part_22.pkl"
chess_games_filepath_part_23 = base_directory / ".." / "chess_data" / "chess_games_part_23.pkl"
chess_games_filepath_part_24 = base_directory / ".." / "chess_data" / "chess_games_part_24.pkl"
chess_games_filepath_part_25 = base_directory / ".." / "chess_data" / "chess_games_part_25.pkl"
chess_games_filepath_part_26 = base_directory / ".." / "chess_data" / "chess_games_part_26.pkl"
chess_games_filepath_part_27 = base_directory / ".." / "chess_data" / "chess_games_part_27.pkl"
chess_games_filepath_part_28 = base_directory / ".." / "chess_data" / "chess_games_part_28.pkl"
chess_games_filepath_part_29 = base_directory / ".." / "chess_data" / "chess_games_part_29.pkl"
chess_games_filepath_part_30 = base_directory / ".." / "chess_data" / "chess_games_part_30.pkl"

chess_games_filepath_part_31 = base_directory / ".." / "chess_data" / "chess_games_part_31.pkl"
chess_games_filepath_part_32 = base_directory / ".." / "chess_data" / "chess_games_part_32.pkl"
chess_games_filepath_part_33 = base_directory / ".." / "chess_data" / "chess_games_part_33.pkl"
chess_games_filepath_part_34 = base_directory / ".." / "chess_data" / "chess_games_part_34.pkl"
chess_games_filepath_part_35 = base_directory / ".." / "chess_data" / "chess_games_part_35.pkl"
chess_games_filepath_part_36 = base_directory / ".." / "chess_data" / "chess_games_part_36.pkl"
chess_games_filepath_part_37 = base_directory / ".." / "chess_data" / "chess_games_part_37.pkl"
chess_games_filepath_part_38 = base_directory / ".." / "chess_data" / "chess_games_part_38.pkl"
chess_games_filepath_part_39 = base_directory / ".." / "chess_data" / "chess_games_part_39.pkl"
chess_games_filepath_part_40 = base_directory / ".." / "chess_data" / "chess_games_part_40.pkl"

chess_games_filepath_part_41 = base_directory / ".." / "chess_data" / "chess_games_part_41.pkl"
chess_games_filepath_part_42 = base_directory / ".." / "chess_data" / "chess_games_part_42.pkl"
chess_games_filepath_part_43 = base_directory / ".." / "chess_data" / "chess_games_part_43.pkl"
chess_games_filepath_part_44 = base_directory / ".." / "chess_data" / "chess_games_part_44.pkl"
chess_games_filepath_part_45 = base_directory / ".." / "chess_data" / "chess_games_part_45.pkl"
chess_games_filepath_part_46 = base_directory / ".." / "chess_data" / "chess_games_part_46.pkl"
chess_games_filepath_part_47 = base_directory / ".." / "chess_data" / "chess_games_part_47.pkl"
chess_games_filepath_part_48 = base_directory / ".." / "chess_data" / "chess_games_part_48.pkl"
chess_games_filepath_part_49 = base_directory / ".." / "chess_data" / "chess_games_part_49.pkl"
chess_games_filepath_part_50 = base_directory / ".." / "chess_data" / "chess_games_part_50.pkl"

chess_games_filepath_part_51 = base_directory / ".." / "chess_data" / "chess_games_part_51.pkl"
chess_games_filepath_part_52 = base_directory / ".." / "chess_data" / "chess_games_part_52.pkl"
chess_games_filepath_part_53 = base_directory / ".." / "chess_data" / "chess_games_part_53.pkl"
chess_games_filepath_part_54 = base_directory / ".." / "chess_data" / "chess_games_part_54.pkl"
chess_games_filepath_part_55 = base_directory / ".." / "chess_data" / "chess_games_part_55.pkl"
chess_games_filepath_part_56 = base_directory / ".." / "chess_data" / "chess_games_part_56.pkl"
chess_games_filepath_part_57 = base_directory / ".." / "chess_data" / "chess_games_part_57.pkl"
chess_games_filepath_part_58 = base_directory / ".." / "chess_data" / "chess_games_part_58.pkl"
chess_games_filepath_part_59 = base_directory / ".." / "chess_data" / "chess_games_part_59.pkl"
chess_games_filepath_part_60 = base_directory / ".." / "chess_data" / "chess_games_part_60.pkl"

chess_games_filepath_part_61 = base_directory / ".." / "chess_data" / "chess_games_part_61.pkl"
chess_games_filepath_part_62 = base_directory / ".." / "chess_data" / "chess_games_part_62.pkl"
chess_games_filepath_part_63 = base_directory / ".." / "chess_data" / "chess_games_part_63.pkl"
chess_games_filepath_part_64 = base_directory / ".." / "chess_data" / "chess_games_part_64.pkl"
chess_games_filepath_part_65 = base_directory / ".." / "chess_data" / "chess_games_part_65.pkl"
chess_games_filepath_part_66 = base_directory / ".." / "chess_data" / "chess_games_part_66.pkl"
chess_games_filepath_part_67 = base_directory / ".." / "chess_data" / "chess_games_part_67.pkl"
chess_games_filepath_part_68 = base_directory / ".." / "chess_data" / "chess_games_part_68.pkl"
chess_games_filepath_part_69 = base_directory / ".." / "chess_data" / "chess_games_part_69.pkl"
chess_games_filepath_part_70 = base_directory / ".." / "chess_data" / "chess_games_part_70.pkl"

chess_games_filepath_part_71 = base_directory / ".." / "chess_data" / "chess_games_part_71.pkl"
chess_games_filepath_part_72 = base_directory / ".." / "chess_data" / "chess_games_part_72.pkl"
chess_games_filepath_part_73 = base_directory / ".." / "chess_data" / "chess_games_part_73.pkl"
chess_games_filepath_part_74 = base_directory / ".." / "chess_data" / "chess_games_part_74.pkl"
chess_games_filepath_part_75 = base_directory / ".." / "chess_data" / "chess_games_part_75.pkl"
chess_games_filepath_part_76 = base_directory / ".." / "chess_data" / "chess_games_part_76.pkl"
chess_games_filepath_part_77 = base_directory / ".." / "chess_data" / "chess_games_part_77.pkl"
chess_games_filepath_part_78 = base_directory / ".." / "chess_data" / "chess_games_part_78.pkl"
chess_games_filepath_part_79 = base_directory / ".." / "chess_data" / "chess_games_part_79.pkl"
chess_games_filepath_part_80 = base_directory / ".." / "chess_data" / "chess_games_part_80.pkl"

chess_games_filepath_part_81 = base_directory / ".." / "chess_data" / "chess_games_part_81.pkl"
chess_games_filepath_part_82 = base_directory / ".." / "chess_data" / "chess_games_part_82.pkl"
chess_games_filepath_part_83 = base_directory / ".." / "chess_data" / "chess_games_part_83.pkl"
chess_games_filepath_part_84 = base_directory / ".." / "chess_data" / "chess_games_part_84.pkl"
chess_games_filepath_part_85 = base_directory / ".." / "chess_data" / "chess_games_part_85.pkl"
chess_games_filepath_part_86 = base_directory / ".." / "chess_data" / "chess_games_part_86.pkl"
chess_games_filepath_part_87 = base_directory / ".." / "chess_data" / "chess_games_part_87.pkl"
chess_games_filepath_part_88 = base_directory / ".." / "chess_data" / "chess_games_part_88.pkl"
chess_games_filepath_part_89 = base_directory / ".." / "chess_data" / "chess_games_part_89.pkl"
chess_games_filepath_part_90 = base_directory / ".." / "chess_data" / "chess_games_part_90.pkl"

chess_games_filepath_part_91 = base_directory / ".." / "chess_data" / "chess_games_part_91.pkl"
chess_games_filepath_part_92 = base_directory / ".." / "chess_data" / "chess_games_part_92.pkl"
chess_games_filepath_part_93 = base_directory / ".." / "chess_data" / "chess_games_part_93.pkl"
chess_games_filepath_part_94 = base_directory / ".." / "chess_data" / "chess_games_part_94.pkl"
chess_games_filepath_part_95 = base_directory / ".." / "chess_data" / "chess_games_part_95.pkl"
chess_games_filepath_part_96 = base_directory / ".." / "chess_data" / "chess_games_part_96.pkl"
chess_games_filepath_part_97 = base_directory / ".." / "chess_data" / "chess_games_part_97.pkl"
chess_games_filepath_part_98 = base_directory / ".." / "chess_data" / "chess_games_part_98.pkl"
chess_games_filepath_part_99 = base_directory / ".." / "chess_data" / "chess_games_part_99.pkl"
chess_games_filepath_part_100 = base_directory / ".." / "chess_data" / "chess_games_part_100.pkl"

############################################################################################################


chess_data = pd.read_pickle(chess_games_filepath_part_20, compression = 'zip')

# est_q_vals_file_path = 


# only set this for certain scripts
# chess_data = chess_data.head()