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
chess_engine_num_moves_to_return = 1
chess_engine_depth_limit = 1

default_learning_rate = 0.6
default_discount_factor = 0.35