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

        try: 
            train_one_game(game_number, est_q_val_table, chess_data, w_agent, b_agent, w_curr_q_value, b_curr_q_value)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at train_one_game: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            raise Exception from e

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
            rl_agent_plays_move_during_training(w_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at rl_agent_plays_move_during_training: {e}\n')
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

        # get latest curr_state since rl_agent_plays_move_during_training updated the chessboard
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
            rl_agent_plays_move_during_training(b_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at rl_agent_plays_move_during_training: {e}\n')
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

        # get latest curr_state since rl_agent_plays_move_during_training updated the chessboard
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
        try: 
            generate_q_est_df_one_game(chess_data, game_number, w_agent, b_agent)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at generate_q_est_df_one_game: {e}\n')
            training_functions_logger.error(f'at game: {game_number}\n')
            raise Exception from e

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
            rl_agent_plays_move_during_training(w_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at rl_agent_plays_move_during_training: {e}\n')
            training_functions_logger.error(f'at game_number: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # get latest curr_state since rl_agent_plays_move_during_training updated the chessboard
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
            rl_agent_plays_move_during_training(b_chess_move, game_number, environ)
        except Exception as e:
            training_functions_logger.error(f'An error occurred at rl_agent_plays_move_during_training: {e}\n')
            training_functions_logger.error(f'at: {game_number}\n')
            training_functions_logger.error(f'at state: {curr_state}\n')
            raise Exception from e

        # get latest curr_state since rl_agent_plays_move_during_training updated the chessboard
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


# def continue_training_rl_agents(num_games_to_play: int, w_agent, b_agent, environ) -> None:
#     """ continues to train the agent, this time the agents make their own decisions instead 
#         of playing through the database.

#         precondition: the agents have already been trained using the SARSA algorithm on the database.
#                       and the respective q tables have been populated. Each agent passed to this 
#                       function should have their `is_trained` flag set to True. And the q tables
#                       have been assigned to the agents.
#         Args:
#             num_games_to_play (int): The number of games to play.
#             w_agent (RLAgent): The white agent.
#             b_agent (RLAgent): The black agent.
#     """ 
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
    except Exception as e:
        training_functions_logger.error(f'at Bradley.find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error(f'failed to analyze_board_state\n')
        raise Exception from e
    
    # load up the chess board with opponent's anticipated chess move 
    try:
        environ.load_chessboard_for_q_est(anticipated_next_move)
    except Exception as e:
        training_functions_logger.error(f'at Bradley.find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error(f'failed to load_chessboard_for_q_est\n')
        raise Exception from e
    
    # check if the game would be over with the anticipated next move, like unstopable checkmate.
    if environ.board.is_game_over() or not environ.get_legal_moves():
        try:
            environ.pop_chessboard()
        except Exception as e:
            training_functions_logger.error(f'at Bradley.find_estimated_q_value. An error occurred: {e}\n')
            training_functions_logger.error(f'failed at environ.pop_chessboard\n')
            raise Exception from e

    # this is the q estimated value due to what the opposing agent is likely to play in response to our move.    
    try:
        est_qval_analysis = analyze_board_state(environ.board, engine)
    except Exception as e:
        training_functions_logger.error(f'at Bradley.find_estimated_q_value. An error occurred: {e}\n')
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
        training_functions_logger.error(f'@ Bradley.find_estimated_q_value. An error occurred: {e}\n')
        training_functions_logger.error("failed to pop_chessboard after est q val analysis values found\n")
        raise Exception from e

    return est_qval
# end of find_estimated_q_value

def find_next_q_value(curr_qval: int, learn_rate: float, reward: int, discount_factor: float, est_qval: int) -> int:
    """
        Calculates the next q-value using the SARSA (State-Action-Reward-State-Action) algorithm.

        This method calculates the next q-value based on the current q-value, the learning rate, the reward, the 
        discount factor, and the estimated q-value for the next state-action pair. The formula used is:

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

        Side Effects:
            None.
    """
    try:
        next_qval = int(curr_qval + learn_rate * (reward + ((discount_factor * est_qval) - curr_qval)))
        return next_qval
    except OverflowError:
        training_functions_logger.error(f'@ Bradley.find_next_q_value. An error occurred: OverflowError\n')
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
        include the mate score, the centipawn score, and the anticipated next move. The method first checks if the 
        board is in a valid state. If it's not, it writes an error message to the errors file and raises a ValueError.

        The method then tries to analyze the board using the Stockfish engine. If an error occurs during the analysis, 
        it writes an error message to the errors file and raises an Exception.

        The method then tries to extract the mate score and the centipawn score from the analysis results. If an error 
        occurs while extracting the scores, it writes an error message to the errors file and raises an Exception.

        Finally, the method tries to extract the anticipated next move from the analysis results. If an error occurs 
        while extracting the anticipated next move, it writes an error message to the errors file and raises an Exception.

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
        training_functions_logger.error(f'at Bradley.analyze_board_state. Board is in invalid state\n')
        raise custom_exceptions.InvalidBoardStateError(f'at Bradley.analyze_board_state. Board is in invalid state\n')

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

        # Check if the score is a mate score and get the mate score, otherwise get the centipawn score
        if pov_score.is_mate():
            mate_score = pov_score.mate()
        else:
            centipawn_score = pov_score.score()
    except Exception as e:
        training_functions_logger.error(f'An error occurred while extracting scores: {e}\n')
        raise custom_exceptions.ScoreExtractionError("Error occurred while extracting scores from analysis") from e

    try:
        # Extract the anticipated next move from the analysis
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

def rl_agent_plays_move_during_training(chess_move: str, game_number, environ: Environ.Environ) -> None:
    """
        Loads the chessboard with the given move and updates the current state of the environment.
        This method is used during training. It first attempts to load the chessboard with the given move. If an 
        error occurs while loading the chessboard, it writes an error message to the errors file and raises an 
        exception. It then attempts to update the current state of the environment. If an error occurs while 
        updating the current state, it writes an error message to the errors file and raises an exception.

        Args: 
            chess_move (str): A string representing the chess move in standard algebraic notation.
            game_number: The current game being played during training.
        Raises:
            Exception: An exception is raised if an error occurs while loading the chessboard or updating the 
            current state. The original exception is included in the raised exception.
        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
            Writes to the errors file if an error occurs.
    """
    try:
        environ.load_chessboard(chess_move, game_number)
    except custom_exceptions.ChessboardLoadError as e:
        training_functions_logger.error(f'at Bradley.rl_agent_plays_move_during_training. An error occurred at {game_number}: {e}\n')
        training_functions_logger.error(f"failed to load_chessboard with move {chess_move}\n")
        raise Exception from e

    try:
        environ.update_curr_state()
    except custom_exceptions.StateUpdateError as e:
        training_functions_logger.error(f'at Bradley.rl_agent_plays_move_during_training. update_curr_state() failed to increment turn_index, Caught exception: {e}\n')
        training_functions_logger.error(f'Current state is: {environ.get_curr_state()}\n')
        raise Exception from e
# end of rl_agent_plays_move_during_training

def get_reward(chess_move: str) -> int:
    """
        Calculates the reward for a given chess move based on the type of move.

        This method calculates the reward for a given chess move by checking for specific patterns in the move string 
        that correspond to different types of moves. The reward is calculated as follows:

        1. If the move involves the development of a piece (N, R, B, Q), the reward is increased by the value 
        associated with 'piece_development' in the game settings.
        2. If the move involves a capture (indicated by 'x' in the move string), the reward is increased by the value 
        associated with 'capture' in the game settings.
        3. If the move involves a promotion (indicated by '=' in the move string), the reward is increased by the value 
        associated with 'promotion' in the game settings. If the promotion is to a queen (indicated by '=Q' in the 
        move string), the reward is further increased by the value associated with 'promotion_queen' in the game 
        settings.

        Args:
            chess_move (str): A string representing the selected chess move in standard algebraic notation.

        Returns:
            int: The total reward for the given chess move, calculated based on the type of move.

        Raises:
            ValueError: If the chess_move string is empty or invalid.

        Side Effects:
            None.
    """
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
    except Exception as e:             # <<<<< !!!!!! Need a custom exception here later
        training_functions_logger.error(f'An error occurred at start_chess_engine: {e}\n')
        raise Exception from e
# end of start_chess_engine


def assign_points_to_q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
    """
        Assigns points to the q table for the given chess move, current turn, current q value, and RL agent color.
        This method assigns points to the q table for the RL agent of the given color. It calls the 
        `change_q_table_pts` method on the RL agent, passing in the chess move, the current turn, and the current q 
        value. If a KeyError is raised because the chess move is not represented in the q table, the method writes 
        an error message to the errors file, updates the q table to include the chess move, and tries to assign 
        points to the q table again.

        Args:
            chess_move (str): The chess move to assign points to in the q table.
            curr_turn (str): The current turn of the game.
            curr_qval (int): The current q value for the given chess move.
            rl_agent_color (str): The color of the RL agent making the move.

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