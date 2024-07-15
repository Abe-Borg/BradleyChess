import pandas as pd
import game_settings
import random
import chess.engine
import logging
import Agent

def play_game(bubs, chess_agent) -> None:
    player_turn = 'W'
    while bubs.is_game_over() == False:
        try:
            print(f'\nCurrent turn is :  {bubs.environ.get_curr_turn()}\n')
            chess_move = handle_move(player_turn, chess_agent)
            print(f'{player_turn} played {chess_move}\n')
        except Exception as e:
            # put in logger here.
            raise Exception from e

        player_turn = 'B' if player_turn == 'W' else 'W'

    print(f'Game is over, result is: {bubs.get_game_outcome()}')
    print(f'The game ended because of: {bubs.get_game_termination_reason()}')
    bubs.reset_environ()
### end of play_game

def handle_move(player_color, chess_agent) -> str:
    if player_color == chess_agent.color:
        print('=== RL AGENT\'S TURN ===\n')
        return bubs.rl_agent_selects_chess_move(player_color, chess_agent)
    else:
        print('=== OPPONENT\'S TURN ===')
        move = input('Enter chess move: ')
        while not bubs.receive_opp_move(move):
            print('Invalid move, try again.')
            move = input('Enter chess move: ')
        return move

def agent_vs_agent(bubs, w_agent, b_agent) -> None:
    try:
        while bubs.is_game_over() == False:
            # agent_vs_agent_file.write(f'\nCurrent turn: {bubs.environ.get_curr_turn()}')
            play_turn(w_agent)
            
            # sometimes game ends after white's turn
            if bubs.is_game_over() == False:
                play_turn(b_agent)

        # agent_vs_agent_file.write('Game is over, chessboard looks like this:\n')
        # agent_vs_agent_file.write(bubs.environ.board + '\n\n')
        # agent_vs_agent_file.write(f'Game result is: {bubs.get_game_outcome()}\n')
        # agent_vs_agent_file.write(f'Game ended because of: {bubs.get_game_termination_reason()}\n')
    except Exception as e:
        # agent_vs_agent_file.write(f'An unhandled error occurred: {e}\n')
        raise Exception from e

    bubs.reset_environ()
### end of agent_vs_agent

def play_turn(chess_agent):
    try:
        chess_move = bubs.rl_agent_selects_chess_move(chess_agent)
        # agent_vs_agent_file.write(f'{chess_agent.color} agent played {chess_move}\n')
    except Exception as e:
        # agent_vs_agent_file.write(f'An error occurred: {e}\n')
        raise Exception from e

def pikl_q_table(bubs, chess_agent, q_table_path: str) -> None:
    chess_agent.q_table.to_pickle(q_table_path, compression = 'zip')
### end of pikl_Q_table

def bootstrap_agent(chess_agent, existing_q_table_path: str) -> Agent.Agent:
    chess_agent.q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    chess_agent.is_trained = True
    return chess_agent
### end of bootstrap_agent

def get_number_with_probability(probability: float) -> int:
    """Generate a random number with a given probability.
    Args:
        probability (float): A float representing the probability of generating a 1.
    Returns:
        int: A random integer value of either 0 or 1.
    """
    if random.random() < probability:
        return 1
    else:
        return 0
### end of get_number_with_probability

    def reset_q_table(q_table) -> None:
        q_table.iloc[:, :] = 0
        return q_table    
    ### end of reset_Q_table ###

def start_chess_engine(): 
    chess_engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    return chess_engine

def is_game_over(environ) -> bool:
    """
        This method determines whether the game is over based on three conditions: if the game is over according to 
        the chessboard, if the current turn index has reached the maximum turn index defined in the game settings, 
        or if there are no legal moves left.

        Arg: environ object, which manages a chessboard

        Returns:
            bool: A boolean value indicating whether the game is over. Returns True if any of the three conditions 
            are met, and False otherwise.
        Side Effects:
            None.
    """
    return (
        environ.board.is_game_over() or
        environ.turn_index >= game_settings.max_turn_index or
        (len(environ.get_legal_moves()) == 0)
    )
### end of is_game_over

def get_game_outcome(environ) -> str:
    """
        Returns the outcome of the chess game.
        This method returns the outcome of the chess game. It calls the `outcome` method on the chessboard, which 
        returns an instance of the `chess.Outcome` class, and then calls the `result` method on this instance to 
        get the outcome of the game. If an error occurs while getting the game outcome, an error message is 
        returned.

        Returns:
            str: A string representing the outcome of the game. The outcome is a string in the format '1-0', '0-1', 
            or '1/2-1/2', representing a win for white, a win for black, or a draw, respectively. If an error 
            occurred while getting the game outcome, the returned string starts with 'error at get_game_outcome: ' 
            and includes the error message.
        Raises:
            GameOutcomeError: If the game outcome cannot be determined.
    """
    try:
        return environ.board.outcome().result()
    except custom_exceptions.GameOutcomeError as e:
        # self.error_logger.error('hello from Bradley.get_game_outcome\n')
        return f'error at get_game_outcome: {e}'
### end of get_game_outcome

def get_game_termination_reason(environ) -> str:
    """
        Returns a string that describes the reason for the game ending.
        This method returns a string that describes the reason for the game ending. It calls the `outcome` method 
        on the chessboard, which returns an instance of the `chess.Outcome` class, and then gets the termination 
        reason from this instance. If an error occurs while getting the termination reason, an error message is 
        returned.

        Returns:
            str: A string representing the reason for the game ending. 
            If an error occurred while getting the termination reason, the returned string starts with 'error at 
            get_game_termination_reason: ' and includes the error message.
        Raises:
            GameTerminationError: If the termination reason cannot be determined.
        Side Effects:
            None.
    """
    try:
        return str(environ.board.outcome().termination)
    except custom_exceptions.GameTerminationError as e:
        # self.error_logger.error('hello from Bradley.get_game_termination_reason\n')
        # self.error_logger.error(f'Error: {e}, failed to get game end reason\n')
        return f'error at get_game_termination_reason: {e}'
    ### end of get_game_termination_reason

def train_rl_agents(self, est_q_val_table, chess_data, w_agent, b_agent):
    """
        Trains the RL agents using the SARSA algorithm and sets their `is_trained` flag to True.
        This method trains two RL agents by having them play games from a database exactly as shown, and learning from that. 
        The agents learn from these games using the SARSA (State-Action-Reward-State-Action) algorithm.
        
        Args:
            est_q_val_table (pd.DataFrame): A DataFrame containing the estimated Q values for each game in the training set.
        Raises:
            Exception: A TrainingError is raised if an error occurs while getting the current state, choosing an action, playing a move, or getting the latest current state. The exception is written to the errors file.
        Side Effects:
            Modifies the Q tables of the RL agents and sets their `is_trained` flag to True.
            Writes the start and end of each game, any errors that occur, and the final state of the chessboard to the initial training results file.
            Writes any errors that occur to the errors file.
            Resets the environment at the end of each game.
    """
    ### FOR EACH GAME IN THE TRAINING SET ###
    for game_num_str in chess_data.index:
        num_chess_moves_curr_training_game: int = chess_data.at[game_num_str, 'PlyCount']

        w_curr_qval: int = game_settings.initial_q_val
        b_curr_qval: int = game_settings.initial_q_val

        train_one_game(game_num_str, est_q_val_table, chess_data, w_agent, b_agent, w_curr_qval, b_curr_qval, num_chess_moves_curr_training_game)

    # training is complete, all games in database have been processed
    # if game_settings.PRINT_STEP_BY_STEP:
        # self.step_by_step_logger.debug(f'training is complete\n')
    
    w_agent.is_trained = True
    b_agent.is_trained = True
    return w_agent, b_agent
### end of train_rl_agents

def train_one_game(game_num_str, est_q_val_table, chess_data, w_agent, b_agent, w_curr_qval, b_curr_qval, num_chess_moves_curr_training_game) -> None:
    # est_q_val_table should probably be a dictionary of lists, where the key is the game number and the value is a list of est q values for each turn.
    environ = Environ.Environ()

    # if game_settings.PRINT_STEP_BY_STEP:
        # self.step_by_step_logger.debug(f'At game: {game_num_str}\n')
        # self.step_by_step_logger.debug(f'num_chess_moves_curr_training_game: {num_chess_moves_curr_training_game}\n')
        # self.step_by_step_logger.debug(f'w_curr_qval: {w_curr_qval}\n')
        # self.step_by_step_logger.debug(f'b_curr_qval: {b_curr_qval}\n')
    
    # if game_settings.PRINT_TRAINING_RESULTS:
        # self.initial_training_logger.info(f'\nStart of {game_num_str} training\n\n')

    try:
        curr_state = environ.get_curr_state()
    except Exception as e:
        # self.error_logger.error(f'An error occurred environ.get_curr_state: {e}\n')
        # self.error_logger.error(f'curr board is:\n{environ.board}\n\n')
        # self.error_logger.error(f'at game: {game_num_str}\n')
        # self.error_logger.error(f'at turn: {curr_state['turn_index']}')
        return
    
    # if game_settings.PRINT_STEP_BY_STEP:
        # self.step_by_step_logger.debug(f'curr_state: {curr_state}\n')

    ### THIS WHILE LOOP PLAYS THROUGH ONE GAME ###
    while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
        ##################### WHITE'S TURN ####################
        # choose action a from state s, using policy
        w_chess_move = w_agent.choose_action(curr_state, game_num_str)

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'w_chess_move: {w_chess_move}\n')

        if not w_chess_move:
            # self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
            # self.error_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
            return # game is over, exit function.

        ### ASSIGN POINTS TO Q TABLE FOR WHITE AGENT ###
        # on the first turn for white, this would assign to W1 col at chess_move row.
        # on W's second turn, this would be Q_next which is calculated on the first loop.                
        helper_methods.assign_points_to_Q_table(w_chess_move, curr_state['curr_turn'], w_curr_qval, w_agent)

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'curr_turn_for_q_est: {curr_turn_for_q_est}\n')

        ### WHITE AGENT PLAYS THE SELECTED MOVE ###
        # take action a, observe r, s', and load chessboard
        try:
            helper_methods.rl_agent_plays_move(w_chess_move, game_num_str, environ)
        except Exception as e:
            # self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
            # self.error_logger.error(f'at curr_game: {game_num_str}\n')
            # self.error_logger.error(f'at state: {curr_state}\n')
            return # game is over, exit function.

        W_reward = helper_methods.get_reward(w_chess_move)

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'W_reward: {W_reward}\n')

        # get latest curr_state since rl_agent_plays_move updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            # self.error_logger.error(f'An error occurred at get_curr_state: {e}\n')
            # self.error_logger.error(f'curr board is:\n{environ.board}\n\n')
            # self.error_logger.error(f'At game: {game_num_str}\n')
            # self.error_logger.error(f'at state: {curr_state}\n')
            return # game is over, exit function.
        
        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'curr_state: {curr_state}\n')

        # find the estimated Q value for White, but first check if game ended
        if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
            
            # if game_settings.PRINT_STEP_BY_STEP:
                # self.step_by_step_logger.debug(f'game {game_num_str} is over\n')
            return # game is over, exit function.

        else: # current game continues
            # the var curr_turn_for_q_est is here because we previously moved to next turn (after move was played)
            # but we want to assign the q est based on turn just before the curr turn was incremented.
            W_est_Qval: int = est_q_val_table.at[game_num_str, curr_turn_for_q_est]

            # if game_settings.PRINT_STEP_BY_STEP:
                # self.step_by_step_logger.debug(f'W_est_Qval: {W_est_Qval}\n')

        ##################### BLACK'S TURN ####################
        # choose action a from state s, using policy
        b_chess_move = b_agent.choose_action(curr_state, game_num_str)

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'b_chess_move: {b_chess_move}\n')
        
        if not b_chess_move:
            # self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
            # self.error_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
            # self.error_logger.error(f'at: {game_num_str}\n')
            return # game is over, exit function

        # assign points to Q table
        helper_methods.assign_points_to_Q_table(b_chess_move, curr_state['curr_turn'], b_curr_qval, b_agent)

        curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'curr_turn_for_q_est: {curr_turn_for_q_est}\n')

        ##### BLACK AGENT PLAYS SELECTED MOVE #####
        # take action a, observe r, s', and load chessboard
        try:
            helper_methods.rl_agent_plays_move(b_chess_move, game_num_str)
        except Exception as e:
            # self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
            # self.error_logger.error(f'at curr_game: {game_num_str}\n')
            # self.error_logger.error(f'at state: {curr_state}\n')
            return # game is over, exit function

        B_reward = helper_methods.get_reward(b_chess_move)

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'B_reward: {B_reward}\n')

        # get latest curr_state since self.rl_agent_plays_move updated the chessboard
        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            # self.error_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
            # self.error_logger.error(f'curr board is:\n{environ.board}\n\n')
            # self.error_logger.error(f'At game: {game_num_str}\n')
            return # game is over, exit function

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'curr_state: {curr_state}\n')

        # find the estimated Q value for Black, but first check if game ended
        if environ.board.is_game_over() or not curr_state['legal_moves']:
            
            # if game_settings.PRINT_STEP_BY_STEP:
                # self.step_by_step_logger.debug(f'game {game_num_str} is over\n')
            return # game is over, exit function
        else: # current game continues
            B_est_Qval: int = est_q_val_table.at[game_num_str, curr_turn_for_q_est]

        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'B_est_Qval: {B_est_Qval}\n')
            # self.step_by_step_logger.debug(f'about to calc next q values\n')
            # self.step_by_step_logger.debug(f'w_curr_qval: {w_curr_qval}\n')
            # self.step_by_step_logger.debug(f'b_curr_qval: {b_curr_qval}\n')
            # self.step_by_step_logger.debug(f'W_reward: {W_reward}\n')
            # self.step_by_step_logger.debug(f'B_reward: {B_reward}\n')
            # self.step_by_step_logger.debug(f'W_est_Qval: {W_est_Qval}\n')
            # self.step_by_step_logger.debug(f'B_est_Qval: {B_est_Qval}\n\n')

        # ***CRITICAL STEP***, this is the main part of the SARSA algorithm.
        W_next_Qval: int = game_settings.find_next_Qval(w_curr_qval, w_agent.learn_rate, W_reward, w_agent.discount_factor, W_est_Qval)
        B_next_Qval: int = game_settings.find_next_Qval(b_curr_qval, b_agent.learn_rate, B_reward, b_agent.discount_factor, B_est_Qval)
    
        # if game_settings.PRINT_STEP_BY_STEP:
            # self.step_by_step_logger.debug(f'sarsa calc complete\n')
            # self.step_by_step_logger.debug(f'W_next_Qval: {W_next_Qval}\n')
            # self.step_by_step_logger.debug(f'B_next_Qval: {B_next_Qval}\n')

        # on the next turn, W_next_Qval and B_next_Qval will be added to the Q table. so if this is the end of the first round,
        # next round it will be W2 and then we assign the q value at W2 col
        w_curr_qval = W_next_Qval
        b_curr_qval = B_next_Qval

        try:
            curr_state = environ.get_curr_state()
            
            # if game_settings.PRINT_STEP_BY_STEP:
                # self.step_by_step_logger.debug(f'curr_state: {curr_state}\n')
        except Exception as e:
            # self.error_logger.error(f'An error occurred: {e}\n')
            # self.error_logger.error("failed to get_curr_state\n") 
            # self.error_logger.error(f'At game: {game_num_str}\n')
            break
    ### END OF CURRENT GAME LOOP ###

    # if game_settings.PRINT_TRAINING_RESULTS:
        # self.initial_training_logger.info(f'{game_num_str} is over.\n')
        # self.initial_training_logger.info(f'\nThe Chessboard looks like this:\n')
        # self.initial_training_logger.info(f'\n{environ.board}\n\n')
        # self.initial_training_logger.info(f'Game result is: {helper_methods.get_game_outcome(environ)}\n')    
        # self.initial_training_logger.info(f'The game ended because of: {helper_methods.get_game_termination_reason()}\n')
        # self.initial_training_logger.info(f'DB shows game ended b/c: {chess_data.at[game_num_str, "Result"]}\n')

    # if game_settings.PRINT_STEP_BY_STEP:
        # self.step_by_step_logger.debug(f'game {game_num_str} is over\n')
    
    environ.reset_environ() # reset and go to next game in training set
### end of train_one_game

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
        raise ValueError("Invalid chess move input")

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

def generate_Q_est_df(chess_data) -> None:
    """
        Generates a dataframe containing the estimated Q-values for each chess move in the chess database.

        This method iterates over each game in the chess database and plays through the game using the reinforcement 
        learning agents. For each move, it calculates the estimated Q-value and writes it to a file.

        The method first tries to get the current state of the game. If an error occurs, it logs the error and the 
        current board state in the errors file and moves on to the next game.

        The method then enters a loop where it alternates between the white and black agents choosing and playing 
        moves. If an error occurs while choosing or playing a move, the method logs the error and the current state 
        in the errors file and breaks out of the loop to move on to the next game.

        After each move, the method tries to get the latest state of the game. If an error occurs, it logs the error 
        and the current board state in the errors file and breaks out of the loop to move on to the next game.

        If the game is not over and there are still legal moves, the method tries to find the estimated Q-value for 
        the current move and writes it to the file. If an error occurs while finding the estimated Q-value, the 
        method logs the error and the current state in the errors file and breaks out of the loop to move on to the 
        next game.

        The loop continues until the game is over, there are no more legal moves, or the maximum number of moves for 
        the current training game has been reached.

        After each game, the method resets the environment to prepare for the next game.

        Args:
            chess_data (pd.DataFrame): A DataFrame containing the chess database.
        Returns:
            estimated_q_values (pd.DataFrame): A DataFrame containing the estimated Q-values for each chess move.
    """
    environ = Environ.Environ()
    estimated_q_values = chess_data.copy(deep = True)
    estimated_q_values = estimated_q_values.astype('int64')
    estimated_q_values.iloc[:, 1:] = 0

    ### FOR EACH GAME IN THE TRAINING SET ###
    for game_num_str in chess_data.index:
        num_chess_moves_curr_training_game: int = chess_data.at[game_num_str, 'PlyCount']

        try:
            curr_state = environ.get_curr_state()
        except Exception as e:
            # self.error_logger.error(f'An error occurred at self.environ.get_curr_state: {e}\n')
            # self.error_logger.error(f'at: {game_num_str}\n')
            return
        
        ### LOOP PLAYS THROUGH ONE GAME ###
        while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
            ##################### WHITE'S TURN ####################
            # choose action a from state s, using policy
            w_chess_move = w_agent.choose_action(curr_state, game_num_str)
            if not w_chess_move:
                # self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                # self.error_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
                return

            # assign curr turn to new var for now. once agent plays move, turn will be updated, but we need 
            # to track the turn before so that the est q value can be assigned to the correct column.
            curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

            ### WHITE AGENT PLAYS THE SELECTED MOVE ###
            # take action a, observe r, s', and load chessboard
            try:
                game_settings.rl_agent_plays_move(w_chess_move, game_num_str)
            except Exception as e:
                # self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
                # self.error_logger.error(f'at: {game_num_str}\n')
                return

            # get latest curr_state since self.rl_agent_plays_move updated the chessboard
            try:
                curr_state = environ.get_curr_state()
            except Exception as e:
                # self.error_logger.error(f'An error occurred at get_curr_state: {e}\n')
                # self.error_logger.error(f'at: {game_num_str}\n')
                return
            
            # find the estimated Q value for White, but first check if game ended
            if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                return
            else: # current game continues
                try:
                    W_est_Qval: int = self.find_estimated_Q_value()
                    q_est_vals_file.write(f'{curr_turn_for_q_est}, {W_est_Qval}\n')
                except Exception as e:
                    # self.error_logger.error(f'An error occurred while retrieving W_est_Qval: {e}\n')
                    # self.error_logger.error(f"at White turn, failed to find_estimated_Q_value\n")
                    # self.error_logger.error(f'curr state is:{curr_state}\n')
                    return

            ##################### BLACK'S TURN ####################
            # choose action a from state s, using policy
            b_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
            if not b_chess_move:
                # self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                # self.error_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
                # self.error_logger.error(f'at: {game_num_str}\n')
                return

            # assign curr turn to new var for now. once agent plays move, turn will be updated, but we need 
            # to track the turn before so that the est q value can be assigned to the correct column.
            curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])
            
            ##### BLACK AGENT PLAYS SELECTED MOVE #####
            # take action a, observe r, s', and load chessboard
            try:
                helper_methods.rl_agent_plays_move(b_chess_move, game_num_str)
            except Exception as e:
                # self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
                # self.error_logger.error(f'at: {game_num_str}\n')
                return 

            # get latest curr_state since self.rl_agent_plays_move updated the chessboard
            try:
                curr_state = environ.get_curr_state()
            except Exception as e:
                # self.error_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
                # self.error_logger.error(f'at: {game_num_str}\n')
                return

            # find the estimated Q value for Black, but first check if game ended
            if environ.board.is_game_over() or not curr_state['legal_moves']:
                return
            else: # current game continues
                try:
                    b_est_q_val: int = helper_methods.find_estimated_Q_value()
                except Exception as e:
                    # self.error_logger.error(f"at Black turn, failed to find_estimated_Qvalue because error: {e}\n")
                    # self.error_logger.error(f'curr state is :{curr_state}\n')
                    # self.error_logger.error(f'at : {game_num_str}\n')
                    return

            try:
                curr_state = environ.get_curr_state()
            except Exception as e:
                # self.error_logger.error(f'An error occurred: {e}\n')
                # self.error_logger.error("failed to get_curr_state\n") 
                # self.error_logger.error(f'at: {game_num_str}\n')
                return
        ### END OF CURRENT GAME LOOP ###

    environ.reset_environ()
# end of generate_Q_est_df

def continue_training_rl_agents(self, num_games_to_play: int) -> None:
    """ continues to train the agent, this time the agents make their own decisions instead 
        of playing through the database.
    """ 
    ### placeholder, will implement this function later.
### end of continue_training_rl_agents