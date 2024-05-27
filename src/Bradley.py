import Environ
import Agent
import game_settings
import chess
import pandas as pd
import re
import copy
import time

class Bradley:
    """Acts as the single point of communication between the RL agent and the player.
    This class trains the agent and helps to manage the chessboard during play between the computer and the user.

    Args:
        chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data.
    Attributes:
        chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data.
        environ (Environ.Environ): An Environ object representing the chessboard environment.
        W_rl_agent (Agent.Agent): A white RL Agent object.
        B_rl_agent (Agent.Agent): A black RL Agent object.
        engine (chess.engine.SimpleEngine): A Stockfish engine used to analyze positions during training.
        errors_file (file): A file object to log errors.
        initial_training_results (file): A file object to log initial training results.
        additional_training_results (file): A file object to log additional training results.
        corrupted_games_list (list): A list of games that are corrupted and cannot be used for training.
    """
    def __init__(self, chess_data: pd.DataFrame):
        """
        Initializes a Bradley object with chess data, environment, agents, and a chess engine.
        This method initializes a Bradley object by setting the chess data, creating an environment, creating two 
        agents (one for white and one for black), and starting a chess engine. It also opens the errors file and 
        the training results files in append mode, and initializes an empty list for corrupted games.

        Side Effects:
            Opens the errors file, the initial training results file, and the additional training results file in 
            append mode.
            Modifies the chess_data, environ, W_rl_agent, B_rl_agent, corrupted_games_list, and engine attributes.
        """

        self.errors_file = open(game_settings.bradley_errors_filepath, 'a')
        self.initial_training_results = open(game_settings.initial_training_results_filepath, 'a')
        self.additional_training_results = open(game_settings.additional_training_results_filepath, 'a')
        self.chess_data = chess_data
        self.environ = Environ.Environ()
        self.W_rl_agent = Agent.Agent('W', self.chess_data)
        self.B_rl_agent = Agent.Agent('B', self.chess_data)
        self.corrupted_games_list = [] # list of games that are corrupted and cannot be used for training

        # stockfish is used to analyze positions during training this is how we estimate the q value 
        # at each position, and also for anticipated next position
        self.engine = chess.engine.SimpleEngine.popen_uci(game_settings.stockfish_filepath)
    ### end of Bradley constructor ###

    def __del__(self):
        self.errors_file.close()
        self.initial_training_results.close()
        self.additional_training_results.close()
    ### end of Bradley destructor ###

    def receive_opp_move(self, chess_move: str) -> bool:                                                                                 
        """
        Receives the opponent's chess move and updates the environment.

        This method receives the opponent's chess move, loads it onto the chessboard, and updates the current state 
        of the environment. If an error occurs while loading the chessboard or updating the current state, an error 
        message is written to the errors file and an exception is raised.

        Args:
            chess_move (str): A string representing the opponent's chess move, such as 'Nf3'.

        Returns:
            bool: A boolean value indicating whether the move was successfully loaded and the current state was 
            successfully updated. Returns False if an error occurred while loading the chessboard, and does not 
            attempt to update the current state.

        Raises:
            Exception: An exception is raised if the chessboard fails to load the move or if the current state fails 
            to update. The original exception is included in the raised exception.

        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
            Writes to the errors file if an error occurs.
        """
    
        try:
            self.environ.load_chessboard(chess_move)
        except Exception as e:
            self.errors_file.write("hello from Bradley.receive_opp_move, an error occurred\n")
            self.errors_file.write(f'Error: {e}, failed to load chessboard with move: {chess_move}\n')
            return False

        try:
            self.environ.update_curr_state()
            return True
        except Exception as e:
            self.errors_file.write(f'hello from Bradley.receive_opp_move, an error occurrd\n')
            self.errors_file.write(f'Error: {e}, failed to update_curr_state\n') 
            raise Exception from e
    ### end of receive_opp_move ###

    def rl_agent_selects_chess_move(self, rl_agent_color: str) -> str:
        """
        The Agent selects a chess move and loads it onto the chessboard.
        This method allows the Reinforcement Learning (RL) agent to select a chess move and load it onto the 
        chessboard. It is used during actual gameplay between the computer and the user, not during training. The 
        method first gets the current state of the environment. If the list of legal moves is empty, an exception 
        is raised. Depending on the color of the RL agent, the appropriate agent selects a move. The selected move 
        is then loaded onto the chessboard and the current state of the environment is updated.

        Args:
            rl_agent_color (str): A string indicating the color of the RL agent, either 'W' or 'B'.

        Returns:
            str: A string representing the selected chess move.

        Raises:
            Exception: An exception is raised if the current state is not valid, if the list of legal moves is 
            empty, if the chessboard fails to load the move, or if the current state fails to update. The original 
            exception is included in the raised exception.

        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
            Writes to the errors file if an error occurs.
        """

        try:
            curr_state = self.environ.get_curr_state()
        except Exception as e:
            self.errors_file.write("hello from Bradley.rl_agent_selects_chess_move, an error occurred\n")
            self.errors_file.write(f'Error: {e}, failed to get_curr_state\n')
            raise Exception from e
        
        if curr_state['legal_moves'] == []:
            self.errors_file.write('hello from Bradley.rl_agent_selects_chess_move, legal_moves is empty\n')
            self.errors_file.write(f'curr state is: {curr_state}\n')
            raise Exception(f'hello from Bradley.rl_agent_selects_chess_move, legal_moves is empty\n')
        
        if rl_agent_color == 'W':    
            # W agent selects action
            chess_move: str= self.W_rl_agent.choose_action(curr_state)
        else:
            # B agent selects action
            chess_move = self.B_rl_agent.choose_action(curr_state)        

        try:
            self.environ.load_chessboard(chess_move) 
        except Exception as e:
            self.errors_file.write(f'Error {e}: failed to load chessboard with move: {chess_move}\n')
            raise Exception from e

        try:
            self.environ.update_curr_state()
            return chess_move
        except Exception as e:
            self.errors_file.write(f'Error: {e}, failed to update_curr_state\n')
            raise Exception from e
    ### end of rl_agent_selects_chess_move

    def is_game_over(self) -> bool:
        """
        Determines whether the game is over.
        This method determines whether the game is over based on three conditions: if the game is over according to 
        the chessboard, if the current turn index has reached the maximum turn index defined in the game settings, 
        or if there are no legal moves left. This method is used during phase 2 of training and also during human 
        vs agent play.

        Returns:
            bool: A boolean value indicating whether the game is over. Returns True if any of the three conditions 
            are met, and False otherwise.

        Side Effects:
            None.
        """

        if self.environ.board.is_game_over() or (self.environ.turn_index >= game_settings.max_turn_index) or not self.environ.get_legal_moves():
            return True
        else:
            return False
    ### end of is_game_over
        
    def get_game_outcome(self) -> str:
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
            AttributeError: An AttributeError is raised if the game outcome cannot be determined. The original 
            exception is included in the raised exception.

        Side Effects:
            None.
        """

        try:
            game_outcome = self.environ.board.outcome().result()
            return game_outcome
        except AttributeError as e:
            return f'error at get_game_outcome: {e}'
    ### end of get_game_outcome
    
    def get_game_termination_reason(self) -> str:
        """
        Returns a string that describes the reason for the game ending.
        This method returns a string that describes the reason for the game ending. It calls the `outcome` method 
        on the chessboard, which returns an instance of the `chess.Outcome` class, and then gets the termination 
        reason from this instance. If an error occurs while getting the termination reason, an error message is 
        returned.

        Returns:
            str: A string representing the reason for the game ending. The termination reason is one of the 
            following: 'normal', 'abandoned', 'death', 'emergency', 'rules infraction', 'time forfeit', or 'unknown'. 
            If an error occurred while getting the termination reason, the returned string starts with 'error at 
            get_game_termination_reason: ' and includes the error message.

        Raises:
            AttributeError: An AttributeError is raised if the termination reason cannot be determined. The original 
            exception is included in the raised exception.

        Side Effects:
            None.
        """

        try:
            termination_reason = str(self.environ.board.outcome().termination)
            return termination_reason
        except AttributeError as e:
            return f'error at get_game_termination_reason: {e}'
    ### end of get_game_termination_reason
    
    def train_rl_agents(self, est_q_val_table: pd.DataFrame) -> None:
        """
        Trains the RL agents using the SARSA algorithm and sets their `is_trained` flag to True.

        This method trains two RL agents by having them play games from a database exactly as shown, and learning from that. 
        The training process involves the following steps:

        1. For each game in the training set, the method initializes the Q values for the white and black agents.
        2. It then enters a loop that continues until the current turn index reaches the number of moves in the current training game.
        3. On each turn, the agent chooses an action based on the current state and the policy. If an error occurs while choosing an action, the method writes an error message to the errors file and moves on to the next game.
        4. The method then assigns points to the Q table for the agent based on the chosen action, the current turn, and the current Q value.
        5. The agent then plays the chosen move. If an error occurs while playing the move, the method writes an error message to the errors file and moves on to the next game.
        6. The method then gets the reward for the move and updates the current state.
        7. If the game is not over, the method finds the estimated Q value for the agent and calculates the next Q value using the SARSA algorithm.
        8. The method then updates the current Q value to the next Q value and gets the latest current state.
        9. After all turns in the current game have been played, the method resets the environment to prepare for the next game.
        10. Once all games in the training set have been played, the method sets the `is_trained` flag of the agents to True.

        Args:
            est_q_val_table (pd.DataFrame): A DataFrame containing the estimated Q values for each game in the training set.

        Raises:
            Exception: An exception is raised if an error occurs while getting the current state, choosing an action, playing a move, or getting the latest current state. The exception is written to the errors file.

        Side Effects:
            Modifies the Q tables of the RL agents and sets their `is_trained` flag to True.
            Writes the start and end of each game, any errors that occur, and the final state of the chessboard to the initial training results file.
            Writes any errors that occur to the errors file.
            Resets the environment at the end of each game.
        """

        ### FOR EACH GAME IN THE TRAINING SET ###
        for game_num_str in self.chess_data.index:
            num_chess_moves_curr_training_game: int = self.chess_data.at[game_num_str, 'PlyCount']

            W_curr_Qval: int = game_settings.initial_q_val
            B_curr_Qval: int = game_settings.initial_q_val
            
            if game_settings.PRINT_TRAINING_RESULTS:
                self.initial_training_results.write(f'\nStart of {game_num_str} training\n\n')

            try:
                curr_state = self.environ.get_curr_state()
            except Exception as e:
                self.errors_file.write(f'An error occurred at self.environ.get_curr_state: {e}\n')
                self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                self.errors_file.write(f'at game: {game_num_str}\n')
                break

            ### LOOP PLAYS THROUGH ONE GAME ###
            while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                ##################### WHITE'S TURN ####################
                # choose action a from state s, using policy
                W_chess_move = self.W_rl_agent.choose_action(curr_state, game_num_str)
                if not W_chess_move:
                    self.errors_file.write(f'An error occurred at self.W_rl_agent.choose_action\n')
                    self.errors_file.write(f'W_chess_move is empty at state: {curr_state}\n')
                    break # and go to the next game. this game is over.

                ### ASSIGN POINTS TO Q TABLE FOR WHITE AGENT ###
                # on the first turn for white, this would assign to W1 col at chess_move row.
                # on W's second turn, this would be Q_next which is calculated on the first loop.                
                self.assign_points_to_Q_table(W_chess_move, curr_state['curr_turn'], W_curr_Qval, self.W_rl_agent.color)

                curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

                ### WHITE AGENT PLAYS THE SELECTED MOVE ###
                # take action a, observe r, s', and load chessboard
                try:
                    self.rl_agent_plays_move(W_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    break # and go to the next game. this game is over.

                W_reward = self.get_reward(W_chess_move)

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at get_curr_state: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    self.errors_file.write(f'At game: {game_num_str}\n')
                    break
                
                # find the estimated Q value for White, but first check if game ended
                if self.environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                    break # and go to next game

                else: # current game continues
                    W_est_Qval: int = est_q_val_table.at[game_num_str, curr_turn_for_q_est]

                ##################### BLACK'S TURN ####################
                # choose action a from state s, using policy
                B_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
                if not B_chess_move:
                    self.errors_file.write(f'An error occurred at self.W_rl_agent.choose_action\n')
                    self.errors_file.write(f'B_chess_move is empty at state: {curr_state}\n')
                    self.errors_file.write(f'at: {game_num_str}\n')
                    break # game is over, go to next game.

                # assign points to Q table
                self.assign_points_to_Q_table(B_chess_move, curr_state['curr_turn'], B_curr_Qval, self.B_rl_agent.color)

                curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                # take action a, observe r, s', and load chessboard
                try:
                    self.rl_agent_plays_move(B_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    break 

                B_reward = self.get_reward(B_chess_move)

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at environ.get_curr_state: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    self.errors_file.write(f'At game: {game_num_str}\n')
                    break

                # find the estimated Q value for Black, but first check if game ended
                if self.environ.board.is_game_over() or not curr_state['legal_moves']:
                    break # and go to next game
                else: # current game continues
                    B_est_Qval: int = est_q_val_table.at[game_num_str, curr_turn_for_q_est]

                # ***CRITICAL STEP***, this is the main part of the SARSA algorithm.
                W_next_Qval: int = self.find_next_Qval(W_curr_Qval, self.W_rl_agent.learn_rate, W_reward, self.W_rl_agent.discount_factor, W_est_Qval)
                B_next_Qval: int = self.find_next_Qval(B_curr_Qval, self.B_rl_agent.learn_rate, B_reward, self.B_rl_agent.discount_factor, B_est_Qval)
            
                # on the next turn, this Q value will be added to the Q table. so if this is the end of the first round,
                # next round it will be W2 and then we assign the q value at W2 col
                W_curr_Qval = W_next_Qval
                B_curr_Qval = B_next_Qval

                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to get_curr_state\n") 
                    self.errors_file.write(f'At game: {game_num_str}\n')
                    break
            ### END OF CURRENT GAME LOOP ###

            # this curr game is done, reset environ to prepare for the next game
            if game_settings.PRINT_TRAINING_RESULTS:
                self.initial_training_results.write(f'{game_num_str} is over.\n')
                self.initial_training_results.write(f'\nThe Chessboard looks like this:\n')
                self.initial_training_results.write(f'\n{self.environ.board}\n\n')
                self.initial_training_results.write(f'Game result is: {self.get_game_outcome()}\n')    
                self.initial_training_results.write(f'The game ended because of: {self.get_game_termination_reason()}\n')
                self.initial_training_results.write(f'DB shows game ended b/c: {self.chess_data.at[game_num_str, "Result"]}\n')

            self.environ.reset_environ() # reset and go to next game in training set
        
        # training is complete, all games in database have been processed
        self.W_rl_agent.is_trained = True
        self.B_rl_agent.is_trained = True
    ### end of train_rl_agents

    def continue_training_rl_agents(self, num_games_to_play: int) -> None:
        """ continues to train the agent, this time the agents make their own decisions instead 
            of playing through the database.
        """ 
        ### placeholder, will implement this function later.
    ### end of continue_training_rl_agents
    
    def assign_points_to_Q_table(self, chess_move: str, curr_turn: str, curr_Qval: int, rl_agent_color: str) -> None:
        """
        Assigns points to the Q table for the given chess move, current turn, current Q value, and RL agent color.
        This method assigns points to the Q table for the RL agent of the given color. It calls the 
        `change_Q_table_pts` method on the RL agent, passing in the chess move, the current turn, and the current Q 
        value. If a KeyError is raised because the chess move is not represented in the Q table, the method writes 
        an error message to the errors file, updates the Q table to include the chess move, and tries to assign 
        points to the Q table again.

        Args:
            chess_move (str): The chess move to assign points to in the Q table.
            curr_turn (str): The current turn of the game.
            curr_Qval (int): The current Q value for the given chess move.
            rl_agent_color (str): The color of the RL agent making the move.

        Raises:
            KeyError: A KeyError is raised if the chess move is not represented in the Q table. The exception is 
            written to the errors file.

        Side Effects:
            Modifies the Q table of the RL agent by assigning points to the given chess move.
            Writes to the errors file if a KeyError is raised.
        """

        if rl_agent_color == 'W':
            try:
                self.W_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
            except KeyError as e: 
                # chess move is not represented in the Q table, update Q table and try again.
                self.errors_file.write(f'caught exception: {e} at assign_points_to_Q_table\n')
                self.errors_file.write(f'Chess move is not represented in the White Q table, updating Q table and trying again...\n')

                self.W_rl_agent.update_Q_table([chess_move])
                self.W_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
        else: # black's turn
            try:
                self.B_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
            except KeyError as e: 
                # chess move is not represented in the Q table, update Q table and try again. 
                self.errors_file.write(f'caught exception: {e} at assign_points_to_Q_table\n')
                self.errors_file.write(f'Chess move is not represented in the White Q table, updating Q table and trying again...\n')

                self.B_rl_agent.update_Q_table([chess_move])
                self.B_rl_agent.change_Q_table_pts(chess_move, curr_turn, curr_Qval)
    # enf of assign_points_to_Q_table

    def rl_agent_plays_move(self, chess_move: str, curr_game) -> None:
        """
        Loads the chessboard with the given move and updates the current state of the environment.
        This method is used during training. It first attempts to load the chessboard with the given move. If an 
        error occurs while loading the chessboard, it writes an error message to the errors file and raises an 
        exception. It then attempts to update the current state of the environment. If an error occurs while 
        updating the current state, it writes an error message to the errors file and raises an exception.

        Args:
            chess_move (str): A string representing the chess move in standard algebraic notation.
            curr_game: The current game being played during training.

        Raises:
            Exception: An exception is raised if an error occurs while loading the chessboard or updating the 
            current state. The original exception is included in the raised exception.

        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
            Writes to the errors file if an error occurs.
        """

        try:
            self.environ.load_chessboard(chess_move, curr_game)
        except Exception as e:
            self.errors_file.write(f'at Bradley.rl_agent_plays_move. An error occurred at {curr_game}: {e}\n')
            self.errors_file.write(f"failed to load_chessboard with move {chess_move}\n")
            # self.corrupted_games_list.append(curr_game)
            raise Exception from e

        try:
            self.environ.update_curr_state()
        except Exception as e:
            self.errors_file.write(f'at Bradley.rl_agent_plays_move. update_curr_state() failed to increment turn_index, Caught exception: {e}\n')
            self.errors_file.write(f'Current state is: {self.environ.get_curr_state()}\n')
            raise Exception from e
    # end of rl_agent_plays_move

    def find_estimated_Q_value(self) -> int:
        """
        Estimates the Q-value for the RL agent's next action without actually playing the move.
        This method simulates the agent's next action and the anticipated response from the opposing agent 
        to estimate the Q-value. The steps are as follows:

        1. Observes the next state of the chessboard after the agent's move.
        2. Analyzes the current state of the board to predict the opposing agent's response.
        3. Loads the board with the anticipated move of the opposing agent.
        4. Estimates the Q-value based on the anticipated state of the board.

        The estimation of the Q-value is derived from analyzing the board state with the help of a chess engine 
        (like Stockfish). If there's no impending checkmate, the estimated Q-value is the centipawn score of 
        the board state. Otherwise, it's computed based on the impending checkmate turns multiplied by a predefined 
        mate score reward.

        After estimating the Q-value, the method reverts the board state to its original state before the simulation.

        Returns:
            int: The estimated Q-value for the agent's next action.

        Raises:
            Exception: An exception is raised if an error occurs while analyzing the board state, loading the 
            chessboard, popping the chessboard, or analyzing the board state for the estimated Q-value. The original 
            exception is included in the raised exception.

        Side Effects:
            Modifies the state of the chessboard by loading and popping moves.
            Writes to the errors file if an error occurs.
        """

        # RL agent just played a move. the board has changed, if stockfish analyzes the board, 
        # it will give points for the agent, based on the agent's latest move.
        # We also need the points for the ANTICIPATED next state, 
        # given the ACTICIPATED next action. In this case, the anticipated response from opposing agent.
        try:
            analysis_results = self.analyze_board_state(self.environ.board)
        except Exception as e:
            self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write(f'failed to analyze_board_state\n')
            raise Exception from e
        
        # load up the chess board with opponent's anticipated chess move 
        try:
            self.environ.load_chessboard_for_Q_est(analysis_results)
        except Exception as e:
            self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write(f'failed to load_chessboard_for_Q_est\n')
            raise Exception from e
        
        # check if the game would be over with the anticipated next move
        if self.environ.board.is_game_over() or not self.environ.get_legal_moves():
            try:
                self.environ.pop_chessboard()
            except Exception as e:
                self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
                self.errors_file.write(f'failed at self.environ.pop_chessboard\n')
                raise Exception from e
            return 1 # just return some value, doesn't matter.
            
        # this is the Q estimated value due to what the opposing agent is likely to play in response to our move.    
        try:
            est_Qval_analysis = self.analyze_board_state(self.environ.board)
        except Exception as e:
            self.errors_file.write(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write(f'failed at self.analyze_board_state\n')
            raise Exception from e

        # get pts for est_Qval 
        if est_Qval_analysis['mate_score'] is None:
            est_Qval = est_Qval_analysis['centipawn_score']
        else: # there is an impending checkmate
            est_Qval = game_settings.CHESS_MOVE_VALUES['mate_score']

        # IMPORTANT STEP, pop the chessboard of last move, we are estimating board states, not
        # playing a move.
        try:
            self.environ.pop_chessboard()
        except Exception as e:
            self.errors_file.write(f'@ Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.errors_file.write("failed to pop_chessboard\n")
            raise Exception from e

        return est_Qval
    # end of find_estimated_Q_value

    def find_next_Qval(self, curr_Qval: int, learn_rate: float, reward: int, discount_factor: float, est_Qval: int) -> int:
        """
        Calculates the next Q-value using the SARSA (State-Action-Reward-State-Action) algorithm.

        This method calculates the next Q-value based on the current Q-value, the learning rate, the reward, the 
        discount factor, and the estimated Q-value for the next state-action pair. The formula used is:

            next_Qval = curr_Qval + learn_rate * (reward + (discount_factor * est_Qval) - curr_Qval)

        This formula is derived from the SARSA algorithm, which is a model-free reinforcement learning method used 
        to estimate the Q-values for state-action pairs in an environment.

        Args:
            curr_Qval (int): The current Q-value for the state-action pair.
            learn_rate (float): The learning rate, a value between 0 and 1. This parameter controls how much the 
            Q-value is updated on each iteration.
            reward (int): The reward obtained from the current action.
            discount_factor (float): The discount factor, a value between 0 and 1. This parameter determines the 
            importance of future rewards.
            est_Qval (int): The estimated Q-value for the next state-action pair.

        Returns:
            int: The next Q-value, calculated using the SARSA algorithm.

        Raises:
            None.

        Side Effects:
            None.
        """

        next_Qval = int(curr_Qval + learn_rate * (reward + ((discount_factor * est_Qval) - curr_Qval)))
        return next_Qval
    # end of find_next_Qval
    
    def analyze_board_state(self, board: chess.Board) -> dict:
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

        Returns:
            dict: A dictionary containing the analysis results. The dictionary includes the mate score, the centipawn 
            score, and the anticipated next move.

        Raises:
            ValueError: A ValueError is raised if the board is in an invalid state.
            Exception: An Exception is raised if an error occurs during the analysis, while extracting the scores, or 
            while extracting the anticipated next move. The original exception is included in the raised exception.

        Side Effects:
            Writes to the errors file if an error occurs.
        """

        if not self.environ.board.is_valid():
            self.errors_file.write(f'at Bradley.analyze_board_state. Board is in invalid state\n')
            raise ValueError(f'at Bradley.analyze_board_state. Board is in invalid state\n')

        try: 
            analysis_result = self.engine.analyse(board, game_settings.search_limit, multipv=game_settings.num_moves_to_return)
        except Exception as e:
            self.errors_file.write(f'@ Bradley_analyze_board_state. An error occurred during analysis: {e}\n')
            self.errors_file.write(f"Chessboard is:\n{board}\n")
            raise Exception from e

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
            self.errors_file.write(f'An error occurred while extracting scores: {e}\n')
            raise Exception from e

        try:
            # Extract the anticipated next move from the analysis
            anticipated_next_move = analysis_result[0]['pv'][0]
        except Exception as e:
            self.errors_file.write(f'An error occurred while extracting the anticipated next move: {e}\n')
            raise Exception from e
        
        return {
            'mate_score': mate_score,
            'centipawn_score': centipawn_score,
            'anticipated_next_move': anticipated_next_move
        }
    ### end of analyze_board_state
 
    def get_reward(self, chess_move: str) -> int:
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
            None.

        Side Effects:
            None.
        """

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

    def identify_corrupted_games(self) -> None:
        """
        Identifies corrupted games in the chess database and logs them in the errors file.

        This method iterates over each game in the chess database and tries to play through the game using the 
        reinforcement learning agents. If an error occurs at any point during the game, the game number is added to 
        the list of corrupted games and the error is logged in the errors file.

        The method first tries to get the current state of the game. If an error occurs, it logs the error and the 
        current board state in the errors file, adds the game number to the list of corrupted games, and moves on to 
        the next game.

        The method then enters a loop where it alternates between the white and black agents choosing and playing 
        moves. If an error occurs while choosing or playing a move, the method logs the error and the current state 
        in the errors file, adds the game number to the list of corrupted games, and breaks out of the loop to move 
        on to the next game.

        After each move, the method tries to get the latest state of the game. If an error occurs, it logs the error 
        and the current board state in the errors file, adds the game number to the list of corrupted games, and 
        breaks out of the loop to move on to the next game.

        The loop continues until the game is over, there are no more legal moves, or the maximum number of moves for 
        the current training game has been reached.

        After each game, the method resets the environment to prepare for the next game. It also prints a progress 
        notification every 1000 games.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Modifies the list of corrupted games and writes to the errors file if an error occurs.
        """

        ### FOR EACH GAME IN THE CHESS DB ###
        game_count = 0
        for game_num_str in self.chess_data.index:
            start_time = time.time()
            num_chess_moves_curr_training_game: int = self.chess_data.at[game_num_str, 'PlyCount']

            try:
                curr_state = self.environ.get_curr_state()
            except Exception as e:
                self.errors_file.write(f'An error occurred at self.environ.get_curr_state: {e}\n')
                self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                self.errors_file.write(f'at game: {game_num_str}\n')
                self.corrupted_games_list.append(game_num_str)
                self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                break

            ### LOOP PLAYS THROUGH ONE GAME ###
            while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                ##################### WHITE'S TURN ####################
                W_chess_move = self.W_rl_agent.choose_action(curr_state, game_num_str)
                if not W_chess_move:
                    self.errors_file.write(f'An error occurred at self.W_rl_agent.choose_action\n')
                    self.errors_file.write(f'W_chess_move is empty at state: {curr_state}\n')
                    self.errors_file.write(f'at game: {game_num_str}\n')
                    self.corrupted_games_list.append(game_num_str)
                    self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                    break # and go to the next game. this game is over.

                ### WHITE AGENT PLAYS THE SELECTED MOVE ###
                try:
                    self.rl_agent_plays_move(W_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    self.errors_file.write(f'at game: {game_num_str}\n')
                    self.corrupted_games_list.append(game_num_str)
                    self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                    break # and go to the next game. this game is over.

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at get_curr_state: {e}\n')
                    self.errors_file.write(f'curr board is:\n{self.environ.board}\n\n')
                    self.errors_file.write(f'at game: {game_num_str}\n')
                    self.corrupted_games_list.append(game_num_str)
                    self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                
                if self.environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                    break # and go to next game

                ##################### BLACK'S TURN ####################
                B_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
                if not B_chess_move:
                    self.errors_file.write(f'An error occurred at self.W_rl_agent.choose_action\n')
                    self.errors_file.write(f'B_chess_move is empty at state: {curr_state}\n')
                    self.errors_file.write(f'at: {game_num_str}\n')
                    self.corrupted_games_list.append(game_num_str)
                    self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                    break # game is over, go to next game.

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                try:
                    self.rl_agent_plays_move(B_chess_move, game_num_str)
                except Exception as e:
                    self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                    self.errors_file.write(f'at {game_num_str}\n')
                    self.corrupted_games_list.append(game_num_str)
                    self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                    break 

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at environ.get_curr_state: {e}\n')
                    self.errors_file.write(f'at: {game_num_str}\n')
                    self.corrupted_games_list.append(game_num_str)
                    self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                    break

                if self.environ.board.is_game_over() or not curr_state['legal_moves']:
                    break # and go to next game

                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred: {e}\n')
                    self.errors_file.write("failed to get_curr_state\n") 
                    self.errors_file.write(f'at: {game_num_str}\n')
                    self.corrupted_games_list.append(game_num_str)
                    self.errors_file.write(f'corrupt games list is: {self.corrupted_games_list}\n')
                    break
            ### END OF CURRENT GAME LOOP ###

            # this curr game is done, reset environ to prepare for the next game
            self.environ.reset_environ() # reset and go to next game in chess database
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Print progress notification every 1000 games
            if game_count % 1000 == 0:
                print(f"Notification: Game {game_count} is done. Time elapsed: {elapsed_time:.2f} seconds.")
            game_count += 1
        
        ### END OF FOR LOOP THROUGH CHESS DB ###
    # end of identify_corrupted_games

    def generate_Q_est_df(self, q_est_vals_file_path) -> None:
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
            q_est_vals_file_path (str): The path to the file where the estimated Q-values will be written.

        Returns:
            None.

        Raises:
            None.

        Side Effects:
            Writes to the errors file if an error occurs.
            Writes to the Q-values file.
            Modifies the current state of the environment.
        """
        
        q_est_vals_file = open(q_est_vals_file_path, 'a')

        try:
            ### FOR EACH GAME IN THE TRAINING SET ###
            for game_num_str in self.chess_data.index:
                num_chess_moves_curr_training_game: int = self.chess_data.at[game_num_str, 'PlyCount']

                try:
                    curr_state = self.environ.get_curr_state()
                except Exception as e:
                    self.errors_file.write(f'An error occurred at self.environ.get_curr_state: {e}\n')
                    self.errors_file.write(f'at: {game_num_str}\n')
                    break
                
                q_est_vals_file.write(f'{game_num_str}\n')

                ### LOOP PLAYS THROUGH ONE GAME ###
                while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                    ##################### WHITE'S TURN ####################
                    # choose action a from state s, using policy
                    W_chess_move = self.W_rl_agent.choose_action(curr_state, game_num_str)
                    if not W_chess_move:
                        self.errors_file.write(f'An error occurred at self.W_rl_agent.choose_action\n')
                        self.errors_file.write(f'W_chess_move is empty at state: {curr_state}\n')
                        break

                    # assign curr turn to new var for now. once agent plays move, turn will be updated, but we need 
                    # to track the turn before so that the est q value can be assigned to the correct column.
                    curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])

                    ### WHITE AGENT PLAYS THE SELECTED MOVE ###
                    # take action a, observe r, s', and load chessboard
                    try:
                        self.rl_agent_plays_move(W_chess_move, game_num_str)
                    except Exception as e:
                        self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                        self.errors_file.write(f'at: {game_num_str}\n')
                        break # and go to the next game. this game is over.

                    # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                    try:
                        curr_state = self.environ.get_curr_state()
                    except Exception as e:
                        self.errors_file.write(f'An error occurred at get_curr_state: {e}\n')
                        self.errors_file.write(f'at: {game_num_str}\n')
                        break
                    
                    # find the estimated Q value for White, but first check if game ended
                    if self.environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                        break # and go to next game
                    else: # current game continues
                        try:
                            W_est_Qval: int = self.find_estimated_Q_value()
                            q_est_vals_file.write(f'{curr_turn_for_q_est}, {W_est_Qval}\n')
                        except Exception as e:
                            self.errors_file.write(f'An error occurred while retrieving W_est_Qval: {e}\n')
                            self.errors_file.write(f"at White turn, failed to find_estimated_Q_value\n")
                            self.errors_file.write(f'curr state is:{curr_state}\n')
                            break

                    ##################### BLACK'S TURN ####################
                    # choose action a from state s, using policy
                    B_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
                    if not B_chess_move:
                        self.errors_file.write(f'An error occurred at self.W_rl_agent.choose_action\n')
                        self.errors_file.write(f'B_chess_move is empty at state: {curr_state}\n')
                        self.errors_file.write(f'at: {game_num_str}\n')
                        break

                    # assign curr turn to new var for now. once agent plays move, turn will be updated, but we need 
                    # to track the turn before so that the est q value can be assigned to the correct column.
                    curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])
                    
                    ##### BLACK AGENT PLAYS SELECTED MOVE #####
                    # take action a, observe r, s', and load chessboard
                    try:
                        self.rl_agent_plays_move(B_chess_move, game_num_str)
                    except Exception as e:
                        self.errors_file.write(f'An error occurred at rl_agent_plays_move: {e}\n')
                        self.errors_file.write(f'at: {game_num_str}\n')
                        break 

                    # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                    try:
                        curr_state = self.environ.get_curr_state()
                    except Exception as e:
                        self.errors_file.write(f'An error occurred at environ.get_curr_state: {e}\n')
                        self.errors_file.write(f'at: {game_num_str}\n')
                        break

                    # find the estimated Q value for Black, but first check if game ended
                    if self.environ.board.is_game_over() or not curr_state['legal_moves']:
                        break # and go to next game
                    else: # current game continues
                        try:
                            B_est_Qval: int = self.find_estimated_Q_value()
                            q_est_vals_file.write(f'{curr_turn_for_q_est}, {B_est_Qval}\n') 
                        except Exception as e:
                            self.errors_file.write(f"at Black turn, failed to find_estimated_Qvalue because error: {e}\n")
                            self.errors_file.write(f'curr state is :{curr_state}\n')
                            self.errors_file.write(f'at : {game_num_str}\n')
                            break

                    try:
                        curr_state = self.environ.get_curr_state()
                    except Exception as e:
                        self.errors_file.write(f'An error occurred: {e}\n')
                        self.errors_file.write("failed to get_curr_state\n") 
                        self.errors_file.write(f'at: {game_num_str}\n')
                        break
                ### END OF CURRENT GAME LOOP ###

                # create a newline between games in the Q_est log file.
                q_est_vals_file.write('\n')

                self.environ.reset_environ() # reset and go to next game in training set
        finally:
            self.engine.quit()