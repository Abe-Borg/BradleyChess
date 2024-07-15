import Environ
import Agent
import game_settings
import chess
import pandas as pd
import re
import copy
import time
import custom_exceptions
import sys
from multiprocessing import Pool, cpu_count
import cProfile
import pstats
import io
import functools
import logging
import helper_methods

class Bradley:
    """
        Acts as the single point of communication between the RL agent and the player.
        This class trains the agent and helps to manage the chessboard during play between the computer and the user.

        Args:
            none
        Attributes:
            environ (Environ.Environ): An Environ object representing the chessboard environment.
    """
    def __init__(self):
        self.error_logger = logging.getLogger(__name__)
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(game_settings.bradley_errors_filepath)
        self.error_logger.addHandler(error_handler)

        self.initial_training_logger = logging.getLogger(__name__ + '.initial_training')
        self.initial_training_logger.setLevel(logging.INFO)
        initial_training_handler = logging.FileHandler(game_settings.initial_training_results_filepath)
        self.initial_training_logger.addHandler(initial_training_handler)

        self.additional_training_logger = logging.getLogger(__name__ + '.additional_training')
        self.additional_training_logger.setLevel(logging.INFO)
        additional_training_handler = logging.FileHandler(game_settings.additional_training_results_filepath)
        self.additional_training_logger.addHandler(additional_training_handler)

        self.step_by_step_logger = logging.getLogger(__name__ + '.step_by_step')
        self.step_by_step_logger.setLevel(logging.DEBUG)
        step_by_step_handler = logging.FileHandler(game_settings.bradley_step_by_step_filepath)
        self.step_by_step_logger.addHandler(step_by_step_handler)

        self.environ = Environ.Environ()       
    ### end of Bradley constructor ###

    def __del__(self):
        # Remove handlers from loggers to ensure they're properly closed
        for logger in [self.error_logger, self.initial_training_logger, self.additional_training_logger, self.step_by_step_logger]:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
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
        """
        try:
            self.environ.load_chessboard(chess_move)
        except custom_exceptions.ChessboardLoadError as e:
            self.error_logger.error("hello from Bradley.receive_opp_move, an error occurred\n")
            self.error_logger.error(f'Error: {e}, failed to load chessboard with move: {chess_move}\n')
            return False

        try:
            self.environ.update_curr_state()
            return True
        except custom_exceptions.StateUpdateError as e:
            self.error_logger.error(f'hello from Bradley.receive_opp_move, an error occurrd\n')
            self.error_logger.error(f'Error: {e}, failed to update_curr_state\n') 
            raise Exception from e
    ### end of receive_opp_move ###

    def rl_agent_selects_chess_move(self, chess_agent) -> str:
        """
        The Agent selects a chess move and loads it onto the chessboard.
        This method allows the agent to select a chess move and load it onto the 
        chessboard. It is used during actual gameplay between the computer and the user, not during training. The 
        method first gets the current state of the environment. If the list of legal moves is empty, an exception 
        is raised. Depending on the color of the RL agent, the appropriate agent selects a move. The selected move 
        is then loaded onto the chessboard and the current state of the environment is updated.

        Args:
        Returns:
            str: A string representing the selected chess move.
        Raises:
            StateUpdateError: If the current state is not valid or fails to update.
            NoLegalMovesError: If the list of legal moves is empty.
            ChessboardLoadError: If the chessboard fails to load the move.
            StateRetrievalError: If the current state is not valid or fails to retrieve.
        Side Effects:
            Modifies the chessboard and the current state of the environment by loading the chess move and updating 
            the current state.
        """
        try:
            curr_state = self.environ.get_curr_state()
        except custom_exceptions.StateRetrievalError as e:
            self.error_logger.error("hello from Bradley.rl_agent_selects_chess_move, an error occurred\n")
            self.error_logger.error(f'Error: {e}, failed to get_curr_state\n')
            raise Exception from e
        
        if curr_state['legal_moves'] == []:
            self.error_logger.error('hello from Bradley.rl_agent_selects_chess_move, legal_moves is empty\n')
            self.error_logger.error(f'curr state is: {curr_state}\n')
            raise custom_exceptions.NoLegalMovesError(f'hello from Bradley.rl_agent_selects_chess_move, legal_moves is empty\n')
        
        chess_move: str = chess_agent.choose_action(curr_state)

        try:
            self.environ.load_chessboard(chess_move)
        except custom_exceptions.ChessboardLoadError as e:
            self.error_logger.error('hello from Bradley.rl_agent_selects_chess_move\n')
            self.error_logger.error(f'Error {e}: failed to load chessboard with move: {chess_move}\n')
            raise Exception from e

        try:
            self.environ.update_curr_state()
            return chess_move
        except custom_exceptions.StateUpdateError as e:
            self.error_logger.error('hello from Bradley.rl_agent_selects_chess_move\n')
            self.error_logger.error(f'Error: {e}, failed to update_curr_state\n')
            raise Exception from e
    ### end of rl_agent_selects_chess_move
        
    def continue_training_rl_agents(self, num_games_to_play: int) -> None:
        """ continues to train the agent, this time the agents make their own decisions instead 
            of playing through the database.
        """ 
        ### placeholder, will implement this function later.
    ### end of continue_training_rl_agents
    
    def assign_points_to_Q_table(chess_move: str, curr_turn: str, curr_q_val: int, chess_agent) -> None:
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
                QTableUpdateError: is raised if the chess move is not represented in the Q table. The exception is 
                written to the errors file.

            Side Effects:
                Modifies the Q table of the RL agent by assigning points to the given chess move.
                Writes to the errors file if a exception is raised.
        """
        try:
            chess_agent.change_Q_table_pts(chess_move, curr_turn, curr_q_val)
        except custom_exceptions.QTableUpdateError as e: 
            # chess move is not represented in the Q table, update Q table and try again.
            # self.error_logger.error(f'caught exception: {e} at assign_points_to_Q_table\n')
            # self.error_logger.error(f'Chess move is not represented in the White Q table, updating Q table and trying again...\n')
            chess_agent.update_Q_table([chess_move])
            chess_agent.change_Q_table_pts(chess_move, curr_turn, curr_q_val)
    # enf of assign_points_to_Q_table

    def rl_agent_plays_move(chess_move: str, curr_game, environ) -> None:
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
            environ.load_chessboard(chess_move, curr_game)
        except custom_exceptions.ChessboardLoadError as e:
            # self.error_logger.error(f'at Bradley.rl_agent_plays_move. An error occurred at {curr_game}: {e}\n')
            # self.error_logger.error(f"failed to load_chessboard with move {chess_move}\n")
            raise Exception from e

        try:
            environ.update_curr_state()
        except custom_exceptions.StateUpdateError as e:
            # self.error_logger.error(f'at Bradley.rl_agent_plays_move. update_curr_state() failed to increment turn_index, Caught exception: {e}\n')
            # self.error_logger.error(f'Current state is: {environ.get_curr_state()}\n')
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
            BoardAnalysisError: An exception is raised if an error occurs while analyzing the board state for the estimated Q-value
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
            analysis_results = self.analyze_board_state(self.environ.board)
        except Exception as e:
            self.error_logger.error(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.error_logger.error(f'failed to analyze_board_state\n')
            raise Exception from e
        
        # load up the chess board with opponent's anticipated chess move 
        try:
            self.environ.load_chessboard_for_Q_est(analysis_results)
        except Exception as e:
            self.error_logger.error(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.error_logger.error(f'failed to load_chessboard_for_Q_est\n')
            raise Exception from e
        
        # check if the game would be over with the anticipated next move
        if self.environ.board.is_game_over() or not self.environ.get_legal_moves():
            try:
                self.environ.pop_chessboard()
            except Exception as e:
                self.error_logger.error(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
                self.error_logger.error(f'failed at self.environ.pop_chessboard\n')
                raise Exception from e
            return 1 # just return some value, doesn't matter.
            
        # this is the Q estimated value due to what the opposing agent is likely to play in response to our move.    
        try:
            est_Qval_analysis = self.analyze_board_state(self.environ.board)
        except Exception as e:
            self.error_logger.error(f'at Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.error_logger.error(f'failed at self.analyze_board_state\n')
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
            self.error_logger.error(f'@ Bradley.find_estimated_Q_value. An error occurred: {e}\n')
            self.error_logger.error("failed to pop_chessboard\n")
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
            QValueCalculationError: If an error or overflow occurs during the calculation of the next Q-value.

        Side Effects:
            None.
        """
        try:
            next_Qval = int(curr_Qval + learn_rate * (reward + ((discount_factor * est_Qval) - curr_Qval)))
            return next_Qval
        except OverflowError:
            self.error_logger.error(f'@ Bradley.find_next_Qval. An error occurred: OverflowError\n')
            self.error_logger.error(f'curr_Qval: {curr_Qval}\n')
            self.error_logger.error(f'learn_rate: {learn_rate}\n')
            self.error_logger.error(f'reward: {reward}\n')
            self.error_logger.error(f'discount_factor: {discount_factor}\n')
            self.error_logger.error(f'est_Qval: {est_Qval}\n')
            raise custom_exceptions.QValueCalculationError("Overflow occurred during Q-value calculation") from OverflowError
    # end of find_next_Qval
    
    def analyze_board_state(self, board: chess.Board, engine) -> dict:
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
        if not self.environ.board.is_valid():
            self.error_logger.error(f'at Bradley.analyze_board_state. Board is in invalid state\n')
            raise custom_exceptions.InvalidBoardStateError(f'at Bradley.analyze_board_state. Board is in invalid state\n')

        try: 
            analysis_result = engine.analyse(board, game_settings.search_limit, multipv=game_settings.num_moves_to_return)
        except Exception as e:
            self.error_logger.error(f'@ Bradley_analyze_board_state. An error occurred during analysis: {e}\n')
            self.error_logger.error(f"Chessboard is:\n{board}\n")
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
            self.error_logger.error(f'An error occurred while extracting scores: {e}\n')
            raise custom_exceptions.ScoreExtractionError("Error occurred while extracting scores from analysis") from e

        try:
            # Extract the anticipated next move from the analysis
            anticipated_next_move = analysis_result[0]['pv'][0]
        except Exception as e:
            self.error_logger.error(f'An error occurred while extracting the anticipated next move: {e}\n')
            raise custom_exceptions.MoveExtractionError("Error occurred while extracting the anticipated next move") from e
        
        return {
            'mate_score': mate_score,
            'centipawn_score': centipawn_score,
            'anticipated_next_move': anticipated_next_move
        }
    ### end of analyze_board_state
 
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

        try:
            ### FOR EACH GAME IN THE TRAINING SET ###
            for game_num_str in chess_data.index:
                num_chess_moves_curr_training_game: int = chess_data.at[game_num_str, 'PlyCount']

                try:
                    curr_state = environ.get_curr_state()
                except Exception as e:
                    # self.error_logger.error(f'An error occurred at self.environ.get_curr_state: {e}\n')
                    # self.error_logger.error(f'at: {game_num_str}\n')
                    break
                
                q_est_vals_file.write(f'{game_num_str}\n')

                ### LOOP PLAYS THROUGH ONE GAME ###
                while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                    ##################### WHITE'S TURN ####################
                    # choose action a from state s, using policy
                    w_chess_move = w_agent.choose_action(curr_state, game_num_str)
                    if not w_chess_move:
                        # self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                        # self.error_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
                        break

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
                        break # and go to the next game. this game is over.

                    # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                    try:
                        curr_state = environ.get_curr_state()
                    except Exception as e:
                        # self.error_logger.error(f'An error occurred at get_curr_state: {e}\n')
                        # self.error_logger.error(f'at: {game_num_str}\n')
                        break
                    
                    # find the estimated Q value for White, but first check if game ended
                    if environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                        break # and go to next game
                    else: # current game continues
                        try:
                            W_est_Qval: int = self.find_estimated_Q_value()
                            q_est_vals_file.write(f'{curr_turn_for_q_est}, {W_est_Qval}\n')
                        except Exception as e:
                            self.error_logger.error(f'An error occurred while retrieving W_est_Qval: {e}\n')
                            self.error_logger.error(f"at White turn, failed to find_estimated_Q_value\n")
                            self.error_logger.error(f'curr state is:{curr_state}\n')
                            break

                    ##################### BLACK'S TURN ####################
                    # choose action a from state s, using policy
                    b_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
                    if not b_chess_move:
                        self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                        self.error_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
                        self.error_logger.error(f'at: {game_num_str}\n')
                        break

                    # assign curr turn to new var for now. once agent plays move, turn will be updated, but we need 
                    # to track the turn before so that the est q value can be assigned to the correct column.
                    curr_turn_for_q_est = copy.copy(curr_state['curr_turn'])
                    
                    ##### BLACK AGENT PLAYS SELECTED MOVE #####
                    # take action a, observe r, s', and load chessboard
                    try:
                        self.rl_agent_plays_move(b_chess_move, game_num_str)
                    except Exception as e:
                        self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
                        self.error_logger.error(f'at: {game_num_str}\n')
                        break 

                    # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                    try:
                        curr_state = self.environ.get_curr_state()
                    except Exception as e:
                        self.error_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
                        self.error_logger.error(f'at: {game_num_str}\n')
                        break

                    # find the estimated Q value for Black, but first check if game ended
                    if self.environ.board.is_game_over() or not curr_state['legal_moves']:
                        break # and go to next game
                    else: # current game continues
                        try:
                            B_est_Qval: int = self.find_estimated_Q_value()
                            q_est_vals_file.write(f'{curr_turn_for_q_est}, {B_est_Qval}\n') 
                        except Exception as e:
                            self.error_logger.error(f"at Black turn, failed to find_estimated_Qvalue because error: {e}\n")
                            self.error_logger.error(f'curr state is :{curr_state}\n')
                            self.error_logger.error(f'at : {game_num_str}\n')
                            break

                    try:
                        curr_state = self.environ.get_curr_state()
                    except Exception as e:
                        self.error_logger.error(f'An error occurred: {e}\n')
                        self.error_logger.error("failed to get_curr_state\n") 
                        self.error_logger.error(f'at: {game_num_str}\n')
                        break
                ### END OF CURRENT GAME LOOP ###

                # create a newline between games in the Q_est log file.
                q_est_vals_file.write('\n')

                self.environ.reset_environ() # reset and go to next game in training set
        except Exception as e:
            self.error_logger.error(f'An error occurred at generate_Q_est_df: {e}\n')
    # end of generate_Q_est_df

    def simply_play_games(self) -> None:
        """
            -
        """
        if game_settings.PRINT_STEP_BY_STEP:
            self.step_by_step_logger.debug(f'hi from simply_play_games\n')
            self.step_by_step_logger.debug(f'White Q table size before games: {w_agent.q_table.shape}\n')
            self.step_by_step_logger.debug(f'Black Q table size before games: {self.B_rl_agent.q_table.shape}\n')
        
        ### FOR EACH GAME IN THE CHESS DB ###
        game_count = 0
        for game_num_str in game_settings.chess_data.index:
            start_time = time.time()
            
            num_chess_moves_curr_training_game: int = game_settings.chess_data.at[game_num_str, 'PlyCount']

            if game_settings.PRINT_STEP_BY_STEP:
                self.step_by_step_logger.debug(f'game_num_str is: {game_num_str}\n')

            try:
                curr_state = self.environ.get_curr_state()
                
                if game_settings.PRINT_STEP_BY_STEP:
                    self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
            except Exception as e:
                self.error_logger.error(f'An error occurred at self.environ.get_curr_state: {e}\n')
                self.error_logger.error(f'curr board is:\n{self.environ.board}\n\n')
                self.error_logger.error(f'at game: {game_num_str}\n')
                break

            ### LOOP PLAYS THROUGH ONE GAME ###
            while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                ##################### WHITE'S TURN ####################
                w_chess_move = w_agent.choose_action(curr_state, game_num_str)

                if game_settings.PRINT_STEP_BY_STEP:
                    self.step_by_step_logger.debug(f'w_chess_move is: {w_chess_move}\n')

                if not w_chess_move:
                    self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                    self.error_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
                    self.error_logger.error(f'at game: {game_num_str}\n')
                    break # and go to the next game. this game is over.

                ### WHITE AGENT PLAYS THE SELECTED MOVE ###
                try:
                    self.rl_agent_plays_move(w_chess_move, game_num_str)
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'White played move: {w_chess_move}\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
                    self.error_logger.error(f'at game: {game_num_str}\n')
                    break # and go to the next game. this game is over.

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()

                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at get_curr_state: {e}\n')
                    self.error_logger.error(f'curr board is:\n{self.environ.board}\n\n')
                    self.error_logger.error(f'at game: {game_num_str}\n')
                
                if self.environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'game is over\n')
                        self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
                    break # and go to next game

                ##################### BLACK'S TURN ####################
                b_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
                
                if game_settings.PRINT_STEP_BY_STEP:
                    self.step_by_step_logger.debug(f'Black chess move: {b_chess_move}\n')

                if not b_chess_move:
                    self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                    self.error_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
                    self.error_logger.error(f'at: {game_num_str}\n')
                    break # game is over, go to next game.

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                try:
                    self.rl_agent_plays_move(b_chess_move, game_num_str)
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'black agent played their move\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
                    self.error_logger.error(f'at {game_num_str}\n')
                    break 

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
                    self.error_logger.error(f'at: {game_num_str}\n')
                    break

                if self.environ.board.is_game_over() or not curr_state['legal_moves']:
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'game is over\n')
                    break # and go to next game
            ### END OF CURRENT GAME LOOP ###

            if game_settings.PRINT_STEP_BY_STEP:
                self.step_by_step_logger.debug(f'game {game_num_str} is over\n')
                self.step_by_step_logger.debug(f'agent q tables sizes are: \n')
                self.step_by_step_logger.debug(f'White Q table: {w_agent.q_table.shape}\n')
                self.step_by_step_logger.debug(f'Black Q table: {self.B_rl_agent.q_table.shape}\n')

            # this curr game is done, reset environ to prepare for the next game
            self.environ.reset_environ() # reset and go to next game in chess database
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Print progress notification every 1000 games
            if game_count % 1000 == 0:
                print(f"Notification: Game {game_count} is done. Time elapsed: {elapsed_time:.2f} seconds.")
            game_count += 1
        ### END OF FOR LOOP THROUGH CHESS DB ###