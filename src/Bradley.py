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

        self.environ = Environ.Environ()       
    ### end of Bradley constructor ###

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
        
    