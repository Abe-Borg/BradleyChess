# custom_exceptions.py

class ChessError(Exception):
    """Base class for exceptions in the chess application."""
    def __init__(self, message="An error occurred in the chess application"):
        super().__init__(message)

class ChessboardError(ChessError):
    """Base class for exceptions related to the chessboard."""
    def __init__(self, message="An error occurred with the chessboard"):
        super().__init__(message)

class ChessboardLoadError(ChessboardError):
    """Exception raised when there's an error loading a move onto the chessboard."""
    def __init__(self, move, message=None):
        self.move = move
        self.message = message or f"Error loading move '{move}' onto chessboard"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move})"

class ChessboardPopError(ChessboardError):
    """Exception raised when there's an error removing a move from the chessboard."""
    def __init__(self, move, message=None):
        self.move = move
        self.message = message or f"Error removing {move} from chessboard"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move})"

class ChessboardManipulationError(ChessboardError):
    """Exception raised when there's an error manipulating the chessboard."""
    def __init__(self, action, message=None):
        self.action = action
        self.message = message or f"Error during chessboard manipulation: {action}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Action: {self.action})"

class StateError(ChessError):
    """Base class for exceptions related to the game state."""
    pass

class StateUpdateError(StateError):
    """Exception raised when there's an error updating the current state of the environment."""
    def __init__(self, current_state, message=None):
        self.current_state = current_state
        self.message = message or "Error updating current state"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Current State: {self.current_state})"

class StateRetrievalError(StateError):
    """Exception raised when there's an error retrieving the current state of the environment."""
    def __init__(self, message="Error retrieving current state"):
        super().__init__(message)

class IllegalMoveError(ChessError):
    """Exception raised when an illegal move is attempted."""
    def __init__(self, move, message=None):
        self.move = move
        self.message = message or f"Illegal move: {move}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move})"

class GameError(ChessError):
    """Base class for exceptions related to game flow."""
    pass

class GameOverError(GameError):
    """Exception raised when trying to make a move in a finished game."""
    def __init__(self, message="The game is already over"):
        super().__init__(message)

class NoLegalMovesError(GameError):
    """Exception raised when there are no legal moves available."""
    def __init__(self, message="No legal moves available"):
        super().__init__(message)

class GameOutcomeError(GameError):
    """Exception raised when the game outcome cannot be determined."""
    def __init__(self, message="Unable to determine game outcome"):
        super().__init__(message)

class GameTerminationError(GameError):
    """Exception raised when the game termination reason cannot be determined."""
    def __init__(self, message="Unable to determine game termination reason"):
        super().__init__(message)

class TrainingError(ChessError):
    """Exception raised for errors during the training process."""
    def __init__(self, stage, message=None):
        self.stage = stage
        self.message = message or f"An error occurred during the training process: {stage}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Training Stage: {self.stage})"

class QTableError(ChessError):
    """Base class for exceptions related to Q-table operations."""
    pass

class QTableUpdateError(QTableError):
    """Exception raised when there's an error updating the Q-table."""
    def __init__(self, move, turn, message=None):
        self.move = move
        self.turn = turn
        self.message = message or f"Error updating Q-table for move '{move}' at turn {turn}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Move: {self.move}, Turn: {self.turn})"

class QValueCalculationError(QTableError):
    """Exception raised when there's an error calculating the Q-value."""
    def __init__(self, params, message=None):
        self.params = params
        self.message = message or "Error calculating Q-value"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Parameters: {self.params})"

class AnalysisError(ChessError):
    """Base class for exceptions related to board analysis."""
    pass

class BoardAnalysisError(AnalysisError):
    """Exception raised when there's an error analyzing the board state."""
    def __init__(self, board_state, message=None):
        self.board_state = board_state
        self.message = message or "Error analyzing board state"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Board State: {self.board_state})"

class InvalidBoardStateError(AnalysisError):
    """Exception raised when the chess board is in an invalid state."""
    def __init__(self, board_state, message=None):
        self.board_state = board_state
        self.message = message or "Chess board is in an invalid state"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Board State: {self.board_state})"

class EngineAnalysisError(AnalysisError):
    """Exception raised when there's an error during engine analysis."""
    def __init__(self, engine, message=None):
        self.engine = engine
        self.message = message or f"Error occurred during chess engine analysis with {engine}"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Engine: {self.engine})"

class ScoreExtractionError(AnalysisError):
    """Exception raised when there's an error extracting scores from analysis."""
    def __init__(self, analysis_result, message=None):
        self.analysis_result = analysis_result
        self.message = message or "Error extracting scores from analysis"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Analysis Result: {self.analysis_result})"

class MoveExtractionError(AnalysisError):
    """Exception raised when there's an error extracting the anticipated move from analysis."""
    def __init__(self, analysis_result, message=None):
        self.analysis_result = analysis_result
        self.message = message or "Error extracting anticipated move from analysis"
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} (Analysis Result: {self.analysis_result})"



class AgentError(ChessError):
    """Base class for exceptions in the Agent class."""
    pass

class InvalidActionError(AgentError):
    """Exception raised when an invalid action is chosen by agent."""
    def __init__(self, action, message=None):
        self.action = action
        self.message = message or f"Invalid action chosen: {action}"
        super().__init__(self.message)

class QTableAccessError(AgentError):
    """Exception raised when there's an error accessing the Q-table."""
    pass

class FailureToChooseActionError(AgentError):
    """Exception raised when the agent fails to choose an action."""
    def __init__(self, message="Agent failed to choose an action"):
        super().__init__(message)

class AgentInitializationError(AgentError):
    """Exception raised when there's an error initializing the agent."""
    def __init__(self, message="Error initializing agent"):
        super().__init__(message)

class EnvironError(ChessError):
    """Base class for exceptions in the Environ class."""
    pass

class TurnIndexError(EnvironError):
    """Exception raised when there's an issue with the turn index."""
    pass

class InvalidMoveError(EnvironError):
    """Exception raised when an invalid move is attempted."""
    pass

class HelperMethodError(ChessError):
    """Base class for exceptions in helper methods."""
    pass

class EngineStartError(HelperMethodError):
    """Exception raised when there's an error starting the chess engine."""
    pass

class RewardCalculationError(HelperMethodError):
    """Exception raised when there's an error calculating rewards."""
    pass

class TrainingFunctionError(ChessError):
    """Base class for exceptions in training functions."""
    pass

class QValueEstimationError(TrainingFunctionError):
    """Exception raised when there's an error estimating Q-values."""
    pass

class GameSimulationError(TrainingFunctionError):
    """Exception raised when there's an error simulating games during training."""
    pass

class EmptyChessMoveError(TrainingFunctionError):
    """Exception raised when an empty chess move is attempted."""
    pass

class GamePlayError(ChessError):
    """Exception raised when there's an error playing games between agents and human v agents."""
    pass