# custom_exceptions.py

class ChessError(Exception):
    """Base class for exceptions in the chess application."""
    pass

class ChessboardLoadError(ChessError):
    """Exception raised when there's an error loading a move onto the chessboard."""
    def __init__(self, message="Error loading move onto chessboard"):
        self.message = message
        super().__init__(self.message)

class StateUpdateError(ChessError):
    """Exception raised when there's an error updating the current state of the environment."""
    def __init__(self, message="Error updating current state"):
        self.message = message
        super().__init__(self.message)

class StateRetrievalError(ChessError):
    """Exception raised when there's an error retrieving the current state of the environment."""
    def __init__(self, message="Error retrieving current state"):
        self.message = message
        super().__init__(self.message)

class IllegalMoveError(ChessError):
    """Exception raised when an illegal move is attempted."""
    def __init__(self, move, message=None):
        self.move = move
        self.message = message or f"Illegal move: {move}"
        super().__init__(self.message)

class GameOverError(ChessError):
    """Exception raised when trying to make a move in a finished game."""
    def __init__(self, message="The game is already over"):
        self.message = message
        super().__init__(self.message)

class QTableUpdateError(ChessError):
    """Exception raised when there's an error updating the Q-table."""
    def __init__(self, message="Error updating Q-table"):
        self.message = message
        super().__init__(self.message)

class NoLegalMovesError(ChessError):
    """Exception raised when there are no legal moves available."""
    def __init__(self, message="No legal moves available"):
        self.message = message
        super().__init__(self.message)

class GameOutcomeError(ChessError):
    """Exception raised when the game outcome cannot be determined."""
    def __init__(self, message="Unable to determine game outcome"):
        self.message = message
        super().__init__(self.message)

class GameTerminationError(ChessError):
    """Exception raised when the game termination reason cannot be determined."""
    def __init__(self, message="Unable to determine game termination reason"):
        self.message = message
        super().__init__(self.message)

class TrainingError(ChessError):
    """Exception raised for errors during the training process."""
    def __init__(self, message="An error occurred during the training process"):
        self.message = message
        super().__init__(self.message)

class StateError(ChessError):
    """Exception raised for errors related to the game state."""
    def __init__(self, message="An error occurred while managing the game state"):
        self.message = message
        super().__init__(self.message)

class MoveError(ChessError):
    """Exception raised for errors related to chess moves."""
    def __init__(self, message="An error occurred while processing a chess move"):
        self.message = message
        super().__init__(self.message)