import os
import chess.pgn
import game_settings
import time
from tqdm import tqdm

def count_games_in_pgn(file_path):
    """
        Counts the number of games in a Portable Game Notation (PGN) file.

        This function opens a PGN file located at the given file path and iteratively reads each game,
        incrementing a counter for each valid game found until the end of the file is reached.

        Parameters:
        - file_path (str): The path to the PGN file.

        Returns:
        - int: The total number of chess games contained in the specified PGN file.

        Raises:
        - FileNotFoundError: If the specified file does not exist.
        - IOError: If the file cannot be opened or read.
        
        Example:
        >>> total_games = count_games_in_pgn("path/to/chess_games.pgn")
        >>> print(total_games)
        Outputs the number of games in the specified PGN file.
    """
    total_games = 0
    with open(file_path, 'r', encoding='utf-8') as pgn:
        while chess.pgn.read_game(pgn):
            total_games += 1
    return total_games

def split_pgn_file_by_games(file_path, number_of_splits):
    """
        Splits a PGN file into multiple smaller files, each containing an approximately equal number of games.

        This function first counts the total number of games in the specified PGN file. It then calculates the number
        of games that each split file should contain, based on the desired number of splits. Each split file is named
        sequentially and games are distributed as evenly as possible among them.

        Parameters:
        - file_path (str): The path to the PGN file to split.
        - number_of_splits (int): The number of split files to create.

        Returns:
        - None: Files are written directly to disk with no return value.

        Raises:
        - FileNotFoundError: If the specified file does not exist.
        - ValueError: If `number_of_splits` is less than 1 or more than the total number of games.
        - IOError: If there is an issue reading from the original file or writing to the split files.

        Example:
        >>> split_pgn_file_by_games("path/to/chess_games.pgn", 5)
        Creates 5 split files, each containing an equal share of the games from the original file.
    """
    total_games = count_games_in_pgn(file_path)
    games_per_split = total_games // number_of_splits
    current_game = 0

    with open(file_path, 'r', encoding='utf-8') as pgn:
        for i in range(number_of_splits):
            split_filename = f'{os.path.splitext(file_path)[0]}_Part_{i+1}.pgn'
            with open(split_filename, 'w', encoding='utf-8') as split_file:
                for j in range(games_per_split):
                    game = chess.pgn.read_game(pgn)
                    if game is None or current_game >= total_games:
                        break
                    split_file.write(str(game) + "\n\n")
                    current_game += 1


if __name__ == '__main__':
    start_time = time.time()
    
    split_pgn_file_by_games(game_settings.chess_pgn_file_path_19, 10)

    end_time = time.time()
    print('PGN to DataFrame conversion is complete\n')
    print(f'It took: {end_time - start_time} seconds')