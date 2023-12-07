import re
import pandas as pd
import chess.pgn
import chess
import json
import game_settings
import time


def load_pgn_file(input_file_path):
    """Loads a PGN file and returns a list of games.
    Args:
        input_file_path: The path to the input PGN file.
    Returns:
        A list of games.
    """
    games = []
    with open(input_file_path, 'r', encoding='utf-8') as file:
        while True:
            game = chess.pgn.read_game(file)
            if game is None:
                break
            games.append(game)
    return games

# Clean up the PGN file and save the results to a JSON file.
def clean_up_pgn_file(input_file_path, output_file_path):
    """Cleans up a PGN file with multiple games and saves only the PlyCount, 
    the Result, and the chess moves for each game, in the form of an array.
    Args:
        input_file_path: The path to the input PGN file.
        output_file_path: The path to the output file.
    """
    games = load_pgn_file(input_file_path)
    cleaned_games = []
    for game in games:
        cleaned_game = {}
        # Save the PlyCount.
        cleaned_game["PlyCount"] = game.end().board().ply()
        # Save the Result.
        cleaned_game["Result"] = game.headers["Result"]
        # save the chess moves in SAN
        board = game.board()
        san_moves = []

        for move in game.mainline_moves():
            san_moves.append(board.san(move))
            board.push(move)
        cleaned_game["ChessMoves"] = san_moves
        cleaned_games.append(cleaned_game)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(cleaned_games, file, indent=4)


if __name__ == '__main__':
    start_time = time.time()

    # clean_up_pgn_file(game_settings.chess_pgn_file_path_1, game_settings.chess_json_file_path_part_1)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_2, game_settings.chess_json_file_path_part_2)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_3, game_settings.chess_json_file_path_part_3)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_4, game_settings.chess_json_file_path_part_4)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_5, game_settings.chess_json_file_path_part_5)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_6, game_settings.chess_json_file_path_part_6)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_7, game_settings.chess_json_file_path_part_7)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_8, game_settings.chess_json_file_path_part_8)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_9, game_settings.chess_json_file_path_part_9)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_10, game_settings.chess_json_file_path_part_10)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_11, game_settings.chess_json_file_path_part_11)

    clean_up_pgn_file(game_settings.chess_pgn_file_path_12, game_settings.chess_json_file_path_part_12)
    
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_13, game_settings.chess_json_file_path_part_13)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_14, game_settings.chess_json_file_path_part_14)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_15, game_settings.chess_json_file_path_part_15)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_16, game_settings.chess_json_file_path_part_16)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_17, game_settings.chess_json_file_path_part_17)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_18, game_settings.chess_json_file_path_part_18)
    # clean_up_pgn_file(game_settings.chess_pgn_file_path_19, game_settings.chess_json_file_path_part_19)
    end_time = time.time()

    total_time = end_time - start_time
     
    print('pgn to json conversion is complete\n')
    print(f'it took: {total_time} seconds')

    
    
