### VERSION 1 ###
# import os
# import chess.pgn
# import game_settings

# def split_pgn_file(file_path, number_of_splits = 10):
#     # Extract the base name of the file without the extension
#     base_name = os.path.splitext(os.path.basename(file_path))[0]

#     # Count total number of games
#     total_games = 0
#     with open(file_path, 'r') as pgn:
#         while chess.pgn.read_game(pgn):
#             total_games += 1

#     games_per_file = total_games // number_of_splits

#     current_game = 0
#     for i in range(number_of_splits):
#         split_filename = f'{base_name}_Part_{i+1}.pgn'
#         with open(split_filename, 'w') as split_file:
#             with open(file_path, 'r') as pgn:
#                 for j in range(games_per_file):
#                     if current_game < total_games:
#                         game = chess.pgn.read_game(pgn)
#                         if game is not None:
#                             split_file.write(str(game) + "\n\n")
#                             current_game += 1

# # Use the function
# split_pgn_file(game_settings.chess_pgn_file_path_12) 

# if __name__ == '__main__':
#     start_time = time.time()
    
#     split_pgn_file(game_settings.chess_pgn_file_path_12)

#     end_time = time.time()
#     print('PGN to DataFrame conversion is complete\n')
#     print(f'It took: {end_time - start_time} seconds')

### VERSION 2 ###
# import os
# import chess.pgn
# import game_settings
# import time

# def split_pgn_file(file_path, number_of_splits=10):
#     base_name = os.path.splitext(os.path.basename(file_path))[0]

#     total_games = 0
#     with open(file_path, 'r', encoding='utf-8') as pgn:  # Specify UTF-8 encoding
#         while chess.pgn.read_game(pgn):
#             total_games += 1

#     games_per_file = total_games // number_of_splits

#     current_game = 0
#     for i in range(number_of_splits):
#         split_filename = f'{base_name}_Part_{i+1}.pgn'
#         with open(split_filename, 'w', encoding='utf-8') as split_file:  # Specify UTF-8 encoding
#             with open(file_path, 'r', encoding='utf-8') as pgn:  # Specify UTF-8 encoding
#                 for j in range(games_per_file):
#                     if current_game < total_games:
#                         game = chess.pgn.read_game(pgn)
#                         if game is not None:
#                             split_file.write(str(game) + "\n\n")
#                             current_game += 1

# if __name__ == '__main__':
#     start_time = time.time()
    
#     split_pgn_file(game_settings.chess_pgn_file_path_12)

#     end_time = time.time()
#     print('PGN to DataFrame conversion is complete\n')
#     print(f'It took: {end_time - start_time} seconds')


import os
import chess.pgn
import game_settings
import time

def count_games_in_pgn(file_path):
    total_games = 0
    with open(file_path, 'r', encoding='utf-8') as pgn:
        while chess.pgn.read_game(pgn):
            total_games += 1
    return total_games

def split_pgn_file_by_games(file_path, number_of_splits):
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