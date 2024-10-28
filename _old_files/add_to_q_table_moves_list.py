import pandas as pd
import numpy as np
import helper_methods
import game_settings
import Bradley
import time

def add_to_q_table_moves_list(chess_data):
    start_time = time.time()
    bradley = Bradley.Bradley()

    helper_methods.bootstrap_agent_fill_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent_fill_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)


    print(f'White Q table size before games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size before games: {bradley.B_rl_agent.Q_table.shape}\n')

    try:
        all_white_moves = set()
        all_black_moves = set()

        for _, game in chess_data.iterrows():
            # Extract moves
            white_moves = game.filter(regex=r'^W\d+$').dropna().unique()
            black_moves = game.filter(regex=r'^B\d+$').dropna().unique()
        
            all_white_moves.update(white_moves)
            all_black_moves.update(black_moves)

        # Update Q-tables with new moves
        new_white_moves = list(all_white_moves - set(bradley.W_rl_agent.Q_table.index))
        new_black_moves = list(all_black_moves - set(bradley.B_rl_agent.Q_table.index))

        bradley.W_rl_agent.update_Q_table(new_white_moves)
        bradley.B_rl_agent.update_Q_table(new_black_moves)
        
        print(f'New white moves: {new_white_moves}')
        print(f'New black moves: {new_black_moves}\n')

        print(f'White Q table size after games: {bradley.W_rl_agent.Q_table.shape}')
        print(f'Black Q table size after games: {bradley.B_rl_agent.Q_table.shape}\n')

        end_time = time.time()
        total_time = end_time - start_time
        print(f'it took: {total_time}\n')

        bradley.W_rl_agent.Q_table.to_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
        bradley.B_rl_agent.Q_table.to_pickle(game_settings.imman_agent_q_table_path, compression = 'zip')

        bradley.engine.quit()
    except Exception as e:
        print(f'program interrupted because of:  {e}')
        bradley.engine.quit()

if __name__ == '__main__':
    chess_data_32 = pd.read_pickle(game_settings.chess_games_filepath_part_32, compression = 'zip')
    add_to_q_table_moves_list(chess_data_32)
    chess_data_32 = None

    chess_data_33 = pd.read_pickle(game_settings.chess_games_filepath_part_33, compression = 'zip')
    add_to_q_table_moves_list(chess_data_33)
    chess_data_33 = None



'''
## Overview

This Python script is designed to update Q-tables for a reinforcement learning (RL) agent that is used to play chess. The script processes multiple datasets of chess games, extracts the moves, and updates the Q-tables for both white and black pieces based on these moves. The script utilizes several external modules and helper functions to accomplish this task.

## Dependencies

The script relies on the following modules:
- `pandas`: For data manipulation and reading from pickle files.
- `numpy`: For numerical operations.
- `helper_methods`: A custom module containing helper functions.
- `game_settings`: A custom module containing various game settings and file paths.
- `Bradley`: A custom module defining the RL agent.
- `time`: For measuring the execution time of the script.

## Script Components

### Import Statements
The script begins by importing the necessary modules and custom helper functions:
```python
import pandas as pd
import numpy as np
import helper_methods
import game_settings
import Bradley
import time
```

### `add_to_q_table_moves_list` Function

This function is responsible for updating the Q-tables with moves extracted from the chess game datasets.

#### Parameters:
- `chess_data`: A DataFrame containing chess games data.

#### Steps:
1. **Initialize**:
   - Start the timer to measure the function's execution time.
   - Create an instance of the `Bradley` class.
   - Load pre-existing Q-tables for both white and black agents using helper methods.
   
2. **Print Initial Q-Table Sizes**:
   - Print the sizes of the Q-tables before processing the new games.
   
3. **Extract Moves**:
   - Initialize sets to store all unique moves for white and black pieces.
   - Iterate over each game in `chess_data` and extract moves for both white and black pieces.
   
4. **Update Q-Tables**:
   - Identify new moves that are not already in the Q-tables.
   - Update the Q-tables with these new moves.
   - Print the new moves and the updated sizes of the Q-tables.
   
5. **Save Updated Q-Tables**:
   - Measure the total execution time and print it.
   - Save the updated Q-tables back to the file paths specified in `game_settings`.
   - Quit the chess engine to release resources.
   
6. **Error Handling**:
   - If an exception occurs, print the error message and ensure the chess engine is quit.

### Main Execution Block

The script processes multiple chess game datasets sequentially, updating the Q-tables with each dataset:
```python
if __name__ == '__main__':
    chess_data_files = [
        game_settings.chess_games_filepath_part_32,
        game_settings.chess_games_filepath_part_33,
        game_settings.chess_games_filepath_part_34,
        game_settings.chess_games_filepath_part_35,
        game_settings.chess_games_filepath_part_36,
        game_settings.chess_games_filepath_part_37,
        game_settings.chess_games_filepath_part_38,
        game_settings.chess_games_filepath_part_39,
        game_settings.chess_games_filepath_part_40,
    ]

    for file_path in chess_data_files:
        chess_data = pd.read_pickle(file_path, compression='zip')
        add_to_q_table_moves_list(chess_data)
        chess_data = None
```
This block reads each dataset from its respective pickle file, calls the `add_to_q_table_moves_list` function to process it, and then releases the memory by setting `chess_data` to `None`.

## Purpose

The primary purpose of this script is to update the Q-tables for a chess-playing RL agent. The Q-tables are used by the agent to store the value of different moves, which are updated based on the moves extracted from historical chess game data. By continuously updating the Q-tables with new data, the RL agent can improve its performance and learn from a larger set of chess games.

## Detailed Function Steps

### 1. Initialization
- The `Bradley` class is instantiated, creating an object `bradley`.
- Q-tables for both white (`W`) and black (`B`) pieces are loaded using `helper_methods.bootstrap_agent_fill_q_table`.

### 2. Extract Moves
- Moves are extracted using regular expressions to filter columns starting with 'W' or 'B' followed by digits.
- Unique moves are collected in sets `all_white_moves` and `all_black_moves`.

### 3. Update Q-Tables
- New moves are identified by finding the difference between the sets of all extracted moves and the current Q-table indices.
- The Q-tables are updated with these new moves using the `update_Q_table` method of the RL agent.

### 4. Save Updated Q-Tables
- The updated Q-tables are saved back to their respective file paths in a compressed format.

### 5. Error Handling
- Any exceptions during execution are caught, and the error message is printed. The chess engine is then quit to ensure no resources are left hanging.

## Conclusion

This script is a crucial component for maintaining and improving the performance of a reinforcement learning agent designed to play chess. By systematically processing game data and updating the Q-tables, the agent can continuously learn and adapt to new patterns and strategies in the game of chess.

'''


    
    
