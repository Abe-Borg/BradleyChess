import helper_methods
import game_settings
import Bradley
import time


# !!! MAKE SURE to set desired chess_data path in game settings before executing this script !!! #

 
if __name__ == '__main__':
    start_time = time.time()
    bradley = Bradley.Bradley()

    # initialize agents with q tables
    helper_methods.bootstrap_agent_fill_q_table(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent_fill_q_table(bradley, 'B', game_settings.imman_agent_q_table_path)

    print(f'White Q table size before games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size before games: {bradley.B_rl_agent.Q_table.shape}\n')

    try:
        bradley.simply_play_games()
        bradley.engine.quit()
    except Exception as e:
        print(f'simply play games, interrupted because of:  {e}')
        bradley.engine.quit()    
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time

    print(f'\nq table populated with new moves from current section of chess db')
    print(f'it took: {total_time}\n')

    print(f'White Q table size after games: {bradley.W_rl_agent.Q_table.shape}')
    print(f'Black Q table size after games: {bradley.B_rl_agent.Q_table.shape}')

    bradley.W_rl_agent.Q_table.to_pickle(game_settings.bradley_agent_q_table_path, compression = 'zip')
    bradley.B_rl_agent.Q_table.to_pickle(game_settings.imman_agent_q_table_path, compression = 'zip')

    quit()



    '''
    ## Overview

This Python script is designed to update Q-tables for a reinforcement learning (RL) agent used to play chess by simulating games. The script initializes the RL agent, populates the Q-tables, simulates games, and then updates the Q-tables based on the outcomes of these games. The script uses several external modules and helper functions to achieve this.

## Dependencies

The script relies on the following modules:
- `helper_methods`: A custom module containing helper functions.
- `game_settings`: A custom module containing various game settings and file paths.
- `Bradley`: A custom module defining the RL agent.
- `time`: For measuring the execution time of the script.

## Script Components

### Import Statements
The script begins by importing the necessary modules and custom helper functions:
```python
import helper_methods
import game_settings
import Bradley
import time
```

### Main Execution Block

This block is the main part of the script and is executed when the script runs. It performs the following steps:

#### Steps:
1. **Initialization**:
   - Start the timer to measure the script's execution time.
   - Create an instance of the `Bradley` class.

2. **Bootstrap Agents with Q-Tables**:
   - Load pre-existing Q-tables for both white (`W`) and black (`B`) agents using `helper_methods.bootstrap_agent_fill_q_table`.
   
3. **Print Initial Q-Table Sizes**:
   - Print the sizes of the Q-tables before simulating the new games.

4. **Simulate Games**:
   - Call the `simply_play_games` method on the `bradley` object to simulate games and update the Q-tables accordingly.

5. **Error Handling**:
   - If an exception occurs, print the error message and ensure the chess engine is quit.

6. **Measure and Print Execution Time**:
   - Measure the total execution time and print it.

7. **Print Updated Q-Table Sizes**:
   - Print the sizes of the Q-tables after the games have been simulated.

8. **Save Updated Q-Tables**:
   - Save the updated Q-tables back to the file paths specified in `game_settings`.

9. **Exit**:
   - Quit the script.

### Detailed Function Steps

#### 1. Initialization
- The `Bradley` class is instantiated, creating an object `bradley`.

#### 2. Bootstrap Agents with Q-Tables
- Q-tables for both white (`W`) and black (`B`) pieces are loaded using `helper_methods.bootstrap_agent_fill_q_table`.

#### 3. Print Initial Q-Table Sizes
- The sizes of the Q-tables are printed before simulating the games.

#### 4. Simulate Games
- The `simply_play_games` method of the `bradley` object is called to simulate games and update the Q-tables.

#### 5. Error Handling
- Any exceptions during execution are caught, and the error message is printed. The chess engine is then quit to ensure no resources are left hanging.

#### 6. Measure and Print Execution Time
- The total execution time is measured and printed.

#### 7. Print Updated Q-Table Sizes
- The sizes of the Q-tables are printed after the games have been simulated.

#### 8. Save Updated Q-Tables
- The updated Q-tables are saved back to their respective file paths in a compressed format.

#### 9. Exit
- The script is quit using `quit()`.

## Conclusion

This script is essential for maintaining and improving the performance of a reinforcement learning agent designed to play chess. By simulating games and updating the Q-tables, the agent can continuously learn and adapt to new strategies and patterns in the game of chess.

    '''