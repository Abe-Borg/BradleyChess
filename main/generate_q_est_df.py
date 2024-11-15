# generate_estimated_q_values.py

import pandas as pd
import time
import sys
from training import training_functions
from utils import game_settings

if __name__ == '__main__':
    start_time = time.time()
    
    # Check for command-line arguments to specify the data part number
    if len(sys.argv) != 2:
        print("Usage: python generate_estimated_q_values.py <part_number>")
        sys.exit(1)
    
    part_number = sys.argv[1]
    
    # Construct file paths based on the part number
    chess_data_filepath = f"{game_settings.chess_games_filepath_prefix}_part_{part_number}.pkl"
    est_q_values_filepath = f"{game_settings.est_q_vals_filepath_prefix}_part_{part_number}.pkl"
    
    # Load the chess data
    try:
        print(f"Loading chess data from {chess_data_filepath}...")
        chess_data = pd.read_pickle(chess_data_filepath, compression='zip')
        print(f"Successfully loaded {len(chess_data)} games.")
    except FileNotFoundError:
        print(f"Error: Chess data file not found at {chess_data_filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading chess data: {e}")
        sys.exit(1)
    
    # Generate the estimated Q-values DataFrame
    try:
        print("Generating estimated Q-values...")
        estimated_q_values = training_functions.generate_q_est_df(chess_data)
        print("Estimated Q-values generation completed.")
    except Exception as e:
        print(f"Error during estimated Q-values generation: {e}")
        sys.exit(1)
    
    # Save the estimated Q-values DataFrame
    try:
        estimated_q_values.to_pickle(est_q_values_filepath, compression='zip')
        print(f"Estimated Q-values saved to {est_q_values_filepath}")
    except Exception as e:
        print(f"Error saving estimated Q-values: {e}")
        sys.exit(1)
    
    end_time = time.time()
    total_time = end_time - start_time
    print('Process completed successfully.')
    print(f'Total time taken: {total_time:.2f} seconds for {len(chess_data)} games.')
