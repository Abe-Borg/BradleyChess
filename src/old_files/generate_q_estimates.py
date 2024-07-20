import helper_methods
import game_settings
import time
import Bradley

# !!! MAKE SURE to set desired chess_data path in game settings before executing this script !!! #

if __name__ == '__main__':
    start_time = time.time()
    bradley = Bradley.Bradley()

    try:
        bradley.generate_q_est_vals(game_settings.est_q_vals_file_path)
        bradley.engine.quit()
    except Exception as e:
        print(f'generate q est interrupted because of:  {e}')
        bradley.engine.quit()
        quit()
    
    end_time = time.time()
    total_time = end_time - start_time
    print('generate q est is complete')
    print(f'it took: {total_time} seconds\n')