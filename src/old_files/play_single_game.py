import helper_methods
import game_settings
import Bradley
 
# !!! MAKE SURE to set desired chess_data path in game settings before executing this script !!! #

if __name__ == '__main__':
    bradley = Bradley.Bradley()
    
    try:
        bradley.simply_play_games()
        bradley.engine.quit()
    except Exception as e:
        print(f'simply play games, interrupted because of:  {e}')
        bradley.engine.quit()    
        quit()
    
    print('play through 1 game is complete')
