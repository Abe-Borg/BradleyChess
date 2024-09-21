    def simply_play_games(self) -> None:

        if game_settings.PRINT_STEP_BY_STEP:
            self.step_by_step_logger.debug(f'hi from simply_play_games\n')
            self.step_by_step_logger.debug(f'White Q table size before games: {w_agent.q_table.shape}\n')
            self.step_by_step_logger.debug(f'Black Q table size before games: {self.B_rl_agent.q_table.shape}\n')
        
        ### FOR EACH GAME IN THE CHESS DB ###
        game_count = 0
        for game_num_str in game_settings.chess_data.index:
            start_time = time.time()
            
            num_chess_moves_curr_training_game: int = game_settings.chess_data.at[game_num_str, 'PlyCount']

            if game_settings.PRINT_STEP_BY_STEP:
                self.step_by_step_logger.debug(f'game_num_str is: {game_num_str}\n')

            try:
                curr_state = self.environ.get_curr_state()
                
                if game_settings.PRINT_STEP_BY_STEP:
                    self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
            except Exception as e:
                self.error_logger.error(f'An error occurred at self.environ.get_curr_state: {e}\n')
                self.error_logger.error(f'curr board is:\n{self.environ.board}\n\n')
                self.error_logger.error(f'at game: {game_num_str}\n')
                break

            ### LOOP PLAYS THROUGH ONE GAME ###
            while curr_state['turn_index'] < (num_chess_moves_curr_training_game):
                ##################### WHITE'S TURN ####################
                w_chess_move = w_agent.choose_action(curr_state, game_num_str)

                if game_settings.PRINT_STEP_BY_STEP:
                    self.step_by_step_logger.debug(f'w_chess_move is: {w_chess_move}\n')

                if not w_chess_move:
                    self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                    self.error_logger.error(f'w_chess_move is empty at state: {curr_state}\n')
                    self.error_logger.error(f'at game: {game_num_str}\n')
                    break # and go to the next game. this game is over.

                ### WHITE AGENT PLAYS THE SELECTED MOVE ###
                try:
                    self.rl_agent_plays_move(w_chess_move, game_num_str)
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'White played move: {w_chess_move}\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
                    self.error_logger.error(f'at game: {game_num_str}\n')
                    break # and go to the next game. this game is over.

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()

                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at get_curr_state: {e}\n')
                    self.error_logger.error(f'curr board is:\n{self.environ.board}\n\n')
                    self.error_logger.error(f'at game: {game_num_str}\n')
                
                if self.environ.board.is_game_over() or curr_state['turn_index'] >= (num_chess_moves_curr_training_game) or not curr_state['legal_moves']:
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'game is over\n')
                        self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
                    break # and go to next game

                ##################### BLACK'S TURN ####################
                b_chess_move = self.B_rl_agent.choose_action(curr_state, game_num_str)
                
                if game_settings.PRINT_STEP_BY_STEP:
                    self.step_by_step_logger.debug(f'Black chess move: {b_chess_move}\n')

                if not b_chess_move:
                    self.error_logger.error(f'An error occurred at w_agent.choose_action\n')
                    self.error_logger.error(f'b_chess_move is empty at state: {curr_state}\n')
                    self.error_logger.error(f'at: {game_num_str}\n')
                    break # game is over, go to next game.

                ##### BLACK AGENT PLAYS SELECTED MOVE #####
                try:
                    self.rl_agent_plays_move(b_chess_move, game_num_str)
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'black agent played their move\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at rl_agent_plays_move: {e}\n')
                    self.error_logger.error(f'at {game_num_str}\n')
                    break 

                # get latest curr_state since self.rl_agent_plays_move updated the chessboard
                try:
                    curr_state = self.environ.get_curr_state()
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'curr_state is: {curr_state}\n')
                except Exception as e:
                    self.error_logger.error(f'An error occurred at environ.get_curr_state: {e}\n')
                    self.error_logger.error(f'at: {game_num_str}\n')
                    break

                if self.environ.board.is_game_over() or not curr_state['legal_moves']:
                    
                    if game_settings.PRINT_STEP_BY_STEP:
                        self.step_by_step_logger.debug(f'game is over\n')
                    break # and go to next game
            ### END OF CURRENT GAME LOOP ###

            if game_settings.PRINT_STEP_BY_STEP:
                self.step_by_step_logger.debug(f'game {game_num_str} is over\n')
                self.step_by_step_logger.debug(f'agent q tables sizes are: \n')
                self.step_by_step_logger.debug(f'White Q table: {w_agent.q_table.shape}\n')
                self.step_by_step_logger.debug(f'Black Q table: {self.B_rl_agent.q_table.shape}\n')

            # this curr game is done, reset environ to prepare for the next game
            self.environ.reset_environ() # reset and go to next game in chess database
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            # Print progress notification every 1000 games
            if game_count % 1000 == 0:
                print(f"Notification: Game {game_count} is done. Time elapsed: {elapsed_time:.2f} seconds.")
            game_count += 1
        ### END OF FOR LOOP THROUGH CHESS DB ###