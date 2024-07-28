class Agent:
    def __init__(self, color: str, learn_rate = 0.6, discount_factor = 0.35, q_table: pd.DataFrame = None):        
        self.learn_rate = learn_rate
        self.discount_factor = discount_factor
        self.color = color
        self.is_trained: bool = False
        self.q_table: pd.DataFrame = q_table # q table will be assigned at program execution.
    ### end of __init__ ###

    def choose_action(self, chess_data, environ_state: dict[str, str, list[str]], curr_game: str = 'Game 1') -> str:
        if environ_state['legal_moves'] == []:
            return ''
        
        self.update_q_table(environ_state['legal_moves']) # this func also checks if there are any new unique move strings

        if self.is_trained:
            return self.policy_game_mode(environ_state['legal_moves'], environ_state['curr_turn'])
        else:
            return self.policy_training_mode(chess_data, curr_game, environ_state["curr_turn"])
    ### end of choose_action ###
    
    def policy_training_mode(self, chess_data, curr_game: str, curr_turn: str) -> str:
        try:
            chess_move = chess_data.at[curr_game, curr_turn]
            return chess_move
        except Exception as e:
            raise Exception from e
    ### end of policy_training_mode ###

    def policy_game_mode(self, legal_moves: list[str], curr_turn: str) -> str:
        dice_roll = helper_methods.get_number_with_probability(game_settings.chance_for_random_move)
        
        try:
            legal_moves_in_q_table = self.q_table[curr_turn].loc[self.q_table[curr_turn].index.intersection(legal_moves)]
        except Exception as e:
            raise Exception from e

        if dice_roll == 1:
            chess_move = legal_moves_in_q_table.sample().index[0]
        else:
            chess_move = legal_moves_in_q_table.idxmax()
        return chess_move
    ### end of policy_game_mode ###

    def change_q_table_pts(self, chess_move: str, curr_turn: str, pts: int) -> None:
        try:    
            self.q_table.at[chess_move, curr_turn] += pts
        except Exception as e:
            raise Exception from e
    ### end of change_q_table_pts ###

    def update_q_table(self, new_chess_moves: Union[str, list[str]]) -> None:
        if isinstance(new_chess_moves, str):
            new_chess_moves = [new_chess_moves]
        
        # Convert to set for efficient lookup
        new_moves_set = set(new_chess_moves)

        # Check if all moves are already in the Q-table
        existing_moves = set(self.q_table.index)
        truly_new_moves = new_moves_set - existing_moves

        # If no new moves, return early
        if not truly_new_moves:
            return

        q_table_new_values: pd.DataFrame = pd.DataFrame(
            0, 
            index = list(truly_new_moves), 
            columns = self.q_table.columns, 
            dtype = np.int64
        )

        self.q_table = pd.concat([self.q_table, q_table_new_values])
    ### update_q_table ###
