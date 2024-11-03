estimated_q_values_table = generate_q_est_df(chess_data)

estimated_q_values_table.to_pickle('path_to_save_estimated_q_values.pkl', compression='zip')
