import helper_methods
import game_settings
import Bradley

if __name__ == '__main__':
    bradley = Bradley.Bradley()
    helper_methods.bootstrap_agent(bradley, 'W', game_settings.bradley_agent_q_table_path)
    helper_methods.bootstrap_agent(bradley, 'B', game_settings.imman_agent_q_table_path)
    helper_methods.agent_vs_agent(bradley)