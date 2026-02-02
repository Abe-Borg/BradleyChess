# CLAUDE.md - BradleyChess

## Project Overview

BradleyChess is a reinforcement learning chess engine that uses the SARSA (State-Action-Reward-State-Action) algorithm. Two RL agents -- "Bradley" (White) and "Imman" (Black) -- learn to play chess through a two-phase training process:

1. **Phase 1 (Imitation Learning):** Agents replay moves from a chess game database exactly as recorded, learning positional chess patterns.
2. **Phase 2 (Self-Play):** Agents choose their own moves using an epsilon-greedy policy, training each other through mutual play.

## Repository Structure

```
BradleyChess/
├── agents/
│   ├── __init__.py
│   └── Agent.py              # RL agent class (choose_action, Q-table management)
├── environment/
│   ├── __init__.py
│   └── Environ.py            # Chess board wrapper (state, legal moves, turn tracking)
├── training/
│   ├── __init__.py
│   └── training_functions.py # Core training loop, SARSA updates, parallel processing
├── utils/
│   ├── __init__.py
│   ├── constants.py          # Hyperparameters and reward values
│   ├── game_settings.py      # File paths, engine config, display settings
│   └── helper_methods.py     # Utility functions (bootstrapping, game-over checks)
├── main/
│   ├── __init__.py
│   └── train_new_agents.py   # Entry point for training
├── Q_Tables/                 # Serialized Q-tables and estimated Q-values (pickle/gzip)
│   ├── bradley_agent_q_table.pkl
│   ├── imman_agent_q_table.pkl
│   ├── unique_chess_moves_list.pkl
│   └── Estimated_Q_Values/   # Pre-computed Q-value estimates (100 parts)
├── debug/                    # Log files (CRITICAL-level only)
├── training_results/         # Training output text files
├── stockfish_15_win_x64_avx2/ # Stockfish engine binary (Windows)
├── Misc Code Testing.ipynb   # Jupyter notebook for experimentation
├── requirements.txt          # Python dependencies
└── README.md
```

## Language and Dependencies

- **Language:** Python 3
- **Core dependencies:** `chess` (python-chess), `pandas`, `numpy`
- **Testing:** `pytest` (listed in requirements but no tests exist yet)
- **Other:** `colorama`, `tqdm`, Jupyter/IPython stack
- **No build system** -- run scripts directly with Python

## How to Run

### Training
```bash
python -m main.train_new_agents
```
Run from the repository root. The entry point is `main/train_new_agents.py`. Before running, ensure:
- Q-table pickle files exist in `Q_Tables/`
- Chess game data files exist in `chess_data/` (currently deleted from repo)
- Estimated Q-value files exist in `Q_Tables/Estimated_Q_Values/`

The script in `train_new_agents.py` has a hardcoded data part number (e.g., `part_1`) that must be changed manually for each training batch.

### Install Dependencies
```bash
pip install -r requirements.txt
```
Note: `requirements.txt` has a UTF-16 encoding with spaces between characters. It may need to be re-encoded to standard UTF-8 before `pip install` works correctly.

## Architecture and Key Concepts

### Agent (`agents/Agent.py`)
- Holds a Q-table as a `pd.DataFrame` (rows = chess moves in SAN, columns = turn labels)
- **Training mode** (`is_trained=False`): replays database moves via `policy_training_mode()`
- **Game mode** (`is_trained=True`): uses epsilon-greedy via `policy_game_mode()` -- 5% random, 95% best Q-value
- New moves are lazily added to the Q-table when first encountered (`update_q_table()`)

### Environment (`environment/Environ.py`)
- Wraps `python-chess` `Board` class
- Manages turn tracking with labels like `W1`, `B1`, `W2`, `B2`, ... up to `W200`, `B200`
- State = `{turn_index, curr_turn, legal_moves}`
- Only class that mutates the board

### Training (`training/training_functions.py`)
- Uses `multiprocessing.Pool` for parallel training across game batches
- SARSA update: `Q(s,a) += lr * (reward + (discount * est_Q(s',a')) - Q(s,a))`
- Reward function based on move properties (captures, piece development, promotions, checks)
- Q-tables from parallel workers are merged by summing overlapping entries
- Logging at CRITICAL level only, written to `debug/` directory

### Constants (`utils/constants.py`)
Key hyperparameters:
- Learning rate: `0.6`
- Discount factor: `0.35`
- Epsilon (random move chance): `0.05`
- Max turns per player: `200` (400 total plies per game)
- Initial Q-value: `1`

Reward values:
- `piece_development`: 200
- `capture`: 150
- `new_move`: 100
- `check`: 300
- `promotion`: 500
- `promotion_queen`: 900
- `mate_score`: 1000

### Settings (`utils/game_settings.py`)
- All file paths are defined here using `pathlib.Path` relative to the `utils/` directory
- Contains 100 numbered path entries each for estimated Q-values and chess game data files
- Stockfish path points to a Windows executable

## Data Format

- **Chess games:** Pandas DataFrames stored as gzip-compressed pickle files. Rows indexed by game name (e.g., "Game 1"), columns are turn labels (`W1`, `B1`, `W2`, `B2`, ...) plus metadata like `PlyCount`.
- **Q-tables:** Pandas DataFrames stored as gzip-compressed pickle files. Rows indexed by chess move in SAN notation (e.g., "e4", "Nxf6"), columns are turn labels.
- **Estimated Q-values:** Same DataFrame format, pre-computed values used during training.

## Code Conventions

### Naming
- **Classes:** PascalCase (`Agent`, `Environ`)
- **Functions/methods:** snake_case (`get_curr_state`, `update_q_table`)
- **Constants:** UPPER_CASE for dicts (`CHESS_MOVE_VALUES`), lower_snake_case for scalars (`default_learning_rate`)
- **Agents:** "Bradley" = White player, "Imman" = Black player
- **Turns:** `W1`, `W2`, ... for White; `B1`, `B2`, ... for Black

### Type Hints
The codebase uses Python type hints throughout:
- `Union`, `Dict`, `List`, `Optional` from `typing`
- Return type annotations on all public methods
- Parameter type annotations on function signatures

### File Headers
Each Python file starts with:
```python
# project name: BradleyChess
# <filename>.py
```

### Style
- No docstrings -- code is meant to be self-documenting
- Minimal inline comments
- Each module has a single responsibility
- `__init__.py` files are empty (no explicit exports)

### Imports
- Standard library imports first, then third-party, then local modules
- Local imports use package-relative style: `from utils import constants`, `from agents import Agent`

## Important Patterns

1. **State dictionary:** The environment state is always a dict with keys `turn_index`, `curr_turn`, and `legal_moves`. This is the primary interface between the environment and agents.

2. **Q-table as DataFrame:** Q-tables use pandas DataFrames rather than nested dicts. Moves are row indices, turns are column headers. This allows vectorized operations and easy serialization.

3. **Parallel training with merge:** Training splits games across CPU cores. Each worker gets a copy of the Q-tables, trains independently, then results are merged by summing Q-values for matching (move, turn) entries.

4. **Chess moves in SAN:** All moves are represented in Standard Algebraic Notation (e.g., "e4", "Nf3", "Bxc6+"). The `python-chess` library handles conversion between internal move representation and SAN.

5. **Lazy Q-table expansion:** The Q-table grows dynamically. When a new move is encountered that isn't in the table, a new row is added with zeros for all columns.

## Logging

- Loggers are configured at CRITICAL level only
- Log files are written to the `debug/` directory
- Each module has its own logger and log file
- Format: `%(asctime)s - %(levelname)s - %(message)s`

## Known Issues and Notes

- The `requirements.txt` file has UTF-16 encoding with spaces between characters, making it unusable with standard `pip install -r`
- The README references a "Bradley Class" as a composite coordinator class, but this class does not exist in the current codebase (it may have been removed or refactored)
- `train_new_agents.py` saves the original `white_q_table`/`black_q_table` DataFrames instead of the trained agent Q-tables (`Bradley.q_table`/`Imman.q_table`) -- this appears to be a bug
- Stockfish path points to a Windows `.exe` binary which won't work on Linux/macOS
- The `chess_data/` directory referenced by `game_settings.py` has been deleted from the repository
- No automated tests exist despite `pytest` being in the dependencies
