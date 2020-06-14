### Delta Schema Network in model-based reinforcement learning

Model for learning environment dynamics as logic relations, predicting future states and planning actions to reach reward.

Code for paper (published soon).

### How to try it

`pip install -r requirements.txt`

Tweak options in `model/constants.py`, such as:

- `DO_PRELOAD_HANDCRAFTED_*` - use handcrafted vectors instead of learned
- `VISUALIZE_*` - visualize stuff
- `LEARNING_SOLVER` - one can use Gurobi to accelerate training, default is CBC

Run `python3 run_agent.py`
