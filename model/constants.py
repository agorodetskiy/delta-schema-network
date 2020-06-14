from environment.schema_games.breakout.constants import \
    BRICK_SIZE, ENV_SIZE, DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_PADDLE_SHAPE


class Constants:
    """
    N: number of entities
    M: number of attributes of each entity
    A: number of available actions
    L: number of schemas
    T: size of look-ahead window
    """

    # --- Agent options ---

    DO_PRELOAD_DUMP_PARAMS = False

    DO_PRELOAD_HANDCRAFTED_ATTRIBUTE_PARAMS = False
    DO_PRELOAD_HANDCRAFTED_REWARD_PARAMS = False

    DO_LEARN_ATTRIBUTE_PARAMS = True
    DO_LEARN_REWARD_PARAMS = True

    # planning options are ('agent', 'hardcoded', 'random')
    PLANNING_TYPE = 'agent'

    LEARNING_PERIOD = 128
    LEARNING_SOLVER = 'cbc'  # 'cbc', 'gurobi'
    USE_EMERGENCY_PLANNING = True

    VISUALIZE_STATE = True
    VISUALIZE_SCHEMAS = False
    VISUALIZE_INNER_STATE = True
    VISUALIZE_BACKTRACKING = True
    LOG_PLANNED_ACTIONS = False

    N_LEARNING_THREADS = 16

    L = 1000
    NEIGHBORHOOD_RADIUS = 2

    if ENV_SIZE == 'SMALL':
        # T = 112 is enough to hit the furthest brick
        # EMERG_PERIOD = 15 is enough to plan with ball being near closest brick
        T = 112  # min 50
        PLANNING_PERIOD = 10  # run planning every *this* steps
        EMERGENCY_PLANNING_PERIOD = 10  # run planning every *this* steps if there are no planned actions
    elif ENV_SIZE == 'DEFAULT':
        T = 130  # min 112
        PLANNING_PERIOD = 10
        EMERGENCY_PLANNING_PERIOD = 30
    else:
        assert False

    # --- Constants ---

    LEARNING_SCHEMA_TOLERANCE = 1e-8
    ADDING_SCHEMA_TOLERANCE = 1e-8

    SCREEN_HEIGHT = DEFAULT_HEIGHT
    SCREEN_WIDTH = DEFAULT_WIDTH
    N = SCREEN_WIDTH * SCREEN_HEIGHT
    M = 5
    N_PREDICTABLE_ATTRIBUTES = M - 1
    ACTION_SPACE_DIM = 3
    REWARD_SPACE_DIM = 2

    FILTER_SIZE = 2 * NEIGHBORHOOD_RADIUS + 1
    NEIGHBORS_NUM = FILTER_SIZE ** 2 - 1
    FRAME_STACK_SIZE = 2
    SCHEMA_VEC_SIZE = FRAME_STACK_SIZE * (M * (NEIGHBORS_NUM + 1)) + ACTION_SPACE_DIM
    TIME_SIZE = FRAME_STACK_SIZE + T
    LEARNING_BATCH_SIZE = FRAME_STACK_SIZE + 1

    # indices of corresponding attributes in entities' vectors
    BALL_IDX = 0
    PADDLE_IDX = 1
    WALL_IDX = 2
    BRICK_IDX = 3
    VOID_IDX = 4
    FAKE_ENTITY_IDX = N

    # action indices
    ACTION_NOP = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2

    ENTITY_NAMES = {
        BALL_IDX: 'BALL',
        PADDLE_IDX: 'PADDLE',
        WALL_IDX: 'WALL',
        BRICK_IDX: 'BRICK',
    }

    REWARD_NAMES = {
        0: 'POSITIVE',
        1: 'NEGATIVE',
    }

    ATTRIBUTE = 'attribute'
    REWARD = 'reward'
    ALLOWED_OBJ_TYPES = {ATTRIBUTE, REWARD}

    DEFAULT_PADDLE_SHAPE = DEFAULT_PADDLE_SHAPE

"""
env changed constants:

BOUNCE_STOCHASTICITY = 0.25
PADDLE_SPEED_DISTRIBUTION[-1] = 0.90
PADDLE_SPEED_DISTRIBUTION[-2] = 0.10
_MAX_SPEED = 2
DEFAULT_PADDLE_SHAPE

DEFAULT_BRICK_SHAPE = np.array([8, 4])
DEFAULT_NUM_BRICKS_ROWS = 6
DEFAULT_NUM_BRICKS_COLS = 11
"""
