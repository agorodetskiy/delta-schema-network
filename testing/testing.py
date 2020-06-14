import numpy as np
from model.constants import Constants
from model.constants import Constants as C
from collections import namedtuple


class HardcodedSchemaVectors(Constants):
    # TODO: find an elegant way to make these names visible
    BALL_IDX = Constants.BALL_IDX
    PADDLE_IDX = Constants.PADDLE_IDX
    WALL_IDX = Constants.WALL_IDX
    BRICK_IDX = Constants.BRICK_IDX
    VOID_IDX = Constants.VOID_IDX
    ACTION_NOP = Constants.ACTION_NOP
    ACTION_MOVE_LEFT = Constants.ACTION_MOVE_LEFT
    ACTION_MOVE_RIGHT = Constants.ACTION_MOVE_RIGHT

    AttributePrecondition = namedtuple('AttributePrecondition', ['time_step', 'di', 'dj', 'entity_type_idx'])
    ActionPrecondition = namedtuple('ActionPrecondition', ['action_idx'])
    wall = [
        (AttributePrecondition('prev', 0, 0, WALL_IDX),
         AttributePrecondition('curr', 0, 0, WALL_IDX)),
    ]
    brick = [
        (AttributePrecondition('prev', 0, 0, BRICK_IDX),
         AttributePrecondition('curr', 0, 0, BRICK_IDX)),
    ]
    paddle = [
        (AttributePrecondition('prev', 0, 0, PADDLE_IDX),
         AttributePrecondition('curr', 0, 0, PADDLE_IDX),
         ActionPrecondition(ACTION_NOP)),
        # paddle growing
        (AttributePrecondition('curr', 0, -2, PADDLE_IDX),
         ActionPrecondition(ACTION_MOVE_RIGHT)),  # to right
        (AttributePrecondition('curr', 0, 2, PADDLE_IDX),
         ActionPrecondition(ACTION_MOVE_LEFT)),  # to left
    ]
    ball = [
        # linear movement
        (AttributePrecondition('prev', -2, -2, BALL_IDX),
         AttributePrecondition('curr', -1, -1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', -2, 2, BALL_IDX),
         AttributePrecondition('curr', -1, 1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', 2, 2, BALL_IDX),
         AttributePrecondition('curr', 1, 1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', 2, -2, BALL_IDX),
         AttributePrecondition('curr', 1, -1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', -2, 0, BALL_IDX),
         AttributePrecondition('curr', -1, 0, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', 2, 0, BALL_IDX),
         AttributePrecondition('curr', 1, 0, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        # bounce from paddle
        (AttributePrecondition('prev', 0, -2, BALL_IDX),  # left to right
         AttributePrecondition('curr', 1, -1, BALL_IDX),
         AttributePrecondition('curr', 2, -1, PADDLE_IDX),
         AttributePrecondition('curr', 2, 0, PADDLE_IDX)),
        (AttributePrecondition('prev', 0, 2, BALL_IDX),  # right to left
         AttributePrecondition('curr', 1, 1, BALL_IDX),
         AttributePrecondition('curr', 2, 1, PADDLE_IDX),
         AttributePrecondition('curr', 2, 0, PADDLE_IDX)),
        (AttributePrecondition('prev', 0, 0, BALL_IDX),  # upright
         AttributePrecondition('curr', 1, 0, BALL_IDX),
         AttributePrecondition('curr', 2, 0, PADDLE_IDX)),
        # bounce from wall
        (AttributePrecondition('prev', 2, 0, BALL_IDX),  # left wall, bounce bottom-up
         AttributePrecondition('curr', 1, -1, BALL_IDX),
         AttributePrecondition('prev', 1, -2, WALL_IDX),
         AttributePrecondition('curr', 1, -2, WALL_IDX)),
        (AttributePrecondition('prev', -2, 0, BALL_IDX),  # left wall, bounce top-down
         AttributePrecondition('curr', -1, -1, BALL_IDX),
         AttributePrecondition('prev', -1, -2, WALL_IDX),
         AttributePrecondition('curr', -1, -2, WALL_IDX)),
        (AttributePrecondition('prev', 2, 0, BALL_IDX),  # right wall, bounce bottom-up
         AttributePrecondition('curr', 1, 1, BALL_IDX),
         AttributePrecondition('prev', 1, 2, WALL_IDX),
         AttributePrecondition('curr', 1, 2, WALL_IDX)),
        (AttributePrecondition('prev', -2, 0, BALL_IDX),  # right wall, bounce top-down
         AttributePrecondition('curr', -1, 1, BALL_IDX),
         AttributePrecondition('prev', -1, 2, WALL_IDX),
         AttributePrecondition('curr', -1, 2, WALL_IDX)),
    ]

    entity_types = (ball, paddle, wall, brick)

    positive_reward = [
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # attack from left
         AttributePrecondition('prev', 1, -1, BALL_IDX),
         AttributePrecondition('curr', -1, 1, BRICK_IDX),),
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # attack from right
         AttributePrecondition('prev', 1, 1, BALL_IDX),
         AttributePrecondition('curr', -1, -1, BRICK_IDX),),
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # upright attack
         AttributePrecondition('prev', 1, 0, BALL_IDX),
         AttributePrecondition('curr', -1, 0, BRICK_IDX),),
    ]
    """
    (AttributePrecondition('curr', 0, 0, BALL_IDX),  # bounce from paddle
     AttributePrecondition('curr', 1, -1, PADDLE_IDX),
     AttributePrecondition('curr', 1, 0, PADDLE_IDX),
     AttributePrecondition('curr', 1, 1, PADDLE_IDX),),
    """
    negative_reward = [
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # all-in-center (just to fill 1 schema)
         AttributePrecondition('curr', 0, 0, PADDLE_IDX),
         AttributePrecondition('curr', 0, 0, WALL_IDX),
         AttributePrecondition('curr', 0, 0, BRICK_IDX),)
    ]
    rewards = [positive_reward, negative_reward]

    @classmethod
    def convert_filter_offset_to_schema_vec_idx(cls, time_step, di, dj, entity_type_idx):
        assert time_step in ('prev', 'curr')

        i = cls.NEIGHBORHOOD_RADIUS + di
        j = cls.NEIGHBORHOOD_RADIUS + dj
        filter_idx = i * cls.FILTER_SIZE + j

        mid = (cls.NEIGHBORS_NUM + 1) // 2
        if filter_idx == mid:
            vec_entity_idx = 0
        elif filter_idx < mid:
            vec_entity_idx = filter_idx + 1
        else:
            vec_entity_idx = filter_idx

        if time_step == 'curr':
            vec_entity_idx += cls.NEIGHBORS_NUM + 1

        vec_idx = vec_entity_idx * cls.M + entity_type_idx
        return vec_idx

    @classmethod
    def convert_action_idx_to_schema_vec_idx(cls, action_vec_idx):
        pass

    @classmethod
    def make_schema_vec(cls, preconditions):
        vec = np.full(cls.SCHEMA_VEC_SIZE, False, dtype=bool)
        for precondition in preconditions:
            if type(precondition) is cls.AttributePrecondition:
                idx = cls.convert_filter_offset_to_schema_vec_idx(precondition.time_step,
                                                                  precondition.di,
                                                                  precondition.dj,
                                                                  precondition.entity_type_idx)
            elif type(precondition) is cls.ActionPrecondition:
                offset = cls.ACTION_SPACE_DIM - precondition.action_idx
                idx = -offset
            else:
                raise AssertionError

            vec[idx] = True
        return vec

    @classmethod
    def make_target_schema_matrices(cls, prediction_targets):
        A = []
        for target in prediction_targets:
            A_i = []
            for schema_preconditions in target:
                vec = cls.make_schema_vec(schema_preconditions)
                A_i.append(vec)
            A.append(A_i)
        A = [np.stack(A_i, axis=0).T for A_i in A]
        return A

    @classmethod
    def gen_schema_matrices(cls):
        W = cls.make_target_schema_matrices(cls.entity_types)
        R = cls.make_target_schema_matrices(cls.rewards)
        return W, R


class HardcodedDeltaSchemaVectors():
    # TODO: find an elegant way to make these names visible
    BALL_IDX = Constants.BALL_IDX
    PADDLE_IDX = Constants.PADDLE_IDX
    WALL_IDX = Constants.WALL_IDX
    BRICK_IDX = Constants.BRICK_IDX
    VOID_IDX = Constants.VOID_IDX
    ACTION_NOP = Constants.ACTION_NOP
    ACTION_MOVE_LEFT = Constants.ACTION_MOVE_LEFT
    ACTION_MOVE_RIGHT = Constants.ACTION_MOVE_RIGHT

    AttributePrecondition = namedtuple('AttributePrecondition', ['time_step', 'di', 'dj', 'entity_type_idx'])
    ActionPrecondition = namedtuple('ActionPrecondition', ['action_idx'])
    wall = []
    brick = []
    paddle = [
        # (AttributePrecondition('curr', 0, 0, PADDLE_IDX),
        # ActionPrecondition(ACTION_NOP)),
        # paddle growing
        (AttributePrecondition('curr', 0, -2, PADDLE_IDX),
         ActionPrecondition(ACTION_MOVE_RIGHT)),  # to right
        (AttributePrecondition('curr', 0, 2, PADDLE_IDX),
         ActionPrecondition(ACTION_MOVE_LEFT)),  # to left
    ]
    ball = [
        # linear movement
        (AttributePrecondition('prev', -2, -2, BALL_IDX),
         AttributePrecondition('curr', -1, -1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', -2, 2, BALL_IDX),
         AttributePrecondition('curr', -1, 1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', 2, 2, BALL_IDX),
         AttributePrecondition('curr', 1, 1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', 2, -2, BALL_IDX),
         AttributePrecondition('curr', 1, -1, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', -2, 0, BALL_IDX),
         AttributePrecondition('curr', -1, 0, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        (AttributePrecondition('prev', 2, 0, BALL_IDX),
         AttributePrecondition('curr', 1, 0, BALL_IDX),
         AttributePrecondition('curr', 0, 0, VOID_IDX)),
        # bounce from paddle
        (AttributePrecondition('prev', 0, -2, BALL_IDX),  # left to right
         AttributePrecondition('curr', 1, -1, BALL_IDX),
         AttributePrecondition('curr', 2, -1, PADDLE_IDX),
         AttributePrecondition('curr', 2, 0, PADDLE_IDX)),
        (AttributePrecondition('prev', 0, 2, BALL_IDX),  # right to left
         AttributePrecondition('curr', 1, 1, BALL_IDX),
         AttributePrecondition('curr', 2, 1, PADDLE_IDX),
         AttributePrecondition('curr', 2, 0, PADDLE_IDX)),
        (AttributePrecondition('prev', 0, 0, BALL_IDX),  # upright
         AttributePrecondition('curr', 1, 0, BALL_IDX),
         AttributePrecondition('curr', 2, 0, PADDLE_IDX)),
        # bounce from wall
        (AttributePrecondition('prev', 2, 0, BALL_IDX),  # left wall, bounce bottom-up
         AttributePrecondition('curr', 1, -1, BALL_IDX),
         AttributePrecondition('prev', 1, -2, WALL_IDX),
         AttributePrecondition('curr', 1, -2, WALL_IDX)),
        (AttributePrecondition('prev', -2, 0, BALL_IDX),  # left wall, bounce top-down
         AttributePrecondition('curr', -1, -1, BALL_IDX),
         AttributePrecondition('prev', -1, -2, WALL_IDX),
         AttributePrecondition('curr', -1, -2, WALL_IDX)),
        (AttributePrecondition('prev', 2, 0, BALL_IDX),  # right wall, bounce bottom-up
         AttributePrecondition('curr', 1, 1, BALL_IDX),
         AttributePrecondition('prev', 1, 2, WALL_IDX),
         AttributePrecondition('curr', 1, 2, WALL_IDX)),
        (AttributePrecondition('prev', -2, 0, BALL_IDX),  # right wall, bounce top-down
         AttributePrecondition('curr', -1, 1, BALL_IDX),
         AttributePrecondition('prev', -1, 2, WALL_IDX),
         AttributePrecondition('curr', -1, 2, WALL_IDX)),
    ]

    entity_types = (ball, paddle, wall, brick)

    # negative schemas
    wall = []
    brick = []
    paddle = [
        (AttributePrecondition('curr', 0, 0, PADDLE_IDX),
         ActionPrecondition(ACTION_MOVE_RIGHT)),  # to right
        (AttributePrecondition('curr', 0, 0, PADDLE_IDX),
         ActionPrecondition(ACTION_MOVE_LEFT)),  # to left
    ]
    ball = [
        (AttributePrecondition('curr', 0, 0, BALL_IDX),)
    ]

    neg_entity_types = (ball, paddle, wall, brick)

    positive_reward = [
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # attack from left
         AttributePrecondition('prev', 1, -1, BALL_IDX),
         AttributePrecondition('curr', -1, 1, BRICK_IDX),),
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # attack from right
         AttributePrecondition('prev', 1, 1, BALL_IDX),
         AttributePrecondition('curr', -1, -1, BRICK_IDX),),
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # upright attack
         AttributePrecondition('prev', 1, 0, BALL_IDX),
         AttributePrecondition('curr', -1, 0, BRICK_IDX),),
    ]
    """
    (AttributePrecondition('curr', 0, 0, BALL_IDX),  # bounce from paddle
     AttributePrecondition('curr', 1, -1, PADDLE_IDX),
     AttributePrecondition('curr', 1, 0, PADDLE_IDX),
     AttributePrecondition('curr', 1, 1, PADDLE_IDX),),
    """
    negative_reward = [
        (AttributePrecondition('curr', 0, 0, BALL_IDX),  # all-in-center (just to fill 1 schema)
         AttributePrecondition('curr', 0, 0, PADDLE_IDX),
         AttributePrecondition('curr', 0, 0, WALL_IDX),
         AttributePrecondition('curr', 0, 0, BRICK_IDX),)
    ]
    rewards = [positive_reward, negative_reward]

    @classmethod
    def convert_filter_offset_to_schema_vec_idx(cls, time_step, di, dj, entity_type_idx):
        assert time_step in ('prev', 'curr')

        i = C.NEIGHBORHOOD_RADIUS + di
        j = C.NEIGHBORHOOD_RADIUS + dj
        filter_idx = i * C.FILTER_SIZE + j

        mid = (C.NEIGHBORS_NUM + 1) // 2
        if filter_idx == mid:
            vec_entity_idx = 0
        elif filter_idx < mid:
            vec_entity_idx = filter_idx + 1
        else:
            vec_entity_idx = filter_idx

        if time_step == 'curr':
            vec_entity_idx += C.NEIGHBORS_NUM + 1

        vec_idx = vec_entity_idx * C.M + entity_type_idx
        return vec_idx

    @classmethod
    def convert_action_idx_to_schema_vec_idx(cls, action_vec_idx):
        pass

    @classmethod
    def make_schema_vec(cls, preconditions):
        vec = np.full(C.SCHEMA_VEC_SIZE, False, dtype=bool)
        for precondition in preconditions:
            if type(precondition) is cls.AttributePrecondition:
                idx = cls.convert_filter_offset_to_schema_vec_idx(precondition.time_step,
                                                                  precondition.di,
                                                                  precondition.dj,
                                                                  precondition.entity_type_idx)
            elif type(precondition) is cls.ActionPrecondition:
                offset = C.ACTION_SPACE_DIM - precondition.action_idx
                idx = -offset
            else:
                raise AssertionError

            vec[idx] = True
        return vec

    @classmethod
    def make_target_schema_matrices(cls, prediction_targets):
        A = []
        for target in prediction_targets:
            A_i = []
            for schema_preconditions in target:
                vec = cls.make_schema_vec(schema_preconditions)
                A_i.append(vec)
            A.append(A_i)
        A = [np.stack(A_i, axis=0).T if len(A_i) else np.ones((C.SCHEMA_VEC_SIZE, 0), dtype=bool) for A_i in A]
        return A

    @classmethod
    def gen_schema_matrices(cls):
        W_pos = cls.make_target_schema_matrices(cls.entity_types)
        W_neg = cls.make_target_schema_matrices(cls.neg_entity_types)
        R = cls.make_target_schema_matrices(cls.rewards)
        return W_pos, W_neg, R
