import time

import numpy as np
from .constants import Constants
from .graph_utils import Schema, Node, Attribute, Action, Reward
from .tensor_handler import TensorHandler
from .planner import Planner
from .visualizer import Visualizer


class SchemaNetwork(Constants):
    @Visualizer.measure_time('Planner')
    def __init__(self):
        """
        :param W: list of M matrices, each of [(MR + A) x L] shape
        :param R: list of 2 matrices, each of [(MN + A) x L] shape, 1st - pos, 2nd - neg
        """
        self._W_pos, self._W_neg = None, None
        self._R = None
        self._R_weights = None

        self._attribute_nodes = None  # tensor ((FRAME_STACK_SIZE + T) x N x M)
        self._action_nodes = None  # tensor ((FRAME_STACK_SIZE + T) x ACTION_SPACE_DIM)
        self._reward_nodes = None  # tensor ((FRAME_STACK_SIZE + T) x REWARD_SPACE_DIM)

        self._gen_attribute_nodes()
        self._gen_action_nodes()
        self._gen_reward_nodes()

        self._tensor_handler = TensorHandler(self._attribute_nodes,
                                             self._action_nodes, self._reward_nodes)
        self._planner = Planner(self._reward_nodes)
        self._visualizer = Visualizer(self._tensor_handler, self._planner, self._attribute_nodes)
        self._iter = None

    def set_weights(self, W_pos, W_neg, R):
        self._process_input(W_pos, W_neg, R)
        self._print_input_stats(W_pos, W_neg, R)

        self._W_pos = W_pos
        self._W_neg = W_neg
        self._R = R
        self._R_weights = np.ones(R[0].shape[1], dtype=np.float)
        self._tensor_handler.set_weights(W_pos, W_neg, R, self._R_weights)

    def _process_input(self, W_pos, W_neg, R):
        for W in (W_pos, W_neg):
            assert len(W) == self.M - 1, 'BAD_W_NUM'

        required_matrix_shape = (self.SCHEMA_VEC_SIZE, self.L)
        for matrix in (W_pos + W_neg + R):
            assert matrix.dtype == bool, 'BAD_MATRIX_DTYPE'
            assert matrix.ndim == 2, 'BAD_MATRIX_NDIM'
            assert (matrix.shape[0] == required_matrix_shape[0]
                    and matrix.shape[1] <= required_matrix_shape[1]), 'BAD_MATRIX_SHAPE'
            assert matrix.size, 'EMPTY_MATRIX'

    @staticmethod
    def _print_input_stats(W_pos, W_neg, R):
        names = ('W_pos', 'W_neg')
        for W, name in zip((W_pos, W_neg), names):
            print(f'Numbers of schemas in {name} are: ', end='')
            for idx, matrix in enumerate(W):
                print('{}'.format(matrix.shape[1]), end='')
                if idx != len(W) - 1:
                    print(' / ', end='')
            print()

        print('Numbers of schemas in R are: ', end='')
        for idx, r in enumerate(R):
            print('{}'.format(r.shape[1]), end='')
            if idx != len(R) - 1:
                print(' / ', end='')
        print()

    def set_curr_iter(self, iter):
        self._iter = iter

    def _gen_attribute_node_matrix(self, t, prev_layer):
        n_rows = self.N
        n_cols = self.M
        matrix = [
            [Attribute(entity_idx, attribute_idx, t, prev_layer) for attribute_idx in range(n_cols)]
            for entity_idx in range(n_rows)
        ]
        return matrix

    def _gen_attribute_nodes(self):
        tensor = []
        prev_layer = None
        for t in range(self.FRAME_STACK_SIZE + self.T):
            tensor.append(self._gen_attribute_node_matrix(t, prev_layer))
            prev_layer = tensor[-1]

        self._attribute_nodes = np.array(tensor)

    def _gen_action_nodes(self):
        action_nodes = [
            [Action(idx, t=t) for idx in range(self.ACTION_SPACE_DIM)]
            for t in range(self.FRAME_STACK_SIZE + self.T)
        ]
        self._action_nodes = np.array(action_nodes)

    def _gen_reward_nodes(self):
        reward_nodes = [
            [Reward(idx, t=t) for idx in range(self.REWARD_SPACE_DIM)]
            for t in range(self.FRAME_STACK_SIZE + self.T)
        ]
        self._reward_nodes = np.array(reward_nodes)

    def plan_actions(self, frame_stack):
        if len(frame_stack) < self.FRAME_STACK_SIZE:
            print('Small ENTITIES_STACK. Abort.')
            return None

        # instantiate schemas, determine nodes feasibility
        self._tensor_handler.forward_pass(frame_stack)

        # visualizing
        self._visualizer.set_iter(self._iter)
        if self.VISUALIZE_SCHEMAS:
            self._visualizer.visualize_schemas(self._W_pos, self._W_neg, self._R)
        if self.VISUALIZE_INNER_STATE:
            self._visualizer.visualize_predicted_entities(check_correctness=False)

        # planning actions
        actions, target_reward_nodes = self._planner.plan_actions()

        if self.VISUALIZE_BACKTRACKING:
            if target_reward_nodes:
                self._visualizer.visualize_backtracking(target_reward_nodes,
                                                        self._planner.node2triplets)

                #self._visualizer.log_balls_at_backtracking(target_reward_nodes[0])

        if self.LOG_PLANNED_ACTIONS:
            self._visualizer.log_planned_actions(actions)

        return actions
