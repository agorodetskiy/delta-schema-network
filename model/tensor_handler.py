import numpy as np
from .constants import Constants
from .shaper import Shaper
from .visualizer import Visualizer


class TensorHandler(Constants):
    def __init__(self, attribute_nodes, action_nodes, reward_nodes):
        self._W_pos, self._W_neg = None, None
        self._R = None
        self._R_weights = None

        self._entities_stack = None

        # ((FRAME_STACK_SIZE + T) x self.N x self.M)
        self._attribute_tensor = None
        self._reward_tensor = None

        # from SchemaNetwork
        self._attribute_nodes = attribute_nodes
        self._action_nodes = action_nodes
        self._reward_nodes = reward_nodes

        # helping tensor for instantiating schemas
        self._reference_attribute_nodes = None  # tensor ((T+1) x N x (MR + A))

        # shaping matrices and node tensors
        self._shaper = Shaper()

        # create tensors
        self._gen_attribute_tensor()
        self._gen_reward_tensor()
        self._gen_reference_attribute_nodes()

    def set_weights(self, W_pos, W_neg, R, R_weights):
        self._W_pos, self._W_neg = W_pos, W_neg
        self._R = R
        self._R_weights = R_weights

    def _get_env_attribute_tensor(self):
        """
        Get observed state
        :returns (FRAME_STACK_SIZE x N x M) tensor
        """
        assert self._entities_stack is not None, 'NO_ENTITIES_STACK'
        assert len(self._entities_stack) == self.FRAME_STACK_SIZE, 'BAD_ENTITIES_STACK'

        matrix_shape = (self.N, self.M)
        attribute_tensor = np.empty((self.FRAME_STACK_SIZE,) + matrix_shape, dtype=bool)
        for i in range(self.FRAME_STACK_SIZE):
            matrix = self._entities_stack[i]
            assert matrix.shape == matrix_shape, 'BAD_MATRIX_SHAPE'
            attribute_tensor[i, :, :] = matrix

        return attribute_tensor

    def _gen_attribute_tensor(self):
        shape = (self.FRAME_STACK_SIZE + self.T, self.N, self.M)
        self._attribute_tensor = np.empty(shape, dtype=bool)

    def _init_attribute_tensor(self, src_tensor):
        """
        :param tensor: (FRAME_STACK_SIZE x N x M) ndarray of attributes
        """
        self._attribute_tensor[:self.FRAME_STACK_SIZE, :, :] = src_tensor
        self._attribute_tensor[self.FRAME_STACK_SIZE:, :, :] = False

    def _gen_reward_tensor(self):
        shape = (self.FRAME_STACK_SIZE + self.T, self.REWARD_SPACE_DIM)
        self._reward_tensor = np.empty(shape, dtype=bool)

    def _init_reward_tensor(self):
        self._reward_tensor[:, :] = False

    @Visualizer.measure_time('init_nodes()')
    def _init_nodes(self):
        for node in self._attribute_nodes.flat:
            reachability = (node.t < self.FRAME_STACK_SIZE and
                            self._attribute_tensor[node.t, node.entity_idx, node.attribute_idx])
            node.reset(is_initially_reachable=reachability)

        for node in self._reward_nodes.flat:
            node.reset()

    def _gen_reference_attribute_nodes(self):
        # ((FRAME_STACK_SIZE + T) x N x (MR*ss + A))
        shape = (self.FRAME_STACK_SIZE + self.T,
                 self.N,
                 self.SCHEMA_VEC_SIZE)
        self._reference_attribute_nodes = np.full(
            shape, None, dtype=object
        )
        offset = self.FRAME_STACK_SIZE - 1
        for t in range(offset, offset + self.T + 1):
            src_slice = self._get_tensor_slice(t, 'nodes')
            self._reference_attribute_nodes[t, :, :] = self._shaper.transform_node_matrix(
                src_slice, self._action_nodes, t
            )

    def _get_tensor_slice(self, t, tensor_type):
        """
        t: time at which last layer is located
        size of slice is FRAME_STACK_SIZE in total
        """
        assert tensor_type in ('attributes', 'nodes')
        begin = t - self.FRAME_STACK_SIZE + 1

        # prevent possible shape mismatch downwards the stack
        assert begin >= 0, 'TENSOR_SLICE_BAD_ARGS'

        end = t + 1
        index = np.index_exp[max(0, begin): end]
        if tensor_type == 'attributes':
            slice_ = self._attribute_tensor[index]
        else:
            slice_ = self._attribute_nodes[index]
        return slice_

    def _get_reference_matrix(self, t):
        """
        :param t: rightmost FS's layer to which references are established,
                  time step where we got matrix
        :return: (N x (MR + A)) matrix of references to nodes
        """
        reference_matrix = self._reference_attribute_nodes[t, :, :]
        return reference_matrix

    def _instantiate_attribute_grounded_schemas(self, attribute_idx, t, reference_matrix, W,
                                                pos_delta, neg_delta):
        """
        :param reference_matrix: (N x (MR + A))
        :param t: schema output time_step
        """
        for entity_idx in range(self.N):
            pos_activity_mask = pos_delta[entity_idx, :]
            active_schemas = W[:, pos_activity_mask].T

            for schema_vec in active_schemas:
                preconditions = reference_matrix[entity_idx, schema_vec]
                self._attribute_nodes[t, entity_idx, attribute_idx].add_schema(preconditions, schema_vec)

        # turn of transitions for nodes, which were predicted by neg_delta
        for node in self._attribute_nodes[t, neg_delta.any(axis=1), attribute_idx]:
            node.transition = None

    def _instantiate_reward_grounded_schemas(self, reward_idx, t, reference_matrix, R, predicted_matrix):
        """
        THIS MAY INSTANTIATE DUPLICATE SCHEMAS!!!
        :param reference_matrix: (1 x (MN + A))
        :param t: schema output time
        """
        n_pos_schemas_instantiated = 0
        for row_idx in range(self.N):
            activity_mask = predicted_matrix[row_idx, :]
            precondition_masks = R[:, activity_mask].T

            # save weights in graph for positive reward nodes
            if reward_idx == 0:
                masks_weights = self._R_weights[activity_mask]
                for mask, weight in zip(precondition_masks, masks_weights):
                    preconditions = reference_matrix[row_idx, mask]
                    self._reward_nodes[t, reward_idx].add_schema(preconditions, mask)
                    self._reward_nodes[t, reward_idx].set_weight(weight)
                    n_pos_schemas_instantiated += 1
            else:
                for mask in precondition_masks:
                    preconditions = reference_matrix[row_idx, mask]
                    self._reward_nodes[t, reward_idx].add_schema(preconditions, mask)
        return n_pos_schemas_instantiated

    def _predict_next_attribute_layer(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        src_slice = self._get_tensor_slice(t, 'attributes')  # (FRAME_STACK_SIZE x N x M)
        transformed_matrix = self._shaper.transform_matrix(src_slice)
        reference_matrix = self._get_reference_matrix(t)
        curr_state = src_slice[-1]

        for attr_idx in range(self.N_PREDICTABLE_ATTRIBUTES):
            pos_delta = ~(~transformed_matrix @ self._W_pos[attr_idx])

            transformed_matrix[:, -self.ACTION_SPACE_DIM+1:] = 0
            neg_delta = ~(~transformed_matrix @ self._W_neg[attr_idx])
            transformed_matrix[:, -self.ACTION_SPACE_DIM+1:] = 1

            purged_state = np.subtract(curr_state[:, attr_idx], neg_delta.any(axis=1), dtype=int)\
                            .clip(min=0).astype(bool)
            next_state = np.add(purged_state, pos_delta.any(axis=1), dtype=int).clip(max=1).astype(bool)
            self._attribute_tensor[t + 1, :, attr_idx] = next_state

            self._instantiate_attribute_grounded_schemas(attr_idx, t+1, reference_matrix,
                                                         self._W_pos[attr_idx], pos_delta, neg_delta)

        # raise void bit
        void_entity_mask = ~self._attribute_tensor[t + 1, :, :].any(axis=1)
        self._attribute_tensor[t + 1, void_entity_mask, self.VOID_IDX] = True

    def _predict_next_reward_layer(self, t):
        """
        t: time at which last known attributes are located
        predict from t to (t + 1)
        """
        src_slice = self._get_tensor_slice(t, 'attributes')  # (FRAME_STACK_SIZE x N x M)
        transformed_matrix = self._shaper.transform_matrix(src_slice)
        reference_matrix = self._get_reference_matrix(t)

        is_pos_reward_predicted = False
        for reward_idx, R in enumerate(self._R):
            predicted_matrix = ~(~transformed_matrix @ R)
            self._reward_tensor[t + 1, reward_idx] = predicted_matrix.any()  # OR over all dimensions

            n_pos_schemas_instantiated = \
                self._instantiate_reward_grounded_schemas(reward_idx, t + 1, reference_matrix, R, predicted_matrix)
            is_pos_reward_predicted |= bool(n_pos_schemas_instantiated)

        return is_pos_reward_predicted

    def forward_pass(self, entities_stack):
        """
        Fill attribute_nodes and reward_nodes with schema information
        """
        self._entities_stack = entities_stack
        src_tensor = self._get_env_attribute_tensor()

        self._init_attribute_tensor(src_tensor)
        self._init_reward_tensor()
        self._init_nodes()

        # propagate forward
        offset = self.FRAME_STACK_SIZE - 1
        for t in range(offset, offset + self.T):
            self._predict_next_attribute_layer(t)
            is_pos_reward_predicted = self._predict_next_reward_layer(t)

            if is_pos_reward_predicted:
                break

    def check_entities_for_correctness(self, t):
        n_predicted_balls = np.count_nonzero(self._attribute_tensor[t, :, self.BALL_IDX])
        if n_predicted_balls > 1:
            print('BAD_BALL: {} balls exist. t: {}'.format(n_predicted_balls, t))

    def get_ball_entity_idx(self, t):
        """
        :param t: time_step you need to look at
        :return: entity_idx of the ball
        """
        entities = self._attribute_tensor[t, :, :]
        row_indices = entities[:, self.BALL_IDX].nonzero()[0]  # returns tuple

        if row_indices.size > 1:
            print('BAD_N_BALLS, n: {}, t: {}'.format(row_indices.size, t))

        if row_indices:
            ball_idx = row_indices[0]
        else:
            ball_idx = None
        return ball_idx

    def get_attribute_tensor(self):
        return self._attribute_tensor
