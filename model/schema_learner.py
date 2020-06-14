from collections import namedtuple
import os
import time

import numpy as np
import mip.model as mip

from model.constants import Constants as C
from model.visualizer import Visualizer


class MipModel:
    """
    instantiated for single attr_idx
    """
    MAX_OPT_SECONDS = 60

    def __init__(self):
        if C.LEARNING_SOLVER == 'cbc':
            solver = mip.CBC
        elif C.LEARNING_SOLVER == 'gurobi':
            solver = mip.GUROBI
        else:
            assert False

        self._model = mip.Model(mip.MINIMIZE, solver_name=solver)
        self._model.verbose = 0
        self._model.threads = C.N_LEARNING_THREADS
        # self._model.emphasis = 1  # feasibility

        self._w = [self._model.add_var(var_type='B') for _ in range(C.SCHEMA_VEC_SIZE)]
        self._constraints_buff = np.empty(0, dtype=object)

    def add_to_constraints_buff(self, batch, unique_idx, replay_renewed_indices=None):
        augmented_entities, target = batch
        batch_size = augmented_entities.shape[0]

        new_constraints = np.empty(batch_size, dtype=object)
        lin_combs = (1 - augmented_entities) @ self._w
        new_constraints[~target] = [lc >= 1 for lc in lin_combs[~target]]
        new_constraints[target] = [lc == 0 for lc in lin_combs[target]]

        concat_constraints = np.concatenate((self._constraints_buff, new_constraints), axis=0)
        self._constraints_buff = concat_constraints[unique_idx]

        if replay_renewed_indices is not None:
            for idx in replay_renewed_indices:
                constr = self._constraints_buff[idx]
                assert constr.sense == '=', constr.sense
                assert constr.const == 0, constr.const
                constr = constr >= 1
                assert constr.sense == '>', constr.sense
                assert constr.const == -1, constr.const
                self._constraints_buff[idx] = constr

    def optimize(self, objective_coefficients, zp_nl_mask, solved):
        model = self._model

        # add objective
        model.objective = mip.xsum(x_i * w_i for x_i, w_i in zip(objective_coefficients, self._w))

        # add constraints
        model.remove([constr for constr in model.constrs])

        constraints_mask = zp_nl_mask
        constraints_mask[solved] = True

        constraints_to_add = self._constraints_buff[constraints_mask]
        for constraint in constraints_to_add:
            model.add_constr(constraint)

        # optimize
        status = model.optimize(max_seconds=self.MAX_OPT_SECONDS)

        if status == mip.OptimizationStatus.OPTIMAL:
            print('Optimal solution cost {} found'.format(
                model.objective_value))
        elif status == mip.OptimizationStatus.FEASIBLE:
            print('Sol.cost {} found, best possible: {}'.format(
                model.objective_value, model.objective_bound))
        elif status == mip.OptimizationStatus.NO_SOLUTION_FOUND:
            print('No feasible solution found, lower bound is: {}'.format(
                model.objective_bound))
        else:
            print('Optimization FAILED: {}'.format(status))

        if status == mip.OptimizationStatus.OPTIMAL or status == mip.OptimizationStatus.FEASIBLE:
            schema_vec = np.array([v.x for v in model.vars])
        else:
            schema_vec = None
        return schema_vec


class ParamMatrix:
    """
    Vectors are stored as columns
    """
    def __init__(self):
        self._VEC_SIZE = C.SCHEMA_VEC_SIZE
        self._CAPACITY = C.L
        self._data = np.ones((self._VEC_SIZE, self._CAPACITY), dtype=bool)
        self._n_vectors = 0

    @property
    def mult_me(self):
        return self._data[:, :max(self._n_vectors, 1)]

    def has_free_space(self):
        return self._n_vectors < self._CAPACITY

    def add_vector(self, vec):
        if self._n_vectors < self._CAPACITY:
            self._data[:, self._n_vectors] = vec
            self._n_vectors += 1

    def purge_vectors(self, vec_indices):
        if not vec_indices.size:
            return

        assert ((vec_indices >= 0) & (vec_indices < self._n_vectors)).all()
        n_purged = vec_indices.size
        purged_matrix = np.delete(self._data, vec_indices, axis=1)
        padding = np.ones((self._VEC_SIZE, n_purged), dtype=bool)

        self._data = np.hstack((purged_matrix, padding))
        self._n_vectors -= n_purged

    def get_matrix(self):
        return self._data[:, :self._n_vectors]

    def set_matrix(self, matrix):
        if matrix is None:
            return

        n_rows, n_cols = matrix.shape
        assert n_rows == self._VEC_SIZE
        assert n_cols <= self._CAPACITY

        self._data[:, :n_cols] = matrix
        self._data[:, n_cols:] = 1
        self._n_vectors = n_cols


class GreedySchemaLearner:
    Batch = namedtuple('Batch', ['x', 'y_creation', 'y_destruction', 'r'])
    CREATION_T, DESTRUCTION_T, REWARD_T = range(3)
    ATTR_SCHEMA_TYPES = (CREATION_T, DESTRUCTION_T)

    @Visualizer.measure_time('Learner')
    def __init__(self):
        self._params = [[ParamMatrix() for _ in range(C.N_PREDICTABLE_ATTRIBUTES)]
                        for _ in range(2)]
        self._R = ParamMatrix()

        self._buff = []
        self._replay = self.Batch(np.empty((0, C.SCHEMA_VEC_SIZE), dtype=bool),
                                  np.empty((0, C.N_PREDICTABLE_ATTRIBUTES), dtype=bool),
                                  np.empty((0, C.N_PREDICTABLE_ATTRIBUTES), dtype=bool),
                                  np.empty(0, dtype=bool))

        self._attr_mip_models = [[MipModel() for _ in range(C.N_PREDICTABLE_ATTRIBUTES)]
                                 for _ in range(2)]
        self._reward_mip_model = MipModel()
        self._solved = []

        self._curr_iter = None
        self._visualizer = Visualizer(None, None, None)

    def set_curr_iter(self, curr_iter):
        self._curr_iter = curr_iter
        self._visualizer.set_iter(curr_iter)

    def set_params(self, W_pos, W_neg, R):
        for schema_type, params in zip(self.ATTR_SCHEMA_TYPES, (W_pos, W_neg)):
            for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
                self._params[schema_type][attr_idx].set_matrix(params[attr_idx])

        self._R.set_matrix(R[0])

    def _handle_duplicates(self, batch, return_index=False):
        augmented_entities, *rest = batch
        samples, idx = np.unique(augmented_entities, axis=0, return_index=True)
        out = self.Batch(samples, *[vec[idx] for vec in rest])
        if return_index:
            return out, idx
        return out

    def take_batch(self, batch):
        for part in batch:
            assert part.dtype == bool

        if batch.x.size:
            assert np.all(batch.r == batch.r[0])
            filtered_batch = self._handle_duplicates(batch)
            self._buff.append(filtered_batch)

    def _get_buff_batch(self):
        out = None
        if self._buff:
            # sort buff to keep r = 0 entries
            self._buff = sorted(self._buff, key=lambda b: b.r[0])

            batch = self.Batch(*[np.concatenate(minibatches_part, axis=0)
                                 for minibatches_part in zip(*self._buff)])

            out = self._handle_duplicates(batch)
            assert isinstance(out, self.Batch)
            self._buff.clear()
        return out

    def _add_to_replay_and_constraints_buff(self, batch):
        batch_size = len(batch.x)
        old_replay_size = len(self._replay.x)

        # concatenate replay + batch
        concat_batch = self.Batch(*[np.concatenate((a, b), axis=0)
                                    for a, b in zip(self._replay, batch)])
        # remove duplicates
        self._replay, unique_idx = self._handle_duplicates(concat_batch, return_index=True)

        concat_size = len(concat_batch.x)

        # find r = 0 duplicates (they can only locate in batch)
        duplicates_mask_concat = np.ones(concat_size, dtype=bool)
        duplicates_mask_concat[unique_idx] = False
        zero_reward_mask_concat = (concat_batch.r == 0)
        reward_renew_indices = np.nonzero(duplicates_mask_concat & zero_reward_mask_concat)[0]
        assert (reward_renew_indices >= old_replay_size).all()
        samples_to_update = concat_batch.x[reward_renew_indices]

        # update rewards to zero
        replay_indices_to_update = []
        for sample in samples_to_update:
            new_replay_indices = np.nonzero((self._replay.x == sample).all(axis=1))[0]
            assert len(new_replay_indices) == 1
            new_replay_idx = new_replay_indices[0]
            if self._replay.r[new_replay_idx] != 0:
                self._replay.r[new_replay_idx] = 0
                replay_indices_to_update.append(new_replay_idx)
        n_updated_indices = len(replay_indices_to_update)
        if n_updated_indices:
            print('Nullified rewards of {} old samples.'.format(n_updated_indices))

        # find non-duplicate indices in new batch (batch-based indexing)
        batch_mask_of_concat = unique_idx >= old_replay_size
        new_non_duplicate_indices = unique_idx[batch_mask_of_concat] - old_replay_size

        # find indices that will index constraints_buff + new_batch_unique synchronously with replay
        constraints_unique_idx = unique_idx.copy()
        constraints_unique_idx[batch_mask_of_concat] = old_replay_size + np.arange(len(new_non_duplicate_indices))

        for schema_type in self.ATTR_SCHEMA_TYPES:
            for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
                y = batch.y_creation if schema_type == self.CREATION_T else batch.y_destruction
                attr_batch = (batch.x[new_non_duplicate_indices],
                              y[new_non_duplicate_indices, attr_idx])
                self._attr_mip_models[schema_type][attr_idx].add_to_constraints_buff(attr_batch, constraints_unique_idx)

        reward_batch = (batch.x[new_non_duplicate_indices],
                        batch.r[new_non_duplicate_indices])
        self._reward_mip_model.add_to_constraints_buff(reward_batch, constraints_unique_idx,
                                                       replay_renewed_indices=replay_indices_to_update)

    def _get_replay_batch(self):
        if self._replay.x.size:
            out = self._replay
        else:
            out = None
        return out

    def _predict_attribute_delta(self, augmented_entities, attr_idx, attr_schema_type):
        assert augmented_entities.dtype == bool
        delta = ~(~augmented_entities @ self._params[attr_schema_type][attr_idx].mult_me)
        return delta

    def _predict_reward(self, augmented_entities):
        assert augmented_entities.dtype == bool
        reward_prediction = ~(~augmented_entities @ self._R.mult_me)
        return reward_prediction

    def _delete_incorrect_schemas(self, batch):
        augmented_entities, target_creation, target_destruction, rewards = batch
        for param_type in self.ATTR_SCHEMA_TYPES:
            for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
                target = target_creation if param_type == self.CREATION_T else target_destruction

                attr_delta = self._predict_attribute_delta(augmented_entities, attr_idx,
                                                           attr_schema_type=param_type)
                # false positive predictions
                mispredicted_samples_mask = attr_delta.any(axis=1) & ~target[:, attr_idx]

                incorrect_schemas_mask = attr_delta[mispredicted_samples_mask, :].any(axis=0)
                incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]

                assert incorrect_schemas_indices.ndim == 1
                n_incorrect_attr_schemas = incorrect_schemas_indices.size
                if n_incorrect_attr_schemas:
                    self._params[param_type][attr_idx].purge_vectors(incorrect_schemas_indices)
                    print('Deleted incorrect attr ({}) delta schemas: {} of {}'.format(
                        param_type, n_incorrect_attr_schemas, C.ENTITY_NAMES[attr_idx]))

        # regarding reward

        reward_prediction = self._predict_reward(augmented_entities)

        # false positive predictions
        mispredicted_samples_mask = reward_prediction.any(axis=1) & ~rewards

        incorrect_schemas_mask = reward_prediction[mispredicted_samples_mask, :].any(axis=0)
        incorrect_schemas_indices = np.nonzero(incorrect_schemas_mask)[0]

        assert incorrect_schemas_indices.ndim == 1
        n_incorrect_reward_schemas = incorrect_schemas_indices.size
        if n_incorrect_reward_schemas:
            self._R.purge_vectors(incorrect_schemas_indices)
            print('Deleted incorrect reward schemas: {}'.format(n_incorrect_reward_schemas))

        return n_incorrect_attr_schemas, n_incorrect_reward_schemas

    def _find_cluster(self, zp_pl_mask, zp_nl_mask, augmented_entities, target, attr_idx, opt_model):
        """
        augmented_entities: zero-predicted only
        target: scalar vector
        """
        assert augmented_entities.dtype == np.int
        assert target.dtype == np.int

        # find all entries, that can be potentially solved (have True labels)
        candidates = augmented_entities[zp_pl_mask]

        if not candidates.size:
            return None

        print('finding cluster...    zp pos samples: {}'.format(candidates.shape[0]))
        # print('augmented_entities: {}'.format(augmented_entities.shape[0]))

        zp_pl_indices = np.nonzero(zp_pl_mask)[0]

        # sample one entry and add it's idx to 'solved'
        idx = np.random.choice(zp_pl_indices)
        self._solved.append(idx)

        # resample candidates
        zp_pl_mask[idx] = False
        zp_pl_indices = np.nonzero(zp_pl_mask)[0]
        candidates = augmented_entities[zp_pl_mask]

        # solve LP
        objective_coefficients = (1 - candidates).sum(axis=0)
        objective_coefficients = list(objective_coefficients)

        new_schema_vector = opt_model.optimize(objective_coefficients, zp_nl_mask, self._solved)

        if new_schema_vector is None:
            print('Cannot find cluster!')
            return None

        # add all samples that are solved by just learned schema vector
        if candidates.size:
            new_predicted_attribute = (1 - candidates) @ new_schema_vector
            cluster_members_mask = np.isclose(new_predicted_attribute, 0, rtol=0, atol=C.ADDING_SCHEMA_TOLERANCE)
            n_new_members = np.count_nonzero(cluster_members_mask)

            if n_new_members:
                print('Also added to solved: {}'.format(n_new_members))
                self._solved.extend(zp_pl_indices[cluster_members_mask])

        return new_schema_vector

    def _simplify_schema(self, zp_nl_mask, schema_vector, opt_model):
        objective_coefficients = [1] * len(schema_vector)

        new_schema_vector = opt_model.optimize(objective_coefficients, zp_nl_mask, self._solved)
        assert new_schema_vector is not None
        return new_schema_vector

    def _binarize_schema(self, schema_vector):
        threshold = 0.5
        return schema_vector > threshold

    def _generate_new_schema(self, augmented_entities, targets, attr_idx, schema_type):
        if schema_type in self.ATTR_SCHEMA_TYPES:
            target = targets[:, attr_idx].astype(np.int, copy=False)
            prediction = self._predict_attribute_delta(augmented_entities, attr_idx, schema_type)
            opt_model = self._attr_mip_models[schema_type][attr_idx]
        elif schema_type == self.REWARD_T:
            target = targets.astype(np.int, copy=False)
            prediction = self._predict_reward(augmented_entities)
            opt_model = self._reward_mip_model
        else:
            assert False

        augmented_entities = augmented_entities.astype(np.int, copy=False)

        # sample only entries with zero-prediction
        zp_mask = ~prediction.any(axis=1)
        pl_mask = (target == 1)
        # pos and neg labels' masks
        zp_pl_mask = zp_mask & pl_mask
        zp_nl_mask = zp_mask & ~pl_mask

        new_schema_vector = self._find_cluster(zp_pl_mask, zp_nl_mask,
                                               augmented_entities, target, attr_idx,
                                               opt_model)
        if new_schema_vector is None:
            return None

        new_schema_vector = self._simplify_schema(zp_nl_mask, new_schema_vector, opt_model)
        new_schema_vector = self._binarize_schema(new_schema_vector)

        self._solved.clear()

        return new_schema_vector

    def get_params(self):
        W_pos = [W.get_matrix() for W in self._params[self.CREATION_T]]
        W_neg = [W.get_matrix() for W in self._params[self.DESTRUCTION_T]]
        R = [self._R.get_matrix()]
        return W_pos, W_neg, R

    def _dump_params(self):
        dir_name = 'dump'
        os.makedirs(dir_name, exist_ok=True)

        W_pos, W_neg, R = self.get_params()
        names = ['w_pos', 'w_neg', 'r']

        for params, name in zip((W_pos, W_neg, R), names):
            for idx, matrix in enumerate(params):
                file_name = name + '_{}'.format(idx)
                path = os.path.join(dir_name, file_name)
                np.save(path, matrix, allow_pickle=False)

    @Visualizer.measure_time('learn()')
    def learn(self):
        print('Launching learning procedure...')

        # get full batch from buffer
        buff_batch = self._get_buff_batch()
        if buff_batch is not None:
            self._add_to_replay_and_constraints_buff(buff_batch)
            self._delete_incorrect_schemas(buff_batch)

        # get all data to learn on
        replay_batch = self._get_replay_batch()
        if replay_batch is None:
            return

        # check if replay consistent
        # a, b = self._delete_incorrect_schemas(replay_batch)
        # assert a == 0 and b == 0

        augmented_entities, targets_construction, targets_destruction, rewards = replay_batch

        if C.DO_LEARN_ATTRIBUTE_PARAMS:
            for schema_type in self.ATTR_SCHEMA_TYPES:
                params = self._params[schema_type]
                targets = targets_construction if schema_type == self.CREATION_T else targets_destruction

                for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
                    while params[attr_idx].has_free_space():
                        new_schema_vec = self._generate_new_schema(augmented_entities, targets, attr_idx, schema_type)
                        if new_schema_vec is None:
                            break
                        params[attr_idx].add_vector(new_schema_vec)

        if C.DO_LEARN_REWARD_PARAMS:
            while self._R.has_free_space():
                new_schema_vec = self._generate_new_schema(augmented_entities, rewards, None, self.REWARD_T)
                if new_schema_vec is None:
                    break
                self._R.add_vector(new_schema_vec)

        self._dump_params()
        if C.VISUALIZE_SCHEMAS:
            W_pos, W_neg, R = self.get_params()
            self._visualizer.visualize_schemas(W_pos, W_neg, R)
