import unittest

import numpy as np

from model.schema_learner import *
from model.constants import Constants as C

C.SCHEMA_VEC_SIZE = 3
C.N_PREDICTABLE_ATTRIBUTES = 1


class TestPurgeMatrixColumns(unittest.TestCase):
    def test_matrix(self):
        learner = GreedySchemaLearner()

        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])
        columns = np.array([1])
        answer = np.array([[1, 3, 1],
                           [4, 6, 1],
                           [7, 9, 1]])

        result = learner._purge_matrix_columns(matrix, columns)
        cmp = np.array_equal(result, answer)
        self.assertTrue(cmp)


class TestDeleteAttrSchemaVectors(unittest.TestCase):
    def test_matrix(self):
        learner = GreedySchemaLearner()

        for attr_idx in range(C.N_PREDICTABLE_ATTRIBUTES):
            learner._W_creation[attr_idx] = np.array([[1, 2, 3, 8],
                                                      [4, 5, 6, 8],
                                                      [7, 8, 9, 8]])
            columns = np.array([1, 3])
            answer = np.array([[1, 3, 1, 1],
                               [4, 6, 1, 1],
                               [7, 9, 1, 1]])

            learner._n_attr_schemas[attr_idx] = 4

            learner._delete_attr_schema_vectors(attr_idx, columns)
            result = learner._W_creation[attr_idx]
            cmp = np.array_equal(result, answer)
            self.assertTrue(cmp)
            self.assertEqual(learner._n_attr_schemas[attr_idx], 2)


class TestDeleteRewardSchemaVectors(unittest.TestCase):
    def test_matrix(self):
        learner = GreedySchemaLearner()

        learner._R = np.array([[1, 2, 3, 8],
                               [4, 5, 6, 8],
                               [7, 8, 9, 8]])
        columns = np.array([1, 3])
        answer = np.array([[1, 3, 1, 1],
                           [4, 6, 1, 1],
                           [7, 9, 1, 1]])

        learner._n_reward_schemas = 4

        learner._delete_reward_schema_vectors(columns)
        result = learner._R
        cmp = np.array_equal(result, answer)
        self.assertTrue(cmp)
        self.assertEqual(learner._n_reward_schemas, 2)


class TestLearn(unittest.TestCase):
    def test_full_comb(self):
        x = np.array([[0, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [0, 1, 1],
                      [1, 1, 0],
                      [1, 0, 1]]).astype(bool)

        y = np.array([0, 0, 0, 0, 0, 0, 1]).astype(bool).reshape(-1, 1)
        r = np.array([0, 0, 0, 0, 1, 1, 0]).astype(bool)

        # ---
        learner = GreedySchemaLearner()

        batch_size = 2
        for i in range(4):
            batch_slice = np.index_exp[i*batch_size: (i+1)*batch_size]

            learner.take_batch(GreedySchemaLearner.Batch(x[batch_slice],
                                                         y[batch_slice],
                                                         r[batch_slice]))

        learner.learn()

        W, R = learner.get_weights()
        print('W:')
        print(W)
        print('R:')
        print(R)

        print('new batches go BRRR')

        batch_size = 1
        new_r = np.array([1, 1, 1, 1, 0, 1, 1]).astype(bool)

        for i in range(7):
            batch_slice = np.index_exp[i*batch_size: (i+1)*batch_size]

            learner.take_batch(GreedySchemaLearner.Batch(x[batch_slice],
                                                         y[batch_slice],
                                                         new_r[batch_slice]))
        learner.learn()
        W, R = learner.get_weights()
        print('W:')
        print(W)
        print('R:')
        print(R)
























