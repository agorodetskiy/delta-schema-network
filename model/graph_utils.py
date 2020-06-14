import numpy as np
from collections import defaultdict
from .constants import Constants


class Schema(Constants):
    """Grounded schema type."""

    def __init__(self, t, attribute_preconditions, action_preconditions, vector):
        """
        preconditions: list of Nodes
        """
        self.t = t
        self.attribute_preconditions = attribute_preconditions
        self.action_preconditions = action_preconditions
        self.vector = vector

        self.is_reachable = None
        self.required_cumulative_actions = None
        self.harmfulness = None


class Node:
    def __init__(self, t):
        """
        schemas: list of schemas
        """
        self.t = t

        self.is_feasible = False
        self.is_reachable = None
        self.activating_schema = None  # reachable by this schema

        # map: action_idx -> list of Schema objects
        self.schemas = defaultdict(list)

        # set of actions at node.t - 1, with which node can be potentially activated
        self.acceptable_constraints = set()

    def reset(self):
        self.is_feasible = False
        self.is_reachable = None
        self.activating_schema = None
        self.schemas.clear()
        self.acceptable_constraints.clear()

    def add_schema(self, preconditions, vector):
        # in current implementation schemas are instantiated only on feasible nodes
        if not self.is_feasible:
            self.is_feasible = True

        attribute_preconditions = []
        action_preconditions = []

        for precondition in preconditions:
            if type(precondition) is Attribute:
                attribute_preconditions.append(precondition)
            elif type(precondition) is Action:
                # assuming only one action precondition
                # it's needed for simple action planning during reward backtrace
                if len(action_preconditions) >= 1:
                    print('schema is preconditioned more than on one action')
                    raise AssertionError
                action_preconditions.append(precondition)

                # update constraints of current node
                self.acceptable_constraints.add(precondition.idx)

        schema = Schema(self.t, attribute_preconditions, action_preconditions, vector)
        key = action_preconditions[0].idx if action_preconditions else None
        self.schemas[key].append(schema)

    def sort_schemas_by_harmfulness(self, neg_schemas):
        """
        :param neg_schemas: list of schemas for negative reward at the same time step
                            as node's time step
        """
        for schema in self.schemas:
            schema.compute_harmfulness(neg_schemas)

        self.schemas = sorted(self.schemas,
                              key=lambda x: x.harmfulness)

    def sort_schemas_by_priority(self):
        def cmp(schema):
            actions = schema.action_preconditions
            if len(actions) == 0:
                priority = 0
            elif actions[0].idx == 0:
                priority = 1
            else:
                priority = 2
            return priority
        
        self.schemas = sorted(self.schemas, key=cmp)


class Attribute(Node, Constants):
    def __init__(self, entity_idx, attribute_idx, t, prev_layer):
        """
        entity_idx: entity unique idx [0, N)
        attribute_idx: attribute index in entity's attribute vector
        """
        self.entity_idx = entity_idx
        self.attribute_idx = attribute_idx

        self.transition = prev_layer[entity_idx][attribute_idx] if prev_layer is not None else None
        self._transition = self.transition

        super().__init__(t)

    def reset(self, is_initially_reachable=False):
        super().reset()

        if is_initially_reachable or self.attribute_idx == self.VOID_IDX:
            self.is_reachable = True

        self.transition = self._transition


class FakeAttribute:
    """
    Attribute of entity that is out of screen (zero-padding in matrix)
    """
    pass


class Action:
    not_planned_idx = 0

    def __init__(self, idx, t):
        """
        action_idx: action unique idx
        """
        self.idx = idx
        self.t = t


class Reward(Node):
    sign2idx = {'pos': 0,
                'neg': 1}
    allowed_signs = sign2idx.keys()

    def __init__(self, idx, t):
        self.idx = idx
        self.weight = None

        super().__init__(t)

    def set_weight(self, w):
        self.weight = w

    def reset(self):
        super().reset()
        self.weight = None


class Constraint:
    def __init__(self):
        self.action_idx = None
        self.committed_nodes = set()

    def reset(self):
        self.action_idx = None
        self.committed_nodes.clear()
