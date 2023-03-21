# from scipy import sparse
# from pyformlang.finite_automaton import (
#     State,
#     NondeterministicFiniteAutomaton,
#     FiniteAutomaton,
# )
# from project.rsm import RSM
# __all__ = ["BooleanMatrices"]
#
#
# class BooleanMatrices:
#     """
#     Class representing boolean matrix decomposition of finite automaton
#     """
#
#     def __init__(self):
#         self.num_states = 0
#         self.start_states = set()
#         self.final_states = set()
#         self.bool_matrices = {}
#         self.state_indices = {}
#
#     @classmethod
#     def from_automaton(cls, automaton: FiniteAutomaton):
#         """
#         Transform automaton to set of labeled boolean matrix
#         Parameters
#         ----------
#         automaton
#             Automaton for transforming
#         Returns
#         -------
#         BooleanMatrices
#             Result of transforming
#         """
#
#         automaton_matrix = cls()
#         automaton_matrix.num_states = len(automaton.states)
#         automaton_matrix.start_states = automaton.start_states
#         automaton_matrix.final_states = automaton.final_states
#         automaton_matrix.state_indices = {
#             state: idx for idx, state in enumerate(automaton.states)
#         }
#
#         for s_from, trans in automaton.to_dict().items():
#             for label, states_to in trans.items():
#                 if not isinstance(states_to, set):
#                     states_to = {states_to}
#                 for s_to in states_to:
#                     idx_from = automaton_matrix.state_indices[s_from]
#                     idx_to = automaton_matrix.state_indices[s_to]
#                     if label not in automaton_matrix.bool_matrices.keys():
#                         automaton_matrix.bool_matrices[label] = sparse.csr_matrix(
#                             (automaton_matrix.num_states, automaton_matrix.num_states),
#                             dtype=bool,
#                         )
#                     automaton_matrix.bool_matrices[label][idx_from, idx_to] = True
#
#         return automaton_matrix
#
#     @classmethod
#     def from_rsm(cls, rsm: RSM):
#         states, start_states, final_states = set(), set(), set()
#         for var, nfa in rsm.boxes.items():
#             for s in nfa.states:
#                 state = State((var, s.value))
#                 states.add(state)
#                 if s in nfa.start_states:
#                     start_states.add(state)
#                 if s in nfa.final_states:
#                     final_states.add(state)
#
#         states = sorted(states, key=lambda v: (v.value[0].value, v.value[1]))
#         state_to_idx = {s: i for i, s in enumerate(states)}
#
#         automaton_matrix = cls()
#         automaton_matrix.num_states = len(states)
#         automaton_matrix.start_states = start_states
#         automaton_matrix.final_states = final_states
#         automaton_matrix.state_indices = state_to_idx
#
#         for var, nfa in rsm.boxes.items():
#             for state_from, transitions in nfa.to_dict().items():
#                 for label, states_to in transitions.items():
#                     if label not in automaton_matrix.bool_matrices.keys():
#                         automaton_matrix.bool_matrices[label] = sparse.csr_matrix(
#                             (automaton_matrix.num_states, automaton_matrix.num_states),
#                             dtype=bool,
#                         )
#                     states_to = states_to if isinstance(states_to, set) else {states_to}
#                     for state_to in states_to:
#                         automaton_matrix.bool_matrices[label][
#                             state_to_idx[State((var, state_from.value))],
#                             state_to_idx[State((var, state_to.value))],
#                         ] = True
#
#         return automaton_matrix
#
#     def to_automaton(self) -> NondeterministicFiniteAutomaton:
#         """
#         Transform set of labeled boolean matrix to automaton.
#         Parameters
#         ----------
#         self
#             Set of boolean matrix with label as key
#         Returns
#         -------
#         BooleanMatrices
#             Resulting automaton
#         """
#
#         automaton = NondeterministicFiniteAutomaton()
#         for label in self.bool_matrices.keys():
#             for s_from, s_to in zip(*self.bool_matrices[label].nonzero()):
#                 automaton.add_transition(s_from, label, s_to)
#
#         for state in self.start_states:
#             automaton.add_start_state(State(state))
#
#         for state in self.final_states:
#             automaton.add_final_state(State(state))
#
#         return automaton
#
#     @property
#     def get_states(self):
#         return self.state_indices.keys()
#
#     @property
#     def get_start_states(self):
#         return self.start_states.copy()
#
#     @property
#     def get_final_states(self):
#         return self.final_states.copy()
#
#     def transitive_closure(self):
#         """
#         Get transitive closure of sparse.csr_matrix
#         Parameters
#         ----------
#         self
#             Class exemplar
#         Returns
#         -------
#             Transitive closure
#         """
#         tc = sparse.csr_matrix((0, 0), dtype=bool)
#
#         if len(self.bool_matrices) != 0:
#             tc = sum(self.bool_matrices.values())
#             prev_nnz = tc.nnz
#             new_nnz = 0
#
#             while prev_nnz != new_nnz:
#                 tc += tc @ tc
#                 prev_nnz, new_nnz = new_nnz, tc.nnz
#
#         return tc
#
#     def intersect(self, other):
#         """
#         Get intersection of two automatons
#         Parameters
#         ----------
#         self
#             First automaton
#         other
#             Second automaton
#         Returns
#         -------
#         BooleanMatrices
#             Result of intersection
#         """
#         res = BooleanMatrices()
#         res.num_states = self.num_states * other.num_states
#         common_labels = set(self.bool_matrices.keys()).union(other.bool_matrices.keys())
#
#         for label in common_labels:
#             if label not in self.bool_matrices.keys():
#                 self.bool_matrices[label] = sparse.csr_matrix(
#                     (self.num_states, self.num_states), dtype=bool
#                 )
#             if label not in other.bool_matrices.keys():
#                 other.bool_matrices[label] = sparse.csr_matrix(
#                     (other.num_states, other.num_states), dtype=bool
#                 )
#
#         for label in common_labels:
#             res.bool_matrices[label] = sparse.kron(
#                 self.bool_matrices[label], other.bool_matrices[label], format="csr"
#             )
#
#         for state_first, state_first_idx in self.state_indices.items():
#             for state_second, state_second_idx in other.state_indices.items():
#                 new_state = new_state_idx = (
#                     state_first_idx * other.num_states + state_second_idx
#                 )
#                 res.state_indices[new_state] = new_state_idx
#
#                 if (
#                     state_first in self.start_states
#                     and state_second in other.start_states
#                 ):
#                     res.start_states.add(new_state)
#
#                 if (
#                     state_first in self.final_states
#                     and state_second in other.final_states
#                 ):
#                     res.final_states.add(new_state)
#         return res
#
#     def direct_sum(self, other: "BooleanMatrices"):
#         d_sum = BooleanMatrices()
#         d_sum.num_states = self.num_states + other.num_states
#
#         common_symbols = self.bool_matrices.keys() & other.bool_matrices.keys()
#
#         for symbol in common_symbols:
#             d_sum.bool_matrices[symbol] = sparse.bmat(
#                 [
#                     [self.bool_matrices[symbol], None],
#                     [None, other.bool_matrices[symbol]],
#                 ]
#             )
#
#         # for state in self.states:
#         #     d_sum.states.add(state)
#         #     d_sum.state_indexes[state] = self.state_indexes[state]
#         for state, idx in self.state_indices.items():
#             d_sum.state_indices[state] = idx
#
#             # если состояние является стартовым у одной из матриц, то и у матрицы прямой суммы оно тоже будет стартовым
#             if state in self.start_states:
#                 d_sum.start_states.add(state)
#
#             if state in self.final_states:
#                 d_sum.final_states.add(state)
#
#         # for state in other.states:
#         for state, idx in other.state_indices.items():
#             new_state = State(state.value + self.num_states)
#             # d_sum.states.add(new_state)
#             # d_sum.state_indexes[new_state] = (
#             #     other.state_indexes[state] + self.num_states
#             # )
#             d_sum.state_indices[new_state] = (
#                 other.state_indices[state] + self.num_states
#             )
#
#             if state in other.start_states:
#                 d_sum.start_states.add(new_state)
#
#             if state in other.final_states:
#                 d_sum.final_states.add(new_state)
#
#         return d_sum
#
#     def get_state_by_index(self, index):
#         for state, ind in self.state_indices.items():
#             if ind == index:
#                 return state


from pyformlang.finite_automaton import (
    NondeterministicFiniteAutomaton,
    State,
    EpsilonNFA,
)
from scipy import sparse
from scipy.sparse import dok_matrix

from project.rsm import RSM


class BooleanMatrices:
    def __init__(
        self,
        state_to_index: dict,
        start_states: set,
        final_states: set,
        bool_matrices: dict,
    ):
        self.state_to_index = state_to_index
        self.start_states = start_states
        self.final_states = final_states
        self.bool_matrices = bool_matrices
        self.num_states = len(self.state_to_index.keys())

    @staticmethod
    def _create_boolean_matrices(nfa: EpsilonNFA, state_to_index: dict[State, int]):
        b_matrices = {}
        for state_from, transition in nfa.to_dict().items():
            for label, states_to in transition.items():
                if not isinstance(states_to, set):
                    states_to = {states_to}
                for state_to in states_to:
                    index_from = state_to_index[state_from]
                    index_to = state_to_index[state_to]

                    if label not in b_matrices:
                        b_matrices[label] = sparse.dok_matrix(
                            (len(nfa.states), len(nfa.states)), dtype=bool
                        )

                    b_matrices[label][index_from, index_to] = True

        return b_matrices

    @classmethod
    def from_automaton(cls, nfa: EpsilonNFA) -> "BooleanMatrices":
        state_to_index = {state: index for index, state in enumerate(nfa.states)}
        return cls(
            state_to_index,
            nfa.start_states.copy(),
            nfa.final_states.copy(),
            cls._create_boolean_matrices(nfa, state_to_index),
        )

    @classmethod
    def from_rsm(cls, rsm: RSM) -> "BooleanMatrices":
        states, start_states, final_states = set(), set(), set()
        for v, dfa in rsm.boxes.items():
            for s in dfa.states:
                state = State((v, s.value))
                states.add(state)
                if s in dfa.start_states:
                    start_states.add(state)
                if s in dfa.final_states:
                    final_states.add(state)

        states = sorted(states, key=lambda s: s.value)
        state_to_index = {s: i for i, s in enumerate(states)}
        bool_matrices = dict()

        for v, dfa in rsm.boxes.items():
            for state_from, transitions in dfa.to_dict().items():
                for label, states_to in transitions.items():
                    states_to = states_to if isinstance(states_to, set) else {states_to}
                    for state_to in states_to:
                        if label not in bool_matrices:
                            bool_matrices[label] = dok_matrix(
                                (len(states), len(states)), dtype=bool
                            )

                        i = state_to_index[State((v, state_from.value))]
                        j = state_to_index[State((v, state_to.value))]
                        bool_matrices[label][i, j] = True
        return cls(
            state_to_index,
            start_states,
            final_states,
            bool_matrices,
        )

    def to_automaton(self):
        nfa = NondeterministicFiniteAutomaton()

        for symbol, matrix in self.bool_matrices.items():
            rows, columns = matrix.nonzero()
            for row, column in zip(rows, columns):
                nfa.add_transition(row, symbol, column)

        for state in self.start_states:
            nfa.add_start_state(state)

        for state in self.final_states:
            nfa.add_final_state(state)

        return nfa

    def transitive_closure(self: "BooleanMatrices") -> dok_matrix:
        if not self.bool_matrices.values():
            return dok_matrix((1, 1))
        transitive_closure = sum(self.bool_matrices.values())
        prev_nnz = transitive_closure.nnz
        new_nnz = 0

        while prev_nnz != new_nnz:
            transitive_closure += transitive_closure @ transitive_closure
            prev_nnz, new_nnz = new_nnz, transitive_closure.nnz

        return transitive_closure

    def intersect(
        self: "BooleanMatrices", another: "BooleanMatrices"
    ) -> "BooleanMatrices":
        bool_matrices = dict()
        for label in self.bool_matrices.keys() & another.bool_matrices.keys():
            bool_matrices[label] = sparse.kron(
                self.bool_matrices[label], another.bool_matrices[label], format="dok"
            )

        state_to_index = dict()
        start_states, final_states = set(), set()
        for s1, s1_index in self.state_to_index.items():
            for s2, s2_index in another.state_to_index.items():
                index = s1_index * another.num_states + s2_index
                state = index
                state_to_index[state] = index

                if s1 in self.start_states and s2 in another.start_states:
                    start_states.add(state)

                if s1 in self.final_states and s2 in another.final_states:
                    final_states.add(state)

        return BooleanMatrices(
            state_to_index, start_states, final_states, bool_matrices
        )

    def get_start_states(self):
        return self.start_states.copy()

    def get_final_states(self):
        return self.final_states.copy()

    def get_state_by_index(self, index):
        for state, idx in self.state_to_index.items():
            if idx == index:
                return state

    def direct_sum(self, other: "BooleanMatrices"):
        common_symbols = self.bool_matrices.keys() & other.bool_matrices.keys()
        bool_matrices = dict()
        for symbol in common_symbols:
            bool_matrices[symbol] = sparse.bmat(
                [
                    [self.bool_matrices[symbol], None],
                    [None, other.bool_matrices[symbol]],
                ]
            )

        state_to_index = dict()
        start_states, final_states = set(), set()
        for state, index in self.state_to_index.items():
            state_to_index[state] = index

            # если состояние является стартовым у одной из матриц, то и у матрицы прямой суммы оно тоже будет стартовым
            if state in self.start_states:
                start_states.add(state)

            if state in self.final_states:
                final_states.add(state)

        for state, index in other.state_to_index.items():
            new_state = State(state.value + self.num_states)
            state_to_index[new_state] = index + self.num_states

            if state in other.start_states:
                start_states.add(new_state)

            if state in other.final_states:
                final_states.add(new_state)

        return BooleanMatrices(
            state_to_index, start_states, final_states, bool_matrices
        )
