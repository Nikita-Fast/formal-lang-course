from scipy import sparse
from pyformlang.finite_automaton import (
    State,
    NondeterministicFiniteAutomaton,
    FiniteAutomaton,
)

__all__ = ["BooleanMatrices"]


class BooleanMatrices:
    """
    Class representing boolean matrix decomposition of finite automaton
    """

    def __init__(self):
        self.num_states = 0
        self.start_states = set()
        self.final_states = set()
        self.bool_matrices = {}
        self.state_indices = {}

    @classmethod
    def from_automaton(cls, automaton: FiniteAutomaton):
        """
        Transform automaton to set of labeled boolean matrix
        Parameters
        ----------
        automaton
            Automaton for transforming
        Returns
        -------
        BooleanMatrices
            Result of transforming
        """

        automaton_matrix = cls()
        automaton_matrix.num_states = len(automaton.states)
        automaton_matrix.start_states = automaton.start_states
        automaton_matrix.final_states = automaton.final_states
        automaton_matrix.state_indices = {
            state: idx for idx, state in enumerate(automaton.states)
        }

        for s_from, trans in automaton.to_dict().items():
            for label, states_to in trans.items():
                if not isinstance(states_to, set):
                    states_to = {states_to}
                for s_to in states_to:
                    idx_from = automaton_matrix.state_indices[s_from]
                    idx_to = automaton_matrix.state_indices[s_to]
                    if label not in automaton_matrix.bool_matrices.keys():
                        automaton_matrix.bool_matrices[label] = sparse.csr_matrix(
                            (automaton_matrix.num_states, automaton_matrix.num_states),
                            dtype=bool,
                        )
                    automaton_matrix.bool_matrices[label][idx_from, idx_to] = True

        return automaton_matrix

    def to_automaton(self) -> NondeterministicFiniteAutomaton:
        """
        Transform set of labeled boolean matrix to automaton.
        Parameters
        ----------
        self
            Set of boolean matrix with label as key
        Returns
        -------
        BooleanMatrices
            Resulting automaton
        """

        automaton = NondeterministicFiniteAutomaton()
        for label in self.bool_matrices.keys():
            for s_from, s_to in zip(*self.bool_matrices[label].nonzero()):
                automaton.add_transition(s_from, label, s_to)

        for state in self.start_states:
            automaton.add_start_state(State(state))

        for state in self.final_states:
            automaton.add_final_state(State(state))

        return automaton

    @property
    def get_states(self):
        return self.state_indices.keys()

    @property
    def get_start_states(self):
        return self.start_states.copy()

    @property
    def get_final_states(self):
        return self.final_states.copy()

    def get_transitive_closure(self):
        """
        Get transitive closure of sparse.csr_matrix
        Parameters
        ----------
        self
            Class exemplar
        Returns
        -------
            Transitive closure
        """
        tc = sparse.csr_matrix((0, 0), dtype=bool)

        if len(self.bool_matrices) != 0:
            tc = sum(self.bool_matrices.values())
            prev_nnz = tc.nnz
            new_nnz = 0

            while prev_nnz != new_nnz:
                tc += tc @ tc
                prev_nnz, new_nnz = new_nnz, tc.nnz

        return tc

    def intersect(self, other):
        """
        Get intersection of two automatons
        Parameters
        ----------
        self
            First automaton
        other
            Second automaton
        Returns
        -------
        BooleanMatrices
            Result of intersection
        """
        res = BooleanMatrices()
        res.num_states = self.num_states * other.num_states
        common_labels = set(self.bool_matrices.keys()).union(other.bool_matrices.keys())

        for label in common_labels:
            if label not in self.bool_matrices.keys():
                self.bool_matrices[label] = sparse.csr_matrix(
                    (self.num_states, self.num_states), dtype=bool
                )
            if label not in other.bool_matrices.keys():
                other.bool_matrices[label] = sparse.csr_matrix(
                    (other.num_states, other.num_states), dtype=bool
                )

        for label in common_labels:
            res.bool_matrices[label] = sparse.kron(
                self.bool_matrices[label], other.bool_matrices[label], format="csr"
            )

        for state_first, state_first_idx in self.state_indices.items():
            for state_second, state_second_idx in other.state_indices.items():
                new_state = new_state_idx = (
                    state_first_idx * other.num_states + state_second_idx
                )
                res.state_indices[new_state] = new_state_idx

                if (
                    state_first in self.start_states
                    and state_second in other.start_states
                ):
                    res.start_states.add(new_state)

                if (
                    state_first in self.final_states
                    and state_second in other.final_states
                ):
                    res.final_states.add(new_state)
        return res

    def direct_sum(self, other: "BooleanMatrices"):
        d_sum = BooleanMatrices()
        d_sum.num_states = self.num_states + other.num_states

        common_symbols = self.bool_matrices.keys() & other.bool_matrices.keys()

        for symbol in common_symbols:
            d_sum.bool_matrices[symbol] = sparse.bmat(
                [
                    [self.bool_matrices[symbol], None],
                    [None, other.bool_matrices[symbol]],
                ]
            )

        # for state in self.states:
        #     d_sum.states.add(state)
        #     d_sum.state_indexes[state] = self.state_indexes[state]
        for state, idx in self.state_indices.items():
            d_sum.state_indices[state] = idx

            # если состояние является стартовым у одной из матриц, то и у матрицы прямой суммы оно тоже будет стартовым
            if state in self.start_states:
                d_sum.start_states.add(state)

            if state in self.final_states:
                d_sum.final_states.add(state)

        # for state in other.states:
        for state, idx in other.state_indices.items():
            new_state = State(state.value + self.num_states)
            # d_sum.states.add(new_state)
            # d_sum.state_indexes[new_state] = (
            #     other.state_indexes[state] + self.num_states
            # )
            d_sum.state_indices[new_state] = (
                other.state_indices[state] + self.num_states
            )

            if state in other.start_states:
                d_sum.start_states.add(new_state)

            if state in other.final_states:
                d_sum.final_states.add(new_state)

        return d_sum

    def get_state_by_index(self, index):
        for state, ind in self.state_indices.items():
            if ind == index:
                return state
