from typing import Dict, Set, Any

from pyformlang.finite_automaton import (
    State,
    EpsilonNFA,
)
from scipy import sparse
from scipy.sparse import dok_matrix, lil_matrix, csr_matrix, csc_matrix, csr_array

# from project.rpq_bfs import create_front_for_each, create_front, correct_front_part
from project.rsm import RSM


class BooleanMatrices:
    def __init__(
        self,
        number_of_states: int,
        state_to_index: Dict[State, int],
        index_to_state: Dict[int, State],
        start_state_indexes: Set[int],
        final_state_indexes: Set[int],
        boolean_matrix,
    ):
        self.number_of_states = number_of_states
        self.state_to_index = state_to_index
        self.index_to_state = index_to_state
        self.start_state_indexes = start_state_indexes
        self.final_state_indexes = final_state_indexes
        self.bool_matrices = boolean_matrix

    @classmethod
    def from_automaton(cls, nfa: EpsilonNFA, type_of_matrix) -> "BooleanMatrices":
        number_of_states = len(nfa.states)
        state_to_index = {state: index for index, state in enumerate(nfa.states)}
        index_to_state = {index: state for state, index in state_to_index.items()}
        start_state_indexes = {
            i for i, s in enumerate(nfa.states) if s in nfa.start_states
        }
        final_state_indexes = {
            i for i, s in enumerate(nfa.states) if s in nfa.final_states
        }
        return cls(
            number_of_states,
            state_to_index,
            index_to_state,
            start_state_indexes,
            final_state_indexes,
            cls.create_boolean_matrix_from_nfa(nfa, state_to_index, type_of_matrix),
        )

    @classmethod
    def from_rsm(cls, rsm: RSM, type_of_matrix) -> "BooleanMatrices":
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
        number_of_states = len(states)
        state_to_index = {s: i for i, s in enumerate(states)}
        index_to_state = {i: s for i, s in enumerate(states)}

        start_state_indexes = {state_to_index[s] for s in start_states}
        final_state_indexes = {state_to_index[s] for s in final_states}

        bool_matrices = dict()
        for v, dfa in rsm.boxes.items():
            for state_from, transitions in dfa.to_dict().items():
                for label, states_to in transitions.items():
                    states_to = states_to if isinstance(states_to, set) else {states_to}
                    for state_to in states_to:
                        if label not in bool_matrices:
                            bool_matrices[label] = type_of_matrix(
                                (number_of_states, number_of_states), dtype=bool
                            )

                        i = state_to_index[State((v, state_from.value))]
                        j = state_to_index[State((v, state_to.value))]
                        bool_matrices[label][i, j] = True
        return cls(
            number_of_states,
            state_to_index,
            index_to_state,
            start_state_indexes,
            final_state_indexes,
            bool_matrices,
        )

    @staticmethod
    def create_boolean_matrix_from_nfa(
        nfa: EpsilonNFA, state_to_index: dict[State, int], type_of_matrix=dok_matrix
    ):
        b_matrices = {}
        for state_from, transition in nfa.to_dict().items():
            for label, states_to in transition.items():
                if not isinstance(states_to, set):
                    states_to = {states_to}
                for state_to in states_to:

                    if label not in b_matrices:

                        b_matrices[label] = type_of_matrix(
                            (len(nfa.states), len(nfa.states)), dtype=bool
                        )

                    b_matrices[label][
                        state_to_index[state_from], state_to_index[state_to]
                    ] = True

        return b_matrices

    def to_automaton(self):
        nfa = EpsilonNFA()

        for symbol, matrix in self.bool_matrices.items():
            rows, columns = matrix.nonzero()
            for row, column in zip(rows, columns):
                nfa.add_transition(row, symbol, column)

        for i in self.start_state_indexes:
            nfa.add_start_state(State(i))

        for i in self.final_state_indexes:
            nfa.add_final_state(State(i))

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
                # todo format = type_of_matrix?
                self.bool_matrices[label],
                another.bool_matrices[label],
            )

        state_to_index = dict()
        start_state_indexes, final_state_indexes = set(), set()
        for s1, s1_index in self.state_to_index.items():
            for s2, s2_index in another.state_to_index.items():
                index = s1_index * another.number_of_states + s2_index
                state = State(index)
                state_to_index[state] = index

                if (
                    s1_index in self.start_state_indexes
                    and s2_index in another.start_state_indexes
                ):
                    start_state_indexes.add(index)

                if (
                    s1_index in self.final_state_indexes
                    and s2_index in another.final_state_indexes
                ):
                    final_state_indexes.add(index)

        return BooleanMatrices(
            self.number_of_states + another.number_of_states,
            state_to_index,
            {i: s for s, i in state_to_index.items()},
            start_state_indexes,
            final_state_indexes,
            bool_matrices,
        )

    def get_start_states(self) -> set[State]:
        return {self.index_to_state[i] for i in self.start_state_indexes}

    def get_final_states(self) -> set[State]:
        return {self.index_to_state[i] for i in self.final_state_indexes}

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
        start_state_indexes, final_state_indexes = set(), set()

        for state, index in self.state_to_index.items():
            state_to_index[state] = index

            # если состояние является стартовым у одной из матриц, то и у матрицы прямой суммы оно тоже будет стартовым
            if index in self.start_state_indexes:
                start_state_indexes.add(index)

            if index in self.final_state_indexes:
                final_state_indexes.add(index)

        for state, index in other.state_to_index.items():
            new_state = State(state.value + self.number_of_states)
            new_index = index + self.number_of_states
            state_to_index[new_state] = new_index

            if index in other.start_state_indexes:
                start_state_indexes.add(new_index)

            if index in other.final_state_indexes:
                final_state_indexes.add(new_index)

        return BooleanMatrices(
            self.number_of_states + other.number_of_states,
            state_to_index,
            {i: s for s, i in state_to_index.items()},
            start_state_indexes,
            final_state_indexes,
            bool_matrices,
        )

    def rpq_bfs(
        self: "BooleanMatrices",
        other: "BooleanMatrices",
        start_states: set = None,
        final_states: set = None,
        separately_for_each: bool = False,
        type_of_matrix=dok_matrix,
    ):
        if start_states is None:
            start_states = set(self.state_to_index.values())
        if final_states is None:
            final_states = set(self.state_to_index.values())

        if len(start_states) == 0 or len(final_states) == 0:
            return set()

        # step-1 form bm
        graph_bm = self
        constraint_bm = other

        k = constraint_bm.number_of_states
        n = graph_bm.number_of_states

        # step-2 apply direct sum
        d_sum = constraint_bm.direct_sum(graph_bm)

        # helper functions
        def _correct_front_part(
            front_part: csr_array, constraint_num_states: int, graph_num_states: int
        ):
            corrected_front_part = type_of_matrix(front_part.shape, dtype=bool)

            for i, j in zip(*front_part.nonzero()):
                if j < constraint_num_states:
                    row_right_part = front_part.getrow(i).tolil()[
                        [0], constraint_num_states:
                    ]
                    if row_right_part.nnz > 0:
                        row_shift = i // graph_num_states * graph_num_states
                        corrected_front_part[row_shift + j, j] = True
                        corrected_front_part[
                            row_shift + j, constraint_num_states:
                        ] = row_right_part

            return corrected_front_part.tocsr()

        def _create_front(
            graph_bm: BooleanMatrices,
            constraint_bm: BooleanMatrices,
            graph_start_states_indexes,
        ) -> csr_matrix:
            right_part_of_row = type_of_matrix(
                (1, graph_bm.number_of_states), dtype=bool
            )

            for i in graph_start_states_indexes:
                right_part_of_row[0, i] = True
            right_part_of_row = right_part_of_row.tocsr()

            front = sparse.csr_matrix(
                (
                    constraint_bm.number_of_states,
                    constraint_bm.number_of_states + graph_bm.number_of_states,
                ),
                dtype=bool,
            )

            for i in constraint_bm.start_state_indexes:
                front[i, i] = True
                front[i, constraint_bm.number_of_states :] = right_part_of_row

            return front

        def _create_front_for_each(
            graph_bm: BooleanMatrices,
            constraint_bm: BooleanMatrices,
            graph_start_states_indexes,
        ):
            front = sparse.vstack(
                [
                    _create_front(graph_bm, constraint_bm, {i})
                    for i in graph_start_states_indexes
                ]
            )

            return front

        # step-3 form front
        graph_start_states_indexes = list(graph_bm.start_state_indexes)

        if separately_for_each:
            front = _create_front_for_each(
                graph_bm, constraint_bm, graph_start_states_indexes
            )
        else:
            front = _create_front(graph_bm, constraint_bm, graph_start_states_indexes)

        # step-4 while-loop with new_front = front @ d_sum
        # думаю надо копировать в том числе содержимое фронта, а не только его форму

        # visited = csr_array(front)
        visited = type_of_matrix(front.shape, dtype=bool)

        while True:
            old_visited_nnz = visited.nnz

            for d_sum_bm in d_sum.bool_matrices.values():
                front_part = visited @ d_sum_bm if front is None else front @ d_sum_bm
                # step-5, 6
                visited += _correct_front_part(front_part, k, n)

            front = None

            if visited.nnz == old_visited_nnz:
                break
        # step-7
        result = set()
        for i, j in zip(*visited.nonzero()):
            if j >= k:

                constraint_state_index = i % k  # % для случая separated_for_each
                graph_state_index = j - k

                if (
                    constraint_state_index in constraint_bm.final_state_indexes
                    and graph_state_index in graph_bm.final_state_indexes
                ):
                    graph_state = graph_bm.index_to_state[graph_state_index]

                    if not separately_for_each:
                        result.add(graph_state)
                    else:
                        result.add((State(i // k), State(graph_state_index)))

        return result
