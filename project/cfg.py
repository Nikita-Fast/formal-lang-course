from os import PathLike
from typing import Union

import numpy as np
from pyformlang.cfg import CFG, Terminal
from pyformlang.cfg import Variable
from scipy.sparse import csr_matrix


def cfg_to_wcnf(cfg: Union[str, CFG], start: str = None) -> CFG:
    """
    Transform context-free-grammar to weak chomsky normal form
    Parameters
    ----------
    cfg: str | CFG
        Grammar
    start: str
        Start symbol
    Returns
    -------
    grammar: CFG
        Grammar in wcnf
    """

    if not isinstance(cfg, CFG):
        print("in not instance")
        cfg = read_cfg(cfg, start if start is not None else "S")
    print("after not instance")

    wcnf_cfg = cfg.remove_useless_symbols()
    print("after useless")
    wcnf_cfg = wcnf_cfg.eliminate_unit_productions()
    wcnf_cfg = wcnf_cfg.remove_useless_symbols()
    wcnf_productions = wcnf_cfg._get_productions_with_only_single_terminals()
    wcnf_productions = wcnf_cfg._decompose_productions(wcnf_productions)

    return CFG(start_symbol=wcnf_cfg._start_symbol, productions=set(wcnf_productions))


def read_grammar_to_str(path: str):
    """
    Load grammar from file to string representation
    Parameters
    ----------
    path: str
        Path to file
    Returns
    -------
    grammar: str
        String grammar representation
    """

    with open(path, "r") as f:
        data = f.read()
    return data


def read_cfg(grammar: str, start: str) -> CFG:
    """
    Build CFG from string representation of grammar
    Parameters
    ----------
    grammar: str
        String grammar
    start:
        Start symbol for grammar
    Returns
    -------
        Grammar as CFG class
    """
    print("in read_cfg")
    return CFG.from_text(grammar, Variable(start))


# def cfg_to_wcnf(cfg: Union[CFG, str], starting: str = "S") -> CFG:
#     # print("in cfg_to_wcnf")
#     # if isinstance(cfg, str):
#     #     cfg = CFG.from_text(cfg, Variable(starting))
#     #
#     # print("after isinstance")
#     # # По записям в моих конспектах единственное отличие в преобразовании CFG к WCNF в сравнении с
#     # # преобразованием CFG к CNF заключается в отсутствии того шага, на котором устраняются эпсилон-продукции.
#     #
#     # # Заимствуем код из pyformlang метода CFG.to_normal_form()
#     # new_cfg = (
#     #     cfg.remove_useless_symbols()
#     #     .eliminate_unit_productions()
#     #     .remove_useless_symbols()
#     # )
#     # print("after new_cfg")
#     #
#     # new_productions = new_cfg._get_productions_with_only_single_terminals()
#     # new_productions = new_cfg._decompose_productions(new_productions)
#     # print("after new_productions")
#     #
#     # cfg_in_wcnf = CFG(start_symbol=cfg._start_symbol, productions=set(new_productions))
#     # print("before return")
#     # return cfg_in_wcnf
#
#     # if not isinstance(cfg, CFG):
#     #     cfg = read_cfg(cfg, starting if starting is not None else "S")
#     print("in cfg_to_wcnf")
#     if isinstance(cfg, str):
#         cfg = CFG.from_text(cfg, Variable(starting))
#     if not isinstance(cfg, CFG):
#         print("SHIT!")
#     print("after isinstance")
#     wcnf_cfg = cfg.remove_useless_symbols()
#     print("after useless")
#     wcnf_cfg = wcnf_cfg.eliminate_unit_productions()
#     print("after unit")
#     wcnf_cfg = wcnf_cfg.remove_useless_symbols()
#     print("after useless")
#     wcnf_productions = wcnf_cfg._get_productions_with_only_single_terminals()
#     print("after p1")
#     wcnf_productions = wcnf_cfg._decompose_productions(wcnf_productions)
#     print("after p2")
#
#     return CFG(start_symbol=wcnf_cfg._start_symbol, productions=set(wcnf_productions))
#
#
# def read_cfg(path: PathLike, starting: str = "S") -> CFG:
#     with open(path, "r") as f:
#         data = f.read()
#     return CFG.from_text(data, Variable(starting))
#
#
def cyk(cfg: CFG, w: str) -> bool:
    # обрабатываем случай пустого слова
    if len(w) == 0:
        return cfg.generate_epsilon()

    # грамматика должна быть в НФХ
    cfg = cfg.to_normal_form()

    n = len(w)
    N = len(cfg.variables)
    M = np.zeros((n, n, N), dtype=bool)

    # продукции вида A -> a
    single_terminal_productions = {
        p for p in cfg.productions if p.body[0] in cfg.terminals
    }
    # продукции вида A -> BC
    var_to_vars_productions = {
        p
        for p in cfg.productions
        if p.body[0] in cfg.variables and p.body[1] in cfg.variables
    }

    # словарь, сопоставляющий переменным их порядковые номера
    var_indexes = {v: i for i, v in enumerate(cfg.variables)}

    def get_var_by_index(index):
        for v, i in var_indexes.items():
            if i == index:
                return v

    def get_variables_in_matrix_cell(i, j):
        variables = {get_var_by_index(i) for i, v in enumerate(M[i][j]) if v}
        return variables

    # 1 заполнить главную диагональ
    for i in range(n):
        for v in cfg.variables:
            # рассматриваем продукции для текущей переменной
            filtered_ps = {p for p in single_terminal_productions if p.head == v}
            # правая часть продукции это i-я буква в слове w
            M[i, i, var_indexes[v]] = any(
                p.body[0] == Terminal(w[i]) for p in filtered_ps
            )

    # 2 динамически заполняем наддиагонали
    for offset in range(1, n):
        for i in range(n - offset):
            j = i + offset

            for v in cfg.variables:
                filtered_ps = {p for p in var_to_vars_productions if p.head == v}
                bounds = [k for k in range(i, j)]

                M[i, j, var_indexes[v]] += any(
                    M[i, k, var_indexes[p.body[0]]]
                    and M[k + 1, j, var_indexes[p.body[1]]]
                    for k in bounds
                    for p in filtered_ps
                )

    return cfg.start_symbol in get_variables_in_matrix_cell(0, n - 1)
