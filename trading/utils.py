import numpy as np
from gurobipy import *


def long_short_position_optimizer(contract_val, portfolio_val_lim,
                                  start_values=None):
    """Optimize number of contracts bought/sold.

    Parameters
    ----------
    contract_val : numpy.ndarray
    portfolio_val_lim : tuple
        of min, max values
    start_values : numpy.ndarray
        default is 1

    Returns
    -------
    contract_num : numpy.ndarray
        number of contracts
    pf_val : numpy.ndarray
        resulting portfolio value

    """
    # number of assets
    n = len(contract_val)

    # start at 1 if not specified
    if start_values is None:
        start_values = np.ones(shape=(n,))

    # init model
    m = Model("pf_opt")
    m.Params.OutputFlag = 0

    # add variables, type integer, lower bound = 1
    vs = m.addVars(n, lb=1, vtype=GRB.INTEGER)
    for p, start_v in enumerate(start_values):
        vs[p].start = start_v

    # objective
    obj_fun = np.var([contract_val[p] * vs[p] for p in range(n)])
    m.setObjective(obj_fun, GRB.MINIMIZE)

    # constraints
    # 1) minimum portfolio value
    m.addConstr(
        lhs=quicksum(vs[p] * contract_val[p] for p in range(n)),
        sense=GRB.GREATER_EQUAL,
        rhs=portfolio_val_lim[0],
        name="min_portfolio_value"
    )

    # 2) maximum portfolio value
    m.addConstr(
        lhs=quicksum(vs[p] * contract_val[p] for p in range(n)),
        sense=GRB.LESS_EQUAL,
        rhs=portfolio_val_lim[1],
        name="max_portfolio_value"
    )

    # optimize
    m.optimize()

    # retrieve optimal values
    contract_num = np.array([v.x for v in m.getVars()])

    # calculate corresponding portfolio value
    pf_val = contract_num.dot(contract_val)

    return contract_num, pf_val
