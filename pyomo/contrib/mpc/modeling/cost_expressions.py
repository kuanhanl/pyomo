#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

from pyomo.common.collections import ComponentMap
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.expression import Expression
from pyomo.core.base.param import Param
from pyomo.core.base.set import Set
from pyomo.core.base.var import Var
from pyomo.core.expr.logical_expr import EqualityExpression

from pyomo.contrib.mpc.data.series_data import get_indexed_cuid
from pyomo.contrib.mpc.data.scalar_data import ScalarData
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.interval_data import IntervalData
from pyomo.contrib.mpc.data.convert import (
    interval_to_series,
)


def get_parameters_from_variables(
    variables,
    time,
    ctype=Param,
):
    n_var = len(variables)
    init_dict = {
        (i, t): var[t].value for i, var in enumerate(variables) for t in time
    }
    var_set, comp = _get_indexed_parameters(
        n_var, time, ctype=ctype, initialize=init_dict
    )
    return var_set, comp


def _get_indexed_parameters(n, time, ctype=Param, initialize=None):
    range_set = Set(initialize=range(n))
    if ctype is Param:
        # Create a mutable parameter
        comp = ctype(range_set, time, mutable=True, initialize=initialize)
    elif ctype is Var:
        # Create a fixed variables
        comp = ctype(range_set, time, initialize=initialize)
        comp.fix()
    return range_set, comp


def get_tracking_cost_from_constant_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    """
    This function returns a tracking cost expression for the given
    time-indexed variables and associated setpoint data.

    Arguments
    ---------
    variables: list
        List of time-indexed variables to include in the tracking cost
        expression
    time: iterable
        Set by which to index the tracking expression
    setpoint_data: ScalarData, dict, or ComponentMap
        Maps variable names to setpoint values
    weight_data: ScalarData, dict, or ComponentMap
        Optional. Maps variable names to tracking cost weights. If not
        provided, weights of one are used.

    Returns
    -------
    Pyomo Expression, indexed by time, containing the sum of weighted
    squared difference between variables and setpoint values.

    """
    if weight_data is None:
        weight_data = ScalarData(ComponentMap((var, 1.0) for var in variables))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, ScalarData):
        setpoint_data = ScalarData(setpoint_data)

    # Make sure data have keys for each var
    for var in variables:
        if not setpoint_data.contains_key(var):
            raise KeyError(
                "Setpoint data dictionary does not contain a"
                " key for variable %s" % var.name
            )
        if not weight_data.contains_key(var):
            raise KeyError(
                "Tracking weight dictionary does not contain a"
                " key for variable %s" % var.name
            )

    # Set up data structures so we don't have to re-process keys for each
    # time index in the rule.
    cuids = [get_indexed_cuid(var) for var in variables]
    setpoint_data = setpoint_data.get_data()
    weight_data = weight_data.get_data()
    def tracking_rule(m, t):
        return sum(
            get_quadratic_tracking_cost_at_time(
                var, t, setpoint_data[cuid], weight=weight_data[cuid]
            )
            for cuid, var in zip(cuids, variables)
        )
    tracking_expr = Expression(time, rule=tracking_rule)
    return tracking_expr


def get_tracking_cost_from_piecewise_constant_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
    tolerance=0.0,
    prefer_left=True,
):
    """
    Arguments
    ---------
    variables: List of Pyomo variables
        Variables that participate in the cost expressions.
    time: Iterable
        Index used for the cost expression
    setpoint_data: IntervalData
        Holds the piecewise constant values that will be used as
        setpoints
    weight_data: ScalarData (optional)
        Weights for variables. Default is all ones.
    tolerance: Float (optional)
        Tolerance used for determining whether a time point
        is within an interval. Default is zero.
    prefer_left: Bool (optional)
        If a time point lies at the boundary of two intervals, whether
        the value on the left will be chosen. Default is True.

    Returns
    -------
    Expression
        Pyomo Expression, indexed by time, for the total weighted
        tracking cost with respect to the provided setpoint.

    """
    if isinstance(setpoint_data, IntervalData):
        setpoint_time_series = interval_to_series(
            setpoint_data,
            time_points=time,
            tolerance=tolerance,
            prefer_left=prefer_left,
        )
    else:
        setpoint_time_series = IntervalData(*setpoint_data)
    tracking_cost = get_tracking_cost_from_time_varying_setpoint(
        variables, time, setpoint_time_series, weight_data=weight_data
    )
    return tracking_cost


def get_quadratic_tracking_cost_at_time(var, t, setpoint, weight=None):
    if weight is None:
        weight = 1.0
    return weight * (var[t] - setpoint)**2


def _get_tracking_cost_expressions_from_time_varying_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    if weight_data is None:
        weight_data = ScalarData(ComponentMap((var, 1.0) for var in variables))
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    if not isinstance(setpoint_data, TimeSeriesData):
        setpoint_data = TimeSeriesData(*setpoint_data)

    # Validate incoming data
    if list(time) != setpoint_data.get_time_points():
        raise RuntimeError(
            "Mismatch in time points between time set and points"
            " in the setpoint data structure"
        )
    for var in variables:
        if not setpoint_data.contains_key(var):
            raise KeyError(
                "Setpoint data does not contain a key for variable"
                " %s" % var 
            )
        if not weight_data.contains_key(var):
            raise KeyError(
                "Tracking weight does not contain a key for"
                " variable %s" % var
            )

    # Get lists of weights and setpoints so we don't have to process
    # the variables (to get CUIDs) and hash the CUIDs for every
    # time index.
    cuids = [
        get_indexed_cuid(var, sets=(time,))
        for var in variables
    ]
    weights = [
        weight_data.get_data_from_key(var)
        for var in variables
    ]
    setpoints = [
        setpoint_data.get_data_from_key(var)
        for var in variables
    ]
    tracking_costs = [
        {
            t: get_quadratic_tracking_cost_at_time(
                var, t, setpoints[j][i], weights[j]
            ) for i, t in enumerate(time)
        } for j, var in enumerate(variables)
    ]
    return tracking_costs


def get_tracking_cost_from_time_varying_setpoint(
    variables,
    time,
    setpoint_data,
    weight_data=None,
):
    """
    Arguments
    ---------
    variables: List of Pyomo variables
        Variables that participate in the cost expressions.
    time: Iterable
        Index used for the cost expression
    setpoint_data: TimeSeriesData
        Holds the trajectory values that will be used as a setpoint
    weight_data: ScalarData (optional)
        Weights for variables. Default is all ones.

    Returns
    -------
    Expression
        Pyomo Expression, indexed by time, for the total weighted
        tracking cost with respect to the provided setpoint.

    """
    # This is a list of dictionaries, one for each variable and each
    # mapping each time point to the quadratic weighted tracking cost term
    # at that time point.
    tracking_costs = _get_tracking_cost_expressions_from_time_varying_setpoint(
        variables, time, setpoint_data, weight_data=weight_data
    )

    def tracking_rule(m, t):
        return sum(cost[t] for cost in tracking_costs)
    tracking_cost = Expression(time, rule=tracking_rule)
    return tracking_cost


def get_constraint_residual_expression(
    constraints,
    time,
    weight_data=None,
    # TODO: Option for norm (including no norm)
):
    if weight_data is None:
        weight_data = ScalarData(
            ComponentMap((var, 1.0) for con in constraints)
        )
    if not isinstance(weight_data, ScalarData):
        weight_data = ScalarData(weight_data)
    for con in constraints:
        if not weight_data.contains_key(con):
            raise KeyError(
                "Tracking weight does not contain a key for"
                " constraint %s" % con
            )
    n_con = len(constraints)
    con_set = Set(initialize=range(n_con))
    resid_expr_list = []
    for con in constraints:
        resid_expr_dict = {}
        for t in time:
            expr = con[t].expr
            if isinstance(expr, EqualityExpression):
                resid_expr_dict[t] = (con[t].body - con[t].upper)
            elif con.upper is None:
                resid_expr_dict[t] = (con[t].lower - con[t].body)
            elif con.lower is None:
                resid_expr_dict[t] = (con[t].body - con[t].upper)
            else:
                raise RuntimeError(
                    "Cannot construct a residual expression from a ranged"
                    " inequality. Error encountered processing the expression"
                    " of constraint %s" % con[t].name
                )
        resid_expr_list.append(resid_expr_dict)
    # NOTE: In KH's implementation, using error vars enforces that constraint
    # residuals are constant throughout a sampling period. Is this necessary?
    # Supposing that it is, we can achieve the same thing by imposing piecewise
    # constant constraints on these expressions.
    weights = [weight_data.get_data_from_key(con) for con in constraints]
    def resid_expr_rule(m, i, t):
        return weights[i]*resid_expr_list[i][t]**2
    resid_expr = Expression(con_set, time, rule=resid_expr_rule)
    return con_set, resid_expr
