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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.core.expr.compare import compare_expressions
from pyomo.contrib.mpc.interfaces.model_interface import DynamicModelInterface
from pyomo.contrib.mpc.data.series_data import TimeSeriesData
from pyomo.contrib.mpc.data.scalar_data import ScalarData


class TestDynamicModelInterface(unittest.TestCase):

    def _make_model(self, n_time_points=3):
        m = pyo.ConcreteModel()
        m.time = pyo.Set(initialize=range(n_time_points))
        m.comp = pyo.Set(initialize=["A", "B"])
        m.var = pyo.Var(
            m.time,
            m.comp,
            initialize={(i, j): 1.0 + i*0.1 for i, j in m.time*m.comp},
        )
        m.input = pyo.Var(
            m.time,
            initialize={i: 1.0 - i*0.1 for i in m.time},
        )
        m.scalar = pyo.Var(initialize=0.5)
        m.var_squared = pyo.Expression(
            m.time,
            m.comp,
            initialize={(i, j): m.var[i, j]**2 for i, j in m.time*m.comp},
        )
        return m

    def _hashRef(self, reference):
        return tuple(id(obj) for obj in reference.values())

    def test_interface_construct(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)

        scalar_vars = interface.get_scalar_variables()
        self.assertEqual(len(scalar_vars), 1)
        self.assertIs(scalar_vars[0], m.scalar)

        dae_vars = interface.get_indexed_variables()
        self.assertEqual(len(dae_vars), 3)
        dae_var_set = set(self._hashRef(var) for var in dae_vars)
        pred_dae_var = [
            pyo.Reference(m.var[:, "A"]),
            pyo.Reference(m.var[:, "B"]),
            m.input,
        ]
        for var in pred_dae_var:
            self.assertIn(self._hashRef(var), dae_var_set)

        dae_expr = interface.get_indexed_expressions()
        dae_expr_set = set(self._hashRef(expr) for expr in dae_expr)
        self.assertEqual(len(dae_expr), 2)
        pred_dae_expr = [
            pyo.Reference(m.var_squared[:, "A"]),
            pyo.Reference(m.var_squared[:, "B"]),
        ]
        for expr in pred_dae_expr:
            self.assertIn(self._hashRef(expr), dae_expr_set)

    def test_get_scalar_var_data(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_scalar_variable_data()
        self.assertEqual(
            {pyo.ComponentUID(m.scalar): 0.5},
            data,
        )

    def test_get_data_at_time_all_points(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_data_at_time(include_expr=True)
        pred_data = TimeSeriesData(
            {
                m.var[:, "A"]: [1.0, 1.1, 1.2],
                m.var[:, "B"]: [1.0, 1.1, 1.2],
                m.input[:]: [1.0, 0.9, 0.8],
                m.var_squared[:, "A"]: [1.0**2, 1.1**2, 1.2**2],
                m.var_squared[:, "B"]: [1.0**2, 1.1**2, 1.2**2],
            },
            m.time,
        )
        self.assertEqual(data, pred_data)

    def test_get_data_at_time_subset(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_data_at_time(time=[0, 2])
        pred_data = TimeSeriesData(
            {
                m.var[:, "A"]: [1.0, 1.2],
                m.var[:, "B"]: [1.0, 1.2],
                m.input[:]: [1.0, 0.8],
            },
            [0, 2],
        )
        self.assertEqual(data, pred_data)

    def test_get_data_at_time_singleton(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = interface.get_data_at_time(time=1, include_expr=True)
        pred_data = ScalarData({
            m.var[:, "A"]: 1.1,
            m.var[:, "B"]: 1.1,
            m.input[:]: 0.9,
            m.var_squared[:, "A"]: 1.1**2,
            m.var_squared[:, "B"]: 1.1**2,
        })
        self.assertEqual(data, pred_data)

    def test_load_scalar_data(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = {pyo.ComponentUID(m.scalar): 6.0}
        interface.load_scalar_data(data)
        self.assertEqual(m.scalar.value, 6.0)

    def test_load_data_at_time_all(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        interface.load_data_at_time(data)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 5.5)
            self.assertEqual(m.input[t].value, 6.6)

    def test_load_data_at_time_subset(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)

        old_A = {t: m.var[t, "A"].value for t in m.time}
        old_input = {t: m.input[t].value for t in m.time}

        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        time_points = [1, 2]
        time_set = set(time_points)
        interface.load_data_at_time(data, time_points=[1, 2])

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            if t in time_set:
                self.assertEqual(m.var[t, "A"].value, 5.5)
                self.assertEqual(m.input[t].value, 6.6)
            else:
                self.assertEqual(m.var[t, "A"].value, old_A[t])
                self.assertEqual(m.input[t].value, old_input[t])

    def test_load_data_from_dict_scalar_var(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = {pyo.ComponentUID(m.scalar): 6.0}
        interface.load_data(data)
        self.assertEqual(m.scalar.value, 6.0)

    def test_load_data_from_dict_indexed_var(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = {pyo.ComponentUID(m.input): 6.0}
        interface.load_data(data)
        for t in m.time:
            self.assertEqual(m.input[t].value, 6.0)

    def test_load_data_from_dict_indexed_var_list_data(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data_list = [2, 3, 4]
        data = {pyo.ComponentUID(m.input): data_list}
        interface.load_data(data)
        for i, t in enumerate(m.time):
            self.assertEqual(m.input[t].value, data_list[i])

    def test_load_data_from_ScalarData_toall(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        interface.load_data(data)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 5.5)
            self.assertEqual(m.input[t].value, 6.6)

    def test_load_data_from_ScalarData_tosubset(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)

        old_A = {t: m.var[t, "A"].value for t in m.time}
        old_input = {t: m.input[t].value for t in m.time}

        data = ScalarData({m.var[:, "A"]: 5.5, m.input[:]: 6.6})
        time_points = [1, 2]
        time_set = set(time_points)
        interface.load_data(data, time_points=[1, 2])

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for t in m.time:
            if t in time_set:
                self.assertEqual(m.var[t, "A"].value, 5.5)
                self.assertEqual(m.input[t].value, 6.6)
            else:
                self.assertEqual(m.var[t, "A"].value, old_A[t])
                self.assertEqual(m.input[t].value, old_input[t])

    def test_load_data_from_TimeSeriesData(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        new_A = [1.0, 2.0, 3.0]
        new_input = [4.0, 5.0, 6.0]
        data = TimeSeriesData(
            {m.var[:, "A"]: new_A, m.input[:]: new_input},
            m.time,
        )
        interface.load_data(data)

        B_data = [m.var[t, "B"].value for t in m.time]
        # var[:,B] has not been changed
        self.assertEqual(B_data, [1.0, 1.1, 1.2])

        for i, t in enumerate(m.time):
            self.assertEqual(m.var[t, "A"].value, new_A[i])
            self.assertEqual(m.input[t].value, new_input[i])

    def test_copy_values_at_time_default(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        interface.copy_values_at_time()
        # Default is to copy values from t0 to all points in time
        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 1.0)
            self.assertEqual(m.var[t, "B"].value, 1.0)
            self.assertEqual(m.input[t].value, 1.0)

    def test_copy_values_at_time_toall(self):
        m = self._make_model()
        tf = m.time.last()
        interface = DynamicModelInterface(m, m.time)
        interface.copy_values_at_time(source_time=tf)
        # Default is to copy values to all points in time
        for t in m.time:
            self.assertEqual(m.var[t, "A"].value, 1.2)
            self.assertEqual(m.var[t, "B"].value, 1.2)
            self.assertEqual(m.input[t].value, 0.8)

    def test_copy_values_at_time_tosubset(self):
        m = self._make_model()
        tf = m.time.last()
        interface = DynamicModelInterface(m, m.time)
        target_points = [t for t in m.time if t != m.time.first()]
        target_subset = set(target_points)
        interface.copy_values_at_time(
            source_time=tf, target_time=target_points
        )
        # Default is to copy values to all points in time
        for t in m.time:
            if t in target_subset:
                self.assertEqual(m.var[t, "A"].value, 1.2)
                self.assertEqual(m.var[t, "B"].value, 1.2)
                self.assertEqual(m.input[t].value, 0.8)
            else:
                # t0 has not been altered.
                self.assertEqual(m.var[t, "A"].value, 1.0)
                self.assertEqual(m.var[t, "B"].value, 1.0)
                self.assertEqual(m.input[t].value, 1.0)

    def test_copy_values_at_time_exception(self):
        m = self._make_model()
        tf = m.time.last()
        interface = DynamicModelInterface(m, m.time)
        msg = "copy_values_at_time can only copy"
        with self.assertRaisesRegex(ValueError, msg):
            interface.copy_values_at_time(
                source_time=m.time, target_time=tf
            )

    def test_shift_values_by_time(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        dt = 1.0
        interface.shift_values_by_time(dt)

        t = 0
        self.assertEqual(m.var[t, "A"].value, 1.1)
        self.assertEqual(m.var[t, "B"].value, 1.1)
        self.assertEqual(m.input[t].value, 0.9)

        t = 1
        self.assertEqual(m.var[t, "A"].value, 1.2)
        self.assertEqual(m.var[t, "B"].value, 1.2)
        self.assertEqual(m.input[t].value, 0.8)

        t = 2
        # For values within dt of the endpoint, the value at
        # the boundary is copied.
        self.assertEqual(m.var[t, "A"].value, 1.2)
        self.assertEqual(m.var[t, "B"].value, 1.2)
        self.assertEqual(m.input[t].value, 0.8)

    def test_get_tracking_cost_from_constant_setpoint(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        setpoint_data = ScalarData({
            m.var[:, "A"]: 1.0,
            m.var[:, "B"]: 2.0,
        })
        weight_data = ScalarData({
            m.var[:, "A"]: 10.0,
            m.var[:, "B"]: 0.1,
        })

        m.tracking_cost = interface.get_tracking_cost_from_constant_setpoint(
            setpoint_data,
            weight_data=weight_data,
        )
        for t in m.time:
            pred_expr = (
                10.0*(m.var[t, "A"] - 1.0)**2
                + 0.1*(m.var[t, "B"] - 2.0)**2
            )
            self.assertEqual(
                pyo.value(pred_expr),
                pyo.value(m.tracking_cost[t]),
            )
            self.assertTrue(compare_expressions(
                pred_expr,
                m.tracking_cost[t].expr,
            ))

    def test_get_tracking_cost_from_constant_setpoint_var_subset(self):
        m = self._make_model()
        interface = DynamicModelInterface(m, m.time)
        setpoint_data = ScalarData({
            m.var[:, "A"]: 1.0,
            m.var[:, "B"]: 2.0,
            m.input[:]: 3.0,
        })
        weight_data = ScalarData({
            m.var[:, "A"]: 10.0,
            m.var[:, "B"]: 0.1,
            m.input[:]: 0.01,
        })

        m.tracking_cost = interface.get_tracking_cost_from_constant_setpoint(
            setpoint_data,
            variables=[m.var[:, "A"], m.var[:, "B"]],
            weight_data=weight_data,
        )
        for t in m.time:
            pred_expr = (
                10.0*(m.var[t, "A"] - 1.0)**2
                + 0.1*(m.var[t, "B"] - 2.0)**2
            )
            self.assertEqual(
                pyo.value(pred_expr),
                pyo.value(m.tracking_cost[t]),
            )
            self.assertTrue(compare_expressions(
                pred_expr,
                m.tracking_cost[t].expr,
            ))


if __name__ == "__main__":
    unittest.main()