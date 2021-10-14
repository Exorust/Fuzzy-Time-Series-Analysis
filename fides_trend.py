import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

ts_universe = np.arange(50, 100, 0.001)
dur_universe = np.arange(0, 5, 0.001)
t_universe = np.arange(-2.0, 2.0, 0.001)

# print(ts_universe)

trend = ctrl.Antecedent(t_universe, 'trend')
duration = ctrl.Antecedent(dur_universe, 'duration')
t_parameter = ctrl.Consequent(t_universe, 't_parameter')

# # Fuzzy Membership Functions
trend['Decrease'] = fuzz.trapmf(trend.universe,[ -2.0, -2.0, -1.4, -0.4])
trend['Stable'] = fuzz.trapmf(trend.universe,[-0.6,-0.4,0.4,0.6])
trend['Increase'] = fuzz.trapmf(trend.universe,[0.2,1.0,2.0,2.0])


duration['Minimum']=fuzz.trapmf(duration.universe,[0,0,0.5,2])
duration['Average']=fuzz.trapmf(duration.universe,[0.75,2.0,2.25,3])
duration['Maximum']=fuzz.trapmf(duration.universe,[2.5,4.5,5,5])

t_parameter['Low'] =fuzz.trapmf(t_parameter.universe,[-2,-2,-1.4,-0.4])
t_parameter['Medium'] =fuzz.trapmf(t_parameter.universe,[-0.6,-0.4,0.4,0.6])
t_parameter['High'] =fuzz.trapmf(t_parameter.universe,[0.2,1.0,2,2])

# trend.view()
# t_parameter.view()
duration.view()

# RULES

rule1 = ctrl.Rule(trend['Decrease'] | duration['Minimum'], t_parameter['Low'])
rule2 = ctrl.Rule(trend['Stable'] | duration['Average'], t_parameter['Medium'])
rule3 = ctrl.Rule(trend['Increase'] | duration['Maximum'], t_parameter['High'])

# CS

trend_inf_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
trend_output = ctrl.ControlSystemSimulation(trend_inf_ctrl)

# # Example
# trend_output.input['trend'] = -0.6
# trend_output.input['duration'] = 4
#
# # Crunch the numbers
# trend_output.compute()
#
# print(trend_output.output['t_parameter'])
