import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

ts_universe = np.arange(0, 100, 0.001)
dur_universe = np.arange(0, 5, 0.001)
l_universe = np.arange(-2.0, 2.0, 0.001)

# print(ts_universe)

level = ctrl.Antecedent(ts_universe, 'level')
duration = ctrl.Antecedent(dur_universe, 'duration')
l_parameter = ctrl.Consequent(l_universe, 'l_parameter')

# # Fuzzy Membership Functions
level['Low']=fuzz.trapmf(level.universe,[0,0,4,13])
level['Medium']=fuzz.trapmf(level.universe,[10,26,34,46])
level['High']=fuzz.trapmf(level.universe,[44,66,100,100])


duration['Minimum']=fuzz.trapmf(duration.universe,[0,0,0.5,2])
duration['Average']=fuzz.trapmf(duration.universe,[0.75,2.0,2.25,3])
duration['Maximum']=fuzz.trapmf(duration.universe,[2.5,4.5,5,5])

l_parameter['Low'] =fuzz.trapmf(l_parameter.universe,[-2,-2,-1.4,-0.4])
l_parameter['Medium'] =fuzz.trapmf(l_parameter.universe,[-0.6,-0.4,0.4,0.6])
l_parameter['High'] =fuzz.trapmf(l_parameter.universe,[0.2,1.0,2,2])

level.view()
l_parameter.view()
# duration.view()

# RULES

rule1 = ctrl.Rule(level['Low'] | duration['Minimum'], l_parameter['Low'])
rule2 = ctrl.Rule(level['Medium'] | duration['Average'], l_parameter['Medium'])
rule3 = ctrl.Rule(level['High'] | duration['Maximum'], l_parameter['High'])

# CS

level_inf_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
level_output = ctrl.ControlSystemSimulation(level_inf_ctrl)

# # Example
# level_output.input['level'] = -0.6
# level_output.input['duration'] = 4
#
# # Crunch the numbers
# level_output.compute()
#
# print(level_output.output['l_parameter'])
