import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

quality  = ctrl.Antecedent(np.arange(0,11,1), 'Quality')
service = ctrl.Antecedent(np.arange(0,11,1), 'Service')
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

quality.automf(3)
service.automf(3)

tip['low'] = fuzz.trimf(tip.universe,[0,0,13])
tip['medium'] = fuzz.trimf(tip.universe,[0,13,25])
tip['high']  = fuzz.trimf(tip.universe,[13,25,25])
quality['average'].view()
