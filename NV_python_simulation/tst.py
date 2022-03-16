import numpy as np
import sympy as sp


def rad2deg(angle):
    return sp.sin(angle * sp.pi / 180), sp.cos(angle * sp.pi / 180)


rad2deg2 = lambda angle: (sp.sin(angle * sp.pi / 180), sp.cos(angle * sp.pi / 180))

print(rad2deg(20))
print(rad2deg2(20))