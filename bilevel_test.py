# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:39:58 2020

@author: Luke
"""

from pyomo.environ import *
from pyomo.bilevel import *
from pyomo.opt import SolverFactory

model = ConcreteModel()
model.x = Var(bounds=(1,2))
model.v = Var(bounds=(1,2))
model.sub = SubModel()
model.sub.y = Var(bounds=(1,2))
model.sub.w = Var(bounds=(-1,1))
model.o = Objective(expr=model.x + model.sub.y + model.v)
model.c = Constraint(expr=model.x + model.v >= 1.5)
model.sub.o = Objective(expr=model.x+model.sub.w, sense=maximize)
model.sub.c = Constraint(expr=model.sub.y + model.sub.w <= 2.5)

xfrm = TransformationFactory('bilevel.linear_mpec')
xfrm.apply_to(model)


solver = SolverFactory("cplex")
solution = solver.solve(model, tee=True, keepfiles=False)