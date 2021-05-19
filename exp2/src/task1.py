from autograd import grad
from scipy.optimize import fsolve
from scipy.optimize import minimize


def objective(X):
    x, y, z = X
    return x**2 + y**2 + z**2


def eq(X):
    x, y, z = X
    return 2 * x - y + z - 3


sol = minimize(objective, [1, -0.5, 0.5],
               constraints={'type': 'eq', 'fun': eq})

print(sol)


def F(L):
    'Augmented Lagrange function'
    x, y, z, lmd = L
    return objective([x, y, z]) - lmd * eq([x, y, z])


dfdL = grad(F, 0)  # Gradients of the Lagrange function


def obj(L):
    # Find L that returns all zeros in this function.
    x, y, z, _ = L
    dFdx, dFdy, dFdz, dFdlam = dfdL(L)
    return [dFdx, dFdy, dFdz, eq([x, y, z])]


x, y, z, _ = fsolve(obj, [0.0, 0.0, 0.0, 1.0])
print(f'The answer is at {x, y, z}')
