import numpy as np
import matplotlib.pyplot as plt

from qutip import basis , qeye , tensor , Qobj , mesolve , expect, tracedist, liouvillian, spre, spost, concurrence

from random_two_qubit_state import random_two_qubit_state, collision_product_state, discord

from Unravel import kossakowski, gksl_ok, strict_unravellable



#============================================================
# Settings
#============================================================

# unscaled left and right coupling constants
gL = 1
gR =1 

# collision time
tau = 0.001

#============================================================
# Sender-Reciver Ancilla
#============================================================
rho_product = Qobj(random_two_qubit_state("product"), dims=[[2,2],[2,2]])
rho_product_collision = collision_product_state(tau, rho_product) # adds the necessary tau scaling

rho_classical = Qobj(random_two_qubit_state("classical"), dims=[[2,2],[2,2]])
rho_discord = Qobj(random_two_qubit_state("discord"), dims=[[2,2],[2,2]])
rho_entangled = Qobj(random_two_qubit_state("entangled"), dims=[[2,2],[2,2]])
rho_random = Qobj(random_two_qubit_state("random"), dims=[[2,2],[2,2]])

K = kossakowski(rho_product_collision, gL, gR)

print(K[0])
print(gksl_ok(K[0]))
print(strict_unravellable(K[0]))