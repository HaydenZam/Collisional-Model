import numpy as np
import matplotlib.pyplot as plt

from qutip import basis , qeye , tensor , Qobj , mesolve , expect, tracedist, liouvillian, spre, spost

#============================================================

# GENERIC N - SITE CHANNEL : COHERENT SENDER AND RECIEVER
# Collision model tensor order : S $ \otimes $ C1 $ \otimes $
# Master equation tensor order : C1 $ \otimes $ ... $ \otimes $
# Convention : |0 > = ground , |1 > = excited
#============================================================

# Choose N
N = 2

#==========================================================
# Basis and single - qubit operators
#============================================================

ket0 = basis(2 , 0)
ket1 = basis(2 , 1)

I = qeye(2)

sp = ket1 * ket0.dag()
sm = ket0 * ket1.dag()

# sigma ^+ = |1 > <0|
# sigma ^ - = |0 > <1|

n_op = ket1 * ket1.dag()
sz = ket1 * ket1.dag() - ket0 * ket0.dag()

#============================================================
# Tensor operators
#============================================================

def op_col( site , op ) :
    """
    Operator on full collision Hilbert space :
    indices : 0 = S , 1... N = channel sites , N +1 = R .
    """
    ops = [ I for _ in range( N + 2) ]
    ops [ site ] = op
    return tensor(* ops )

def op_channel( site , op ) :
    """
    Operator on channel Hilbert space :
    indices : 0... N -1 = channel sites .
    """
    ops = [ I for _ in range(N) ]
    ops [ site ] = op
    return tensor(* ops )

# Full - space labels
S_idx = 0
R_idx = N + 1

# Channel labels in full collision space :
# channel site j = 0 ,... , N -1 corresponds to full index j +1

#============================================================
# Parameters
#============================================================

# chain magnetic fields
omegas = [0.0 for _ in range(N) ]

# ancilla magnetic fields
omega_S = 0
omega_R = 0

# unscaled left and right coupling constants
gL_eff = 0.20
gR_eff = 0.20

# collision time and number of collisions
tau = 0.001
n_cols = 10000
T_total = tau * n_cols

# Weak - collision scaling
Js = [0.3 for _ in range(N-1) ]
gL_col = gL_eff / np.sqrt( tau )
gR_col = gR_eff / np.sqrt( tau )

#============================================================
# Coherent sender
#============================================================

p_sender = 0.7

# Finite coherent amplitude entering H_drive
c_sender = complex(0.4,0.3)
# Actual sender coherence used in collision model
c_sender_col = np.sqrt( tau ) * c_sender

# positivity check
if abs( c_sender_col  ) ** 2 > p_sender *(1 - p_sender ) :
    raise ValueError( " Sender density matrix is not positive .")

rho_S = Qobj(
    [[1 - p_sender , c_sender_col  ] ,
    [ np.conjugate( c_sender_col  ) , p_sender ]] ,
    dims =[[2] , [2]]
)

#============================================================
# Coherent reciever
#============================================================

p_reciever = 0.4

# Finite coherent amplitude entering H_drive
c_reciever = complex(0.3,0.5)
# Actual sender coherence used in collision model
c_reciever_col  = np.sqrt( tau ) * c_reciever

# positivity check
if abs( c_reciever_col  ) ** 2 > p_reciever *(1 - p_reciever ) :
    raise ValueError( " receiver density matrix is not positive .")

rho_R = Qobj(
    [[1 - p_reciever , c_reciever_col  ] ,
    [ np.conjugate( c_reciever_col) , p_reciever ]] ,
    dims =[[2] , [2]]
)

#chain is in initial comp zero state
rho_C0 = tensor(*[ ket0 * ket0.dag() for _ in range(N) ])


#============================================================
# Exact collision model Hamiltonian
#============================================================

H_C_tot = 0

# Local channel energies
for j in range(N) :
    H_C_tot += omegas [ j ] * op_col( j + 1 , n_op )

# Internal hopping
for j in range(N-1) :
    H_C_tot += Js [ j ] *(
        op_col( j + 1 , sp ) * op_col( j + 2 , sm )
        + op_col( j + 1 , sm ) * op_col( j + 2 , sp )
    )

H_S_tot = omega_S * op_col( S_idx , n_op )
H_R_tot = omega_R * op_col( R_idx , n_op )

# Sender couples to site 1
V_L = gL_col *(
    op_col( S_idx , sm ) * op_col(1 , sp )
    + op_col( S_idx , sp ) * op_col(1 , sm )
)

# Receiver couples to site N
V_R = gR_col *(
    op_col(N , sm ) * op_col( R_idx , sp )
    + op_col(N , sp ) * op_col( R_idx , sm )
)

H_tot = H_C_tot + H_S_tot + H_R_tot + V_L + V_R

U = ( -1*complex(0,1) * H_tot * tau ).expm()

#============================================================
# Exact collision dynamics
#============================================================

rho_C = rho_C0

times_col = [0.0]

Rho_C_at_each_step = [rho_C]
for step in range( n_cols ) :
    rho_tot = tensor( rho_S , rho_C , rho_R )
    rho_tot_after = U * rho_tot * U.dag()

    # Keep channel indices 1... N in full collision space
    rho_C = rho_tot_after.ptrace( list( range(1 , N + 1) ) )

    Rho_C_at_each_step.append(rho_C)
    times_col.append(( step + 1) * tau )

#============================================================
# Master Equation Hamiltonian
#============================================================

H_channel = 0

############# First order terms #############
# Local energies
for j in range(N) :
    H_channel += omegas [ j ] * op_channel(j , n_op )

# Internal hopping
for j in range(N-1) :
    H_channel += Js [ j ] *(
        op_channel(j , sp ) * op_channel( j + 1 , sm )
        + op_channel(j , sm ) * op_channel( j + 1 , sp )
    )

# Coherent drive on both sites
H_drive = gL_col *(
    np.conjugate( c_sender_col ) * op_channel(0 , sp )
    + c_sender_col * op_channel(0 , sm )
) + gR_col *(
    np.conjugate( c_reciever_col ) * op_channel(N-1, sp )
    + c_reciever_col * op_channel(N-1, sm )
)

H_eff = H_channel + H_drive

coherant_liouvillian = -complex(0,1) * (spre(H_eff) - spost(H_eff))

############# Second order terms #############

#local rates
Gamma_L_plus =  tau*(gL_col ** 2 * p_sender)
Gamma_L_minus = tau*(gL_col ** 2 *(1 - p_sender ))
Gamma_R_plus =  tau*(gR_col ** 2 * p_reciever)
Gamma_R_minus = tau*(gR_col ** 2 *(1 - p_reciever ))

# local dissapators
def local_D(op1):
    return spre(op1)*spost(op1.dag()) - 0.5*(spre(op1.dag() * op1) + spost(op1.dag() * op1))

L1 = Gamma_L_plus * local_D(op_channel(0 , sp ))
L2 = Gamma_L_minus * local_D(op_channel(0 , sm ))
L3 = Gamma_R_plus * local_D(op_channel( N-1 , sp ))
L4 = Gamma_R_minus * local_D(op_channel( N-1 , sm ))

# non-local rates
Gamma_LR_plus_plus =   tau*(gR_col*gL_col*np.conjugate(c_sender_col)*np.conjugate(c_reciever_col))
Gamma_LR_plus_minus =  tau*(gR_col*gL_col*np.conjugate(c_sender_col)*c_reciever_col)
Gamma_LR_minus_plus =  tau*(gR_col*gL_col*c_sender_col*np.conjugate(c_reciever_col))
Gamma_LR_minus_minus = tau*(gR_col*gL_col*c_sender_col*c_reciever_col)

# non-local dissapators
def non_local_D(op1, op2):
    return spre(op1)*spost(op2) - 0.5*(spre(op2 * op1) + spost(op2 * op1))

sp1 = op_channel(0 , sp)
sm1 = op_channel(0 , sm)
spN = op_channel(N-1 , sp)
smN = op_channel(N-1 , sm)

NL1 = Gamma_LR_plus_plus*(non_local_D(sp1, spN) + non_local_D(spN, sp1))
NL2 = Gamma_LR_plus_minus*(non_local_D(sp1, smN) + non_local_D(smN, sp1))
NL4 = Gamma_LR_minus_plus*(non_local_D(sm1, spN) + non_local_D(spN, sm1))
NL3 = Gamma_LR_minus_minus*(non_local_D(sm1, smN) + non_local_D(smN, sm1))

# Total Liouvillian
L_total = (coherant_liouvillian     # Coherant Terms
           + L1 + L2 + L3 + L4      # Local dissapation
           + NL1 + NL2 + NL3 + NL4) # Non-Local dissapation 

times_me = np.linspace(0 , T_total , n_cols + 1)

result_states = mesolve(
    L_total ,
    rho_C0 ,
    times_me ,
    None ,
    e_ops = [] ,
)

trace_dists = [tracedist(result_states.states[i], Rho_C_at_each_step[i]) for i in range(len(result_states.states))]


#============================================================
# Plot comparison : Trace Distance
#============================================================

plt.figure( figsize =(10 , 6) )

plt.plot(
    times_col ,
    trace_dists ,
    lw =2 )

plt.xlabel( " time " )
plt.ylabel( "Trace Distance" )
plt.title( f" { N } - site channel : ps-{p_sender} cs-{c_sender} pr-{p_reciever} cr-{c_reciever} " )
plt.legend( fontsize =8)
plt.tight_layout()

plt.savefig(f"Collision Model/Plots/Separable/M_vs_E_ps-{p_sender}_cs-{c_sender}_pr-{p_reciever}_cr-{c_reciever}.jpeg", dpi=100)
plt.show()