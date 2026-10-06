import numpy as np
import matplotlib.pyplot as plt

from qutip import basis , qeye , tensor , Qobj , mesolve , expect, tracedist

#============================================================

# GENERIC N - SITE CHANNEL : COHERENT SENDER
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

def op_collision( site , op ) :
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

omegas = [0.0 for _ in range(N) ]

# Nearest - neighbor hopping couplings J_j between site j and j +1
# length must be N -1

omega_S = 0
omega_R = 0

gL_eff = 0.20
gR_eff = 0.20

tau = 0.001
n_collisions = 10000
T_total = tau * n_collisions

# Weak - collision scaling
Js = [0.3 for _ in range(N-1) ]
gL_collision = gL_eff / np.sqrt( tau )
gR_collision = gR_eff / np.sqrt( tau )

#============================================================
# Coherent sender with finite - drive scaling
#============================================================

p_sender = 0.5

# Finite coherent amplitude entering H_drive
c_tilde = 0.5

# Actual sender coherence used in collision model
c_sender = np.sqrt( tau ) * c_tilde

if abs( c_sender ) ** 2 > p_sender *(1 - p_sender ) :
    raise ValueError( " Sender density matrix is not positive .")

rho_S = Qobj(
    [[1 - p_sender , c_sender ] ,
    [ np.conjugate( c_sender ) , p_sender ]] ,
    dims =[[2] , [2]]
)

rho_R = ket0 * ket0.dag()

rho_C0 = tensor(*[ ket0 * ket0.dag() for _ in range(N) ])

p = (ket1.dag() * rho_S * ket1)
c = (ket0.dag() * rho_S * ket1)

print( f" \n ===== N = { N } coherent sender ===== " )
print( " Sender excited population p = " , p )
print( " Actual sender coherence c = " , c )
print( " Scaled coherence c_tilde = c / sqrt( tau ) = " , c/np.sqrt( tau ) )

#============================================================
# Exact collision model Hamiltonian
#============================================================

H_C_tot = 0

# Local channel energies
for j in range(N) :
    H_C_tot += omegas [ j ] * op_collision( j + 1 , n_op )

# Internal hopping
for j in range(N-1) :
    H_C_tot += Js [ j ] *(
        op_collision( j + 1 , sp ) * op_collision( j + 2 , sm )
        + op_collision( j + 1 , sm ) * op_collision( j + 2 , sp )
    )

H_S_tot = omega_S * op_collision( S_idx , n_op )
H_R_tot = omega_R * op_collision( R_idx , n_op )

# Sender couples to site 1
V_L = gL_collision *(
    op_collision( S_idx , sm ) * op_collision(1 , sp )
    + op_collision( S_idx , sp ) * op_collision(1 , sm )
)

# Receiver couples to site N
V_R = gR_collision *(
    op_collision(N , sm ) * op_collision( R_idx , sp )
    + op_collision(N , sp ) * op_collision( R_idx , sm )
)

H_tot = H_C_tot + H_S_tot + H_R_tot + V_L + V_R

U = ( -1*complex(0,1) * H_tot * tau ).expm()

#============================================================
# Exact collision dynamics
#============================================================

rho_C = rho_C0

times_collision = [0.0]

sz_collision = [[] for _ in range(N) ]
n_collision = [[] for _ in range(N) ]

for j in range(N) :
    sz_collision [ j ]. append( expect( op_channel(j , sz ) , rho_C ) )
    n_collision [ j ]. append( expect( op_channel(j , n_op ) , rho_C )
)

Rho_C_at_each_step = [rho_C]
for step in range( n_collisions ) :
    rho_tot = tensor( rho_S , rho_C , rho_R )
    rho_tot_after = U * rho_tot * U.dag()

    # Keep channel indices 1... N in full collision space
    rho_C = rho_tot_after.ptrace( list( range(1 , N + 1) ) )

    Rho_C_at_each_step.append(rho_C)
    times_collision.append(( step + 1) * tau )

    for j in range(N) :
        sz_collision [ j ]. append( expect( op_channel(j , sz ) ,
        rho_C ) )
        n_collision [ j ]. append( expect( op_channel(j , n_op ) ,
        rho_C ) )

#============================================================
# Master Equation Hamiltonian
#============================================================

H_channel = 0

# Local energies
for j in range(N) :
    H_channel += omegas [ j ] * op_channel(j , n_op )

# Internal hopping
for j in range(N-1) :
    H_channel += Js [ j ] *(
        op_channel(j , sp ) * op_channel( j + 1 , sm )
        + op_channel(j , sm ) * op_channel( j + 1 , sp )
    )

# Coherent drive on first site
# Important : use c_tilde , not c_sender
H_drive = gL_eff *(
    c_tilde * op_channel(0 , sp )
    + np.conjugate( c_tilde ) * op_channel(0 , sm )
)

H_eff = H_channel + H_drive

Gamma_L_plus = gL_eff ** 2 * p
Gamma_L_minus = gL_eff ** 2 *(1 - p )
Gamma_R = gR_eff ** 2

c_ops = [
    np.sqrt( Gamma_L_plus ) * op_channel(0 , sp ) ,
    np.sqrt( Gamma_L_minus ) * op_channel(0 , sm ) ,
    np.sqrt( Gamma_R ) * op_channel( N - 1 , sm ) ,
]

times_me = np.linspace(0 , T_total , n_collisions + 1)

e_ops = []
for j in range(N) :
    e_ops.append( op_channel(j , sz ) )
for j in range(N) :
    e_ops.append( op_channel(j , n_op ) )

result = mesolve(
    H_eff ,
    rho_C0 ,
    times_me ,
    c_ops ,
    e_ops = e_ops ,
)

result_states = mesolve(
    H_eff ,
    rho_C0 ,
    times_me ,
    c_ops ,
    e_ops = [] ,
)

sz_me = []
n_me = []

for j in range(N) :
    sz_me.append( np.real( np.array( result.expect [ j ] , dtype =complex ) ) )

for j in range(N) :
    n_me.append( np.real( np.array( result.expect [ N + j ] , dtype= complex ) ) )

sz_collision = [
    np.real( np.array( sz_collision [ j ] , dtype = complex ) )
    for j in range(N)
]

n_collision = [
    np.real( np.array( n_collision [ j ] , dtype = complex ) )
    for j in range(N)
]

print(result_states.states[0])
print(Rho_C_at_each_step[0])

print("\n")
print(result_states.states[1])
print(Rho_C_at_each_step[1])

trace_dists = [tracedist(result_states.states[i], Rho_C_at_each_step[i]) for i in range(len(result_states.states))]

#============================================================
# Error report
#============================================================

def error_report( name , exact , approx ) :

    diff = exact - approx

    mse = np.mean( diff ** 2)
    rmse = np.sqrt( mse )
    max_err = np.max( np.abs( diff ) )

    signal_range = np.max( exact ) - np.min( exact )

    if signal_range > 10**(-12):
        rmse_percent = 100 * rmse / signal_range
        max_percent = 100 * max_err / signal_range
    else :
        rmse_percent = np.nan
        max_percent = np.nan

    print( f" \n { name }: " )
    print( " MSE=",mse )
    print( " RMSE=",rmse )
    print( " Max err=",max_err )
    print( " RMSE percentage=",rmse_percent , " % " )
    print( " Max error percentage=",max_percent , " % " )

    return diff

diff_sz = []
diff_n = []

for j in range(N) :
    diff_sz.append(
        error_report( f" < sigma_z ^{ j +1} > " , sz_collision [ j ] ,
        sz_me [ j ])
    )

for j in range(N) :
    diff_n.append(
        error_report( f" < n_ { j +1} > " , n_collision [ j ] , n_me [ j ])
    )

#============================================================
# Plot comparison : sigma_z
#============================================================

plt.figure( figsize =(10 , 6) )

for j in range(N) :
    plt.plot(
        times_collision ,
        sz_collision [ j ] ,
        lw =2 ,
        label = rf" Collision $ \langle \sigma_z ^{ j +1}\rangle$ ")
    plt.plot(
        times_me ,
        sz_me [ j ] ,
        "--" ,
        lw =2 ,
        label = rf"ME $ \langle \sigma_z ^{j +1}\rangle$ ")

plt.xlabel( " time " )
plt.ylabel( r"$ \langle \sigma_z \rangle$ " )
plt.title( f" { N } - site channel : coherent sender " )
plt.legend( fontsize =8)
plt.tight_layout()
plt.show()

#============================================================
# Plot comparison : populations
#============================================================

plt.figure( figsize =(10 , 6) )

for j in range(N) :
    plt.plot(
        times_collision ,
        n_collision [ j ] ,
        lw =2 ,
        label = rf"Collision $ \langle n_ {j +1}\rangle$ "
    )
    plt.plot(
        times_me ,
        n_me [ j ] ,
        "--" ,
        lw =2 ,
        label = rf"ME $ \langle n_ {j +1}\rangle$ "
    )

plt.xlabel( " time " )
plt.ylabel( " population " )
plt.title( f"{ N } - site channel populations : coherent sender ")
plt.legend( fontsize =8)
plt.tight_layout()
plt.show()

#============================================================
# Plot differences : sigma_z
#============================================================

plt.figure( figsize =(10 , 6) )

for j in range(N) :
    plt.plot(
        times_me ,
        diff_sz [ j ] ,
        lw =2 ,
        label = rf"$ \Delta \langle \sigma_z ^{ j +1}\rangle$ ")

plt.axhline(0.0 , linestyle = "--" , linewidth =1)
plt.xlabel( " time " )
plt.ylabel( " Collision model - Master equation " )
plt.title( f"{ N } - site channel : sigma_z differences ")
plt.legend( fontsize =8)
plt.tight_layout()
plt.show()

#============================================================
# Plot differences : populations
#============================================================

plt.figure( figsize =(10 , 6) )

for j in range(N) :
    plt.plot(
        times_me ,
        diff_n [ j ] ,
        lw =2 ,
        label = rf" $ \Delta \langle n_ { j +1}\rangle$ "
    )

plt.axhline(0.0 , linestyle = "--" , linewidth =1)
plt.xlabel( " time " )
plt.ylabel( " Collision model - Master equation " )
plt.title( f" { N } - site channel : population differences " )
plt.legend( fontsize =8)
plt.tight_layout()
plt.show()

#============================================================
# Plot comparison : Trace Distance
#============================================================

plt.figure( figsize =(10 , 6) )

plt.plot(
    times_collision ,
    trace_dists ,
    lw =2 )

plt.xlabel( " time " )
plt.ylabel( "Trace Distance" )
plt.title( f" { N } - site channel : coherent sender " )
plt.legend( fontsize =8)
plt.tight_layout()
plt.show()