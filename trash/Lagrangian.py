"""
Lagrangian Dynamics — 3R Spatial Manipulator
=============================================
Computes symbolically:
  1. Forward kinematics (DH parameters)
  2. Link Jacobians Jv^ci and Jw^ci for each CoM
  3. Mass matrix M(q)
  4. Kinetic energy K(q, dq)
  5. Potential energy P(q)
  6. Gravity vector g(q) = dP/dq

Robot: 3-DOF revolute arm in 3D space
  Joint 1 — rotation about world z (vertical base rotation)
  Joint 2 — rotation about horizontal axis (shoulder lift)
  Joint 3 — rotation about horizontal axis (elbow)

DH Table (Craig convention):
  Link | a_i | alpha_i | d_i | theta_i
  -----|-----|---------|-----|--------
    1  |  0  |  pi/2   | d1  |   q1
    2  | l2  |    0    |  0  |   q2
    3  | l3  |    0    |  0  |   q3

pip install sympy
"""

import sympy as sp

# =============================================================================
# 1. Symbolic variables
# =============================================================================
q1, q2, q3     = sp.symbols('q1 q2 q3')
dq1, dq2, dq3  = sp.symbols('dq1 dq2 dq3')
q  = sp.Matrix([q1,  q2,  q3])
dq = sp.Matrix([dq1, dq2, dq3])

l2, l3, d1 = sp.symbols('l2 l3 d1', positive=True)   # geometry [m]
m1, m2, m3 = sp.symbols('m1 m2 m3', positive=True)   # link masses [kg]
g_acc       = sp.Symbol('g', positive=True)            # gravitational accel

# Diagonal inertia tensors in local link frames [kg·m²]
I1_local = sp.diag(*sp.symbols('Ixx1 Iyy1 Izz1', positive=True))
I2_local = sp.diag(*sp.symbols('Ixx2 Iyy2 Izz2', positive=True))
I3_local = sp.diag(*sp.symbols('Ixx3 Iyy3 Izz3', positive=True))

# =============================================================================
# 2. DH homogeneous transformation matrix T_{i-1,i}
# =============================================================================
def dh_matrix(a, alpha, d, theta):
    """
    Standard DH transform.  T = Rz(theta) · Tz(d) · Tx(a) · Rx(alpha)
    Columns: [x-axis | y-axis | z-axis | origin]
    """
    ct, st = sp.cos(theta), sp.sin(theta)
    ca, sa = sp.cos(alpha),  sp.sin(alpha)
    return sp.Matrix([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ])

# =============================================================================
# 3. Forward kinematics — build transformation chain
# =============================================================================
T01 = dh_matrix(0,  sp.pi/2, d1, q1)
T02 = sp.trigsimp(T01 * dh_matrix(l2, 0, 0, q2))
T03 = sp.trigsimp(T02 * dh_matrix(l3, 0, 0, q3))

# Extract z-axes (column 2) and origins (column 3) from each frame
T00 = sp.eye(4)
frames = [T00, T01, T02]                       # frames 0, 1, 2
z = [T[:3, 2] for T in frames]                 # z-axes  z0, z1, z2
o = [T[:3, 3] for T in frames]                 # origins o0, o1, o2
R  = [T01[:3, :3], T02[:3, :3], T03[:3, :3]]  # rotation matrices R1, R2, R3

# =============================================================================
# 4. Centre-of-mass positions (midpoint of each link)
#    Replace with actual CoM offsets from your URDF or datasheet.
# =============================================================================
o3   = T03[:3, 3]
p_c  = [
    sp.Rational(1, 2) * (o[0] + o[1]),   # CoM link 1
    sp.Rational(1, 2) * (o[1] + o[2]),   # CoM link 2
    sp.Rational(1, 2) * (o[2] + o3),     # CoM link 3
]

print("Centre-of-mass positions:")
for i, pc in enumerate(p_c, 1):
    print(f"  p_c{i} = {pc.T}")

# =============================================================================
# 5. Link Jacobians — the key building block for M(q)
#
#   For link i (1-indexed), only joints 1 … i affect its CoM.
#   Joints i+1 … n contribute zero columns.
#
#   Active column j (0-indexed):
#     Jv[:, j] = z_{j} × (p_ci − o_{j})   <- linear part
#     Jw[:, j] = z_{j}                      <- angular part
# =============================================================================
def link_jacobian(link_idx, p_ci, z_axes, o_frames, n=3):
    """
    Build 3×n linear and angular Jacobians for the CoM of link `link_idx`.
    link_idx is 1-indexed (1, 2, or 3 for a 3-DOF arm).
    """
    Jv = sp.zeros(3, n)
    Jw = sp.zeros(3, n)
    for j in range(link_idx):           # j = 0 … link_idx-1
        Jv[:, j] = z_axes[j].cross(p_ci - o_frames[j])
        Jw[:, j] = z_axes[j]
    return Jv, Jw

Jv1, Jw1 = link_jacobian(1, p_c[0], z, o)
Jv2, Jw2 = link_jacobian(2, p_c[1], z, o)
Jv3, Jw3 = link_jacobian(3, p_c[2], z, o)

print("\nLinear Jacobian Jv for link 2 (cols 0,1 active):")
sp.pprint(sp.trigsimp(Jv2))

# =============================================================================
# 6. Mass matrix M(q)
#
#   M(q) = Σ_i [ m_i Jv_i^T Jv_i  +  Jw_i^T (R_i I_i R_i^T) Jw_i ]
#
#   R_i I_i R_i^T  rotates the inertia tensor from local frame i
#   to the world frame before accumulating into M.
# =============================================================================
def M_link_contribution(mi, Jvi, Jwi, Ri, Ii_local):
    I_world = Ri * Ii_local * Ri.T          # inertia in world frame
    return mi * Jvi.T * Jvi + Jwi.T * I_world * Jwi

print("\nAssembling M(q)...")
M = sp.trigsimp(
    M_link_contribution(m1, Jv1, Jw1, R[0], I1_local) +
    M_link_contribution(m2, Jv2, Jw2, R[1], I2_local) +
    M_link_contribution(m3, Jv3, Jw3, R[2], I3_local)
)
print("M(q) shape:", M.shape, "(should be 3×3, symmetric)")

# =============================================================================
# 7. Kinetic energy K(q, dq)
#
#   K = (1/2) dq^T M(q) dq
# =============================================================================
K = sp.Rational(1, 2) * (dq.T * M * dq)[0, 0]
K = sp.expand(K)

# =============================================================================
# 8. Potential energy P(q) in 3D
#
#   Gravity vector: g0 = [0, 0, -g]^T  (z is up)
#   P = -Σ_i m_i g0^T p_ci  =  g * Σ_i m_i * z_ci(q)
#
#   Only the z-coordinate (height) of each CoM matters.
# =============================================================================
z_c = [pc[2] for pc in p_c]              # heights of each CoM

P = g_acc * sum(mi * zci for mi, zci in zip([m1, m2, m3], z_c))
P = sp.trigsimp(P)

print("\nPotential energy P(q):")
sp.pprint(P)

# =============================================================================
# 9. Gravity vector  g(q) = dP/dq
#    This is the gravitational torque vector in the equations of motion.
#    Note: g_vec[0] = 0 because joint 1 rotates about the vertical axis.
# =============================================================================
g_vec = sp.Matrix([sp.diff(P, qi) for qi in [q1, q2, q3]])
g_vec = sp.trigsimp(g_vec)

print("\nGravity vector g(q) = dP/dq:")
sp.pprint(g_vec)

# =============================================================================
# 10. Lagrangian L = K - P
#     Full equations of motion: M(q)ddq + C(q,dq)dq + g(q) = tau
# =============================================================================
L = K - P

# =============================================================================
# 11. Numerical evaluation at a sample configuration
# =============================================================================
print("\n" + "="*60)
print("Numerical evaluation — sample configuration")
print("="*60)

Ixx1,Iyy1,Izz1 = sp.symbols('Ixx1 Iyy1 Izz1', positive=True)
Ixx2,Iyy2,Izz2 = sp.symbols('Ixx2 Iyy2 Izz2', positive=True)
Ixx3,Iyy3,Izz3 = sp.symbols('Ixx3 Iyy3 Izz3', positive=True)

params = {
    q1: 0.3,  q2: 0.5,  q3: -0.4,       # joint angles   [rad]
    dq1: 0.1, dq2: -0.2, dq3: 0.15,     # joint velocities [rad/s]
    l2: 0.5,  l3: 0.4,  d1: 0.3,        # link geometry  [m]
    m1: 2.0,  m2: 1.5,  m3: 1.0,        # link masses    [kg]
    Ixx1: 0.010, Iyy1: 0.050, Izz1: 0.050,
    Ixx2: 0.008, Iyy2: 0.040, Izz2: 0.040,
    Ixx3: 0.005, Iyy3: 0.020, Izz3: 0.020,
    g_acc: 9.81,
}

M_num  = M.subs(params).evalf()
K_num  = float(K.subs(params).evalf())
P_num  = float(P.subs(params).evalf())
gv_num = g_vec.subs(params).evalf()

print(f"\nM(q) =\n{M_num}\n")
print(f"K = {K_num:.4f}  J")
print(f"P = {P_num:.4f}  J")
print(f"\ng(q) = {gv_num.T}")
print(f"\nNote: g_vec[0] = {float(gv_num[0]):.6f}  (zero — joint 1 is vertical)")