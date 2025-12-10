import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

# Parameters from Table 1
g = 9.8
m = 0.486
Ix = 3.8278e-3
Iy = 3.8278e-3
Iz = 7.6566e-3
Jr = 2.8385e-5
rho_y = 2.9842e-3  # b, thrust coeff
rho_z = 3.2320e-2  # k, drag coeff
d = 0.225  # Arm length (typical value, not in paper)

K1_base = 0.012
K2_base = 0.012
K3_base = 0.012
K4_base = 0.01
K5_base = 0.01
K6_base = 0.01
noise_amp = 0.0  # Set to 0 for clean

M1 = (Iy - Iz) / Ix
M4 = (Iz - Ix) / Iy
M7 = (Ix - Iy) / Iz

N1 = 1 / Ix
N2 = 1 / Iy
N3 = 1 / Iz

# Control parameters from Table 2
lamb1 = 1.5  # Xi1
lamb2 = 1.5  # Reduced for stability (paper uses 0.5; you can revert if needed) 0.25 , 0.75 a7san mn 0.5 , 0.95 also better , 1.5 best

m_mu = 0.9 # mu_i for all

beta_pos = 2.1487  # Revert to paper
gamma_pos = 1.1  # Revert to paper
k1_pos = 6.0
k2_pos = 2.0

beta_att = 102.15
gamma_att = 1.9
k1_att = 817.6194
k2_att = 2.6997

# Smoothing epsilon increased
eps_pos = 0.05
eps_att = 1.0

# Desired trajectory for Simulation 2
def desired_pos(t):
    xdes = np.sin(0.5 * t)
    dxdes = 0.5 * np.cos(0.5 * t)
    ddxdes = -0.25 * np.sin(0.5 * t)
    ydes = np.cos(0.5 * t)
    dydes = -0.5 * np.sin(0.5 * t)
    ddydes = -0.25 * np.cos(0.5 * t)
    zdes = 0.1 * t + 2
    dzdes = 0.1
    ddzdes = 0.0
    return xdes, dxdes, ddxdes, ydes, dydes, ddydes, zdes, dzdes, ddzdes

def desired_psi(t):
    psides = 0.3
    dpsides = 0.0
    ddpsides = 0.0
    return psides, dpsides, ddpsides

# Disturbances for Simulation 2 (Eq 43)
def disturbances(t):
    # Dx = 1 + np.sin(0.2 * np.pi * t)
    # Dy = 1 + np.cos(0.2 * np.pi * t)
    # Dz = 0.5 * np.cos(0.7 * t) + 0.7 * np.sin(0.3 * t)
    # Dphi = 2 * np.sin(0.7 * t) + 1
    # Dtheta = 2 * np.cos(0.9 * t) + 1
    # Dpsi = 2 * np.tanh(0.7 * t)
    # return Dx, Dy, Dz, Dphi, Dtheta, Dpsi
    return 0,0,0,0,0,0


# Dynamics function
def dynamics(state, t):
    x, dx, y, dy, z, dz, phi, dphi, theta, dtheta, psi, dpsi, int_x, int_y, int_z, int_phi, int_theta, int_psi = state

    # Varying drag
    K1 = K1_base + np.random.normal(0, noise_amp)
    K2 = K2_base + np.random.normal(0, noise_amp)
    K3 = K3_base + np.random.normal(0, noise_amp)
    K4 = K4_base + np.random.normal(0, noise_amp)
    K5 = K5_base + np.random.normal(0, noise_amp)
    K6 = K6_base + np.random.normal(0, noise_amp)

    M3 = -K1 / Ix
    M6 = -K2 / Iy
    M8 = -K3 / Iz
    M9 = -K4 / m
    M10 = -K5 / m
    M11 = -K6 / m

    M2 = 0
    M5 = 0

    # Desired
    xdes, dxdes, ddxdes, ydes, dydes, ddydes, zdes, dzdes, ddzdes = desired_pos(t)
    psides, dpsides, ddpsides = desired_psi(t)

    # Position errors
    ex = x - xdes
    ey = y - ydes
    ez = z - zdes
    dex = dx - dxdes
    dey = dy - dydes
    dez = dz - dzdes

    # Position ITSM (Eq 9)
    s7 = lamb1 * ex + lamb2 * int_x
    ds7 = lamb1 * dex + lamb2 * np.abs(ex)**m_mu * np.sign(ex)
    s9 = lamb1 * ey + lamb2 * int_y
    ds9 = lamb1 * dey + lamb2 * np.abs(ey)**m_mu * np.sign(ey)
    s11 = lamb1 * ez + lamb2 * int_z
    ds11 = lamb1 * dez + lamb2 * np.abs(ez)**m_mu * np.sign(ez)

    # Clip ds
    abs_ds7 = np.clip(np.abs(ds7), 1e-10, 1e6)
    abs_ds9 = np.clip(np.abs(ds9), 1e-10, 1e6)
    abs_ds11 = np.clip(np.abs(ds11), 1e-10, 1e6)

    # Smooth sign
    sign_ds7 = ds7 / (abs_ds7 + eps_pos)
    sign_ds9 = ds9 / (abs_ds9 + eps_pos)
    sign_ds11 = ds11 / (abs_ds11 + eps_pos)
    sign_s7 = s7 / (np.abs(s7) + eps_pos)
    sign_s9 = s9 / (np.abs(s9) + eps_pos)
    sign_s11 = s11 / (np.abs(s11) + eps_pos)

    # NEW: Hyperplane variables (Eq 10)
    sigma7 = s7 + (1 / beta_pos) * sign_ds7 * (abs_ds7 ** gamma_pos)
    sigma9 = s9 + (1 / beta_pos) * sign_ds9 * (abs_ds9 ** gamma_pos)
    sigma11 = s11 + (1 / beta_pos) * sign_ds11 * (abs_ds11 ** gamma_pos)

    # Smooth sign for sigma
    sign_sigma7 = sigma7 / (np.abs(sigma7) + eps_pos)
    sign_sigma9 = sigma9 / (np.abs(sigma9) + eps_pos)
    sign_sigma11 = sigma11 / (np.abs(sigma11) + eps_pos)

    # Position continuous (Eq 17)
    term_x = m_mu * lamb2**2 / lamb1 * np.abs(ex)**(2*m_mu - 1) * np.sign(ex)
    vxc = ddxdes - M9 * dx + 1 / lamb1 * (term_x - beta_pos / gamma_pos * sign_ds7 * abs_ds7**(2 - gamma_pos))
    term_y = m_mu * lamb2**2 / lamb1 * np.abs(ey)**(2*m_mu - 1) * np.sign(ey)
    vyc = ddydes - M10 * dy + 1 / lamb1 * (term_y - beta_pos / gamma_pos * sign_ds9 * abs_ds9**(2 - gamma_pos))
    term_z = m_mu * lamb2**2 / lamb1 * np.abs(ez)**(2*m_mu - 1) * np.sign(ez)
    vzc = ddzdes - M11 * dz + g + 1 / lamb1 * (term_z - beta_pos / gamma_pos * sign_ds11 * abs_ds11**(2 - gamma_pos))

    # Position switching (Eq 18) -- FIXED to use sigma instead of s
    vxs = 1 / lamb1 * (-k1_pos * sigma7 - k2_pos * sign_sigma7)
    vys = 1 / lamb1 * (-k1_pos * sigma9 - k2_pos * sign_sigma9)
    vzs = 1 / lamb1 * (-k1_pos * sigma11 - k2_pos * sign_sigma11)

    vx = vxc + vxs
    vy = vyc + vys
    vz = vzc + vzs

    # Desired attitude (Eq 7a-b)
    denom = vz + g
    if abs(denom) < 1e-10:
        denom = 1e-10 * np.sign(denom) if denom != 0 else 1e-10
    theta_des = np.arctan((np.cos(psides) * vx + np.sin(psides) * vy) / denom)
    phi_des = np.arctan((np.cos(theta_des) * np.sin(psides) * vx - np.cos(psides) * vy) / denom)
    psi_des = psides

    # Attitude errors (Eq 25)
    e1 = phi - phi_des
    de1 = dphi - 0
    e3 = theta - theta_des
    de3 = dtheta - 0
    e5 = psi - psi_des
    de5 = dpsi - 0

    # Attitude ITSM (Eq 26)
    s1 = lamb1 * e1 + lamb2 * int_phi
    ds1 = lamb1 * de1 + lamb2 * np.abs(e1)**m_mu * np.sign(e1)
    s3 = lamb1 * e3 + lamb2 * int_theta
    ds3 = lamb1 * de3 + lamb2 * np.abs(e3)**m_mu * np.sign(e3)
    s5 = lamb1 * e5 + lamb2 * int_psi
    ds5 = lamb1 * de5 + lamb2 * np.abs(e5)**m_mu * np.sign(e5)

    # Clip ds
    abs_ds1 = np.clip(np.abs(ds1), 1e-10, 1e6)
    abs_ds3 = np.clip(np.abs(ds3), 1e-10, 1e6)
    abs_ds5 = np.clip(np.abs(ds5), 1e-10, 1e6)

    # Smooth sign
    sign_ds1 = ds1 / (abs_ds1 + eps_att)
    sign_ds3 = ds3 / (abs_ds3 + eps_att)
    sign_ds5 = ds5 / (abs_ds5 + eps_att)
    sign_s1 = s1 / (np.abs(s1) + eps_att)
    sign_s3 = s3 / (np.abs(s3) + eps_att)
    sign_s5 = s5 / (np.abs(s5) + eps_att)

    # NEW: Hyperplane variables (Eq 27)
    sigma1 = s1 + (1 / beta_att) * sign_ds1 * (abs_ds1 ** gamma_att)
    sigma3 = s3 + (1 / beta_att) * sign_ds3 * (abs_ds3 ** gamma_att)
    sigma5 = s5 + (1 / beta_att) * sign_ds5 * (abs_ds5 ** gamma_att)

    # Smooth sign for sigma
    sign_sigma1 = sigma1 / (np.abs(sigma1) + eps_att)
    sign_sigma3 = sigma3 / (np.abs(sigma3) + eps_att)
    sign_sigma5 = sigma5 / (np.abs(sigma5) + eps_att)

    # Attitude continuous (Eq 29)
    term_phi = m_mu * lamb2**2 / lamb1 * np.abs(e1)**(2*m_mu - 1) * np.sign(e1)
    u2c = 1 / (N1 * lamb1) * (term_phi - beta_att / gamma_att * sign_ds1 * abs_ds1**(2 - gamma_att) - lamb1 * (M1 * dtheta * dpsi + M2 * dtheta + M3 * dphi**2) + 0)
    term_theta = m_mu * lamb2**2 / lamb1 * np.abs(e3)**(2*m_mu - 1) * np.sign(e3)
    u3c = 1 / (N2 * lamb1) * (term_theta - beta_att / gamma_att * sign_ds3 * abs_ds3**(2 - gamma_att) - lamb1 * (M4 * dphi * dpsi + M5 * dphi + M6 * dtheta**2) + 0)
    term_psi = m_mu * lamb2**2 / lamb1 * np.abs(e5)**(2*m_mu - 1) * np.sign(e5)
    u4c = 1 / (N3 * lamb1) * (term_psi - beta_att / gamma_att * sign_ds5 * abs_ds5**(2 - gamma_att) - lamb1 * (M7 * dphi * dtheta + M8 * dpsi**2) + 0)

    # Attitude switching (Eq 30) -- FIXED to use sigma instead of s
    u2s = 1 / (N1 * lamb1) * (-k1_att * sigma1 - k2_att * sign_sigma1)
    u3s = 1 / (N2 * lamb1) * (-k1_att * sigma3 - k2_att * sign_sigma3)
    u4s = 1 / (N3 * lamb1) * (-k1_att * sigma5 - k2_att * sign_sigma5)

    u2 = u2c + u2s
    u3 = u3c + u3s
    u4 = u4c + u4s

    u2 = np.clip(u2, -50, 50)
    u3 = np.clip(u3, -50, 50)
    u4 = np.clip(u4, -50, 50)

    # Thrust (Eq 7c)
    u1 = m * np.sqrt(vx**2 + vy**2 + (vz + g)**2)
    u1 = np.clip(u1, 0, 100)

    # Corrected rotor speeds (inverse allocation)
    w1 = max(0, u1 / (4 * rho_y) - u3 / (2 * rho_y * d) - u4 / (4 * rho_z))
    w2 = max(0, u1 / (4 * rho_y) - u2 / (2 * rho_y * d) + u4 / (4 * rho_z))
    w3 = max(0, u1 / (4 * rho_y) + u3 / (2 * rho_y * d) - u4 / (4 * rho_z))
    w4 = max(0, u1 / (4 * rho_y) + u2 / (2 * rho_y * d) + u4 / (4 * rho_z))
    omega1 = np.sqrt(w1)
    omega2 = np.sqrt(w2)
    omega3 = np.sqrt(w3)
    omega4 = np.sqrt(w4)
    omega_r = omega1 - omega2 + omega3 - omega4

    # Update gyro
    M2 = -omega_r * Jr / Ix
    M5 = omega_r * Jr / Iy

    # Disturbances
    Dx, Dy, Dz, Dphi, Dtheta, Dpsi = disturbances(t)

    # Dynamics (Eq 5)
    ddx = M9 * dx + (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * u1 / m + Dx
    ddy = M10 * dy + (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * u1 / m + Dy
    ddz = M11 * dz - g + np.cos(phi) * np.cos(theta) * u1 / m + Dz
    ddphi = M1 * dtheta * dpsi + M2 * dtheta + M3 * dphi**2 + N1 * u2 + Dphi
    ddtheta = M4 * dphi * dpsi + M5 * dphi + M6 * dtheta**2 + N2 * u3 + Dtheta
    ddpsi = M7 * dphi * dtheta + M8 * dpsi**2 + N3 * u4 + Dpsi

    # Integral dots
    d_int_x = np.abs(ex)**m_mu * np.sign(ex)
    d_int_y = np.abs(ey)**m_mu * np.sign(ey)
    d_int_z = np.abs(ez)**m_mu * np.sign(ez)
    d_int_phi = np.abs(e1)**m_mu * np.sign(e1)
    d_int_theta = np.abs(e3)**m_mu * np.sign(e3)
    d_int_psi = np.abs(e5)**m_mu * np.sign(e5)

    return [dx, ddx, dy, ddy, dz, ddz, dphi, ddphi, dtheta, ddtheta, dpsi, ddpsi, d_int_x, d_int_y, d_int_z, d_int_phi, d_int_theta, d_int_psi]

# Initial conditions (Eq 39)
initial = [0.05, 0, 1, 0, 0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Time
# dt = 0.001
dt = 0.01

t = np.arange(0, 30, dt)

# Integrate
state = odeint(dynamics, initial, t)

# Recompute for plots (use smooth in recompute for consistency)
xdes_list = np.zeros(len(t))
ydes_list = np.zeros(len(t))
zdes_list = np.zeros(len(t))
phides_list = np.zeros(len(t))
thetades_list = np.zeros(len(t))
psides_list = np.zeros(len(t))
s7_list = np.zeros(len(t))
s9_list = np.zeros(len(t))
s11_list = np.zeros(len(t))
s1_list = np.zeros(len(t))
s3_list = np.zeros(len(t))
s5_list = np.zeros(len(t))
u1_list = np.zeros(len(t))
u2_list = np.zeros(len(t))
u3_list = np.zeros(len(t))
u4_list = np.zeros(len(t))

M1_base = M1
M4_base = M4
M7_base = M7
M3_base = -K1_base / Ix
M6_base = -K2_base / Iy
M8_base = -K3_base / Iz
M2_base = 0
M5_base = 0
M9_base = -K4_base / m
M10_base = -K5_base / m
M11_base = -K6_base / m

for i in range(len(t)):
    x = state[i, 0]
    dx = state[i, 1]
    y = state[i, 2]
    dy = state[i, 3]
    z = state[i, 4]
    dz = state[i, 5]
    phi = state[i, 6]
    dphi = state[i, 7]
    theta = state[i, 8]
    dtheta = state[i, 9]
    psi = state[i, 10]
    dpsi = state[i, 11]
    int_x = state[i, 12]
    int_y = state[i, 13]
    int_z = state[i, 14]
    int_phi = state[i, 15]
    int_theta = state[i, 16]
    int_psi = state[i, 17]

    xdes, dxdes, ddxdes, ydes, dydes, ddydes, zdes, dzdes, ddzdes = desired_pos(t[i])
    psides, _, _ = desired_psi(t[i])
    xdes_list[i] = xdes
    ydes_list[i] = ydes
    zdes_list[i] = zdes
    psides_list[i] = psides

    ex = x - xdes
    ey = y - ydes
    ez = z - zdes
    dex = dx - dxdes
    dey = dy - dydes
    dez = dz - dzdes

    s7 = lamb1 * ex + lamb2 * int_x
    s9 = lamb1 * ey + lamb2 * int_y
    s11 = lamb1 * ez + lamb2 * int_z
    s7_list[i] = s7
    s9_list[i] = s9
    s11_list[i] = s11
    ds7 = lamb1 * dex + lamb2 * np.abs(ex)**m_mu * np.sign(ex)
    ds9 = lamb1 * dey + lamb2 * np.abs(ey)**m_mu * np.sign(ey)
    ds11 = lamb1 * dez + lamb2 * np.abs(ez)**m_mu * np.sign(ez)

    abs_ds7 = np.clip(np.abs(ds7), 1e-10, 1e6)
    abs_ds9 = np.clip(np.abs(ds9), 1e-10, 1e6)
    abs_ds11 = np.clip(np.abs(ds11), 1e-10, 1e6)

    sign_ds7 = ds7 / (abs_ds7 + eps_pos)
    sign_ds9 = ds9 / (abs_ds9 + eps_pos)
    sign_ds11 = ds11 / (abs_ds11 + eps_pos)
    sign_s7 = s7 / (np.abs(s7) + eps_pos)
    sign_s9 = s9 / (np.abs(s9) + eps_pos)
    sign_s11 = s11 / (np.abs(s11) + eps_pos)

    # NEW: Hyperplane variables (Eq 10)
    sigma7 = s7 + (1 / beta_pos) * sign_ds7 * (abs_ds7 ** gamma_pos)
    sigma9 = s9 + (1 / beta_pos) * sign_ds9 * (abs_ds9 ** gamma_pos)
    sigma11 = s11 + (1 / beta_pos) * sign_ds11 * (abs_ds11 ** gamma_pos)

    # Smooth sign for sigma
    sign_sigma7 = sigma7 / (np.abs(sigma7) + eps_pos)
    sign_sigma9 = sigma9 / (np.abs(sigma9) + eps_pos)
    sign_sigma11 = sigma11 / (np.abs(sigma11) + eps_pos)

    term_x = m_mu * lamb2**2 / lamb1 * np.abs(ex)**(2*m_mu - 1) * np.sign(ex)
    vxc = ddxdes - M9_base * dx + 1 / lamb1 * (term_x - beta_pos / gamma_pos * sign_ds7 * abs_ds7**(2 - gamma_pos))
    term_y = m_mu * lamb2**2 / lamb1 * np.abs(ey)**(2*m_mu - 1) * np.sign(ey)
    vyc = ddydes - M10_base * dy + 1 / lamb1 * (term_y - beta_pos / gamma_pos * sign_ds9 * abs_ds9**(2 - gamma_pos))
    term_z = m_mu * lamb2**2 / lamb1 * np.abs(ez)**(2*m_mu - 1) * np.sign(ez)
    vzc = ddzdes - M11_base * dz + g + 1 / lamb1 * (term_z - beta_pos / gamma_pos * sign_ds11 * abs_ds11**(2 - gamma_pos))

    # Position switching -- FIXED to use sigma
    vxs = 1 / lamb1 * (-k1_pos * sigma7 - k2_pos * sign_sigma7)
    vys = 1 / lamb1 * (-k1_pos * sigma9 - k2_pos * sign_sigma9)
    vzs = 1 / lamb1 * (-k1_pos * sigma11 - k2_pos * sign_sigma11)

    vx = vxc + vxs
    vy = vyc + vys
    vz = vzc + vzs

    denom = vz + g
    if abs(denom) < 1e-10:
        denom = 1e-10 * np.sign(denom) if denom != 0 else 1e-10
    theta_des = np.arctan((np.cos(psides) * vx + np.sin(psides) * vy) / denom)
    phi_des = np.arctan((np.cos(theta_des) * np.sin(psides) * vx - np.cos(psides) * vy) / denom)
    psi_des = psides
    phides_list[i] = phi_des
    thetades_list[i] = theta_des

    e1 = phi - phi_des
    de1 = dphi - 0
    e3 = theta - theta_des
    de3 = dtheta - 0
    e5 = psi - psi_des
    de5 = dpsi - 0

    s1 = lamb1 * e1 + lamb2 * int_phi
    s3 = lamb1 * e3 + lamb2 * int_theta
    s5 = lamb1 * e5 + lamb2 * int_psi
    s1_list[i] = s1
    s3_list[i] = s3
    s5_list[i] = s5
    ds1 = lamb1 * de1 + lamb2 * np.abs(e1)**m_mu * np.sign(e1)
    ds3 = lamb1 * de3 + lamb2 * np.abs(e3)**m_mu * np.sign(e3)
    ds5 = lamb1 * de5 + lamb2 * np.abs(e5)**m_mu * np.sign(e5)

    abs_ds1 = np.clip(np.abs(ds1), 1e-10, 1e6)
    abs_ds3 = np.clip(np.abs(ds3), 1e-10, 1e6)
    abs_ds5 = np.clip(np.abs(ds5), 1e-10, 1e6)

    sign_ds1 = ds1 / (abs_ds1 + eps_att)
    sign_ds3 = ds3 / (abs_ds3 + eps_att)
    sign_ds5 = ds5 / (abs_ds5 + eps_att)
    sign_s1 = s1 / (np.abs(s1) + eps_att)
    sign_s3 = s3 / (np.abs(s3) + eps_att)
    sign_s5 = s5 / (np.abs(s5) + eps_att)

    # NEW: Hyperplane variables (Eq 27)
    sigma1 = s1 + (1 / beta_att) * sign_ds1 * (abs_ds1 ** gamma_att)
    sigma3 = s3 + (1 / beta_att) * sign_ds3 * (abs_ds3 ** gamma_att)
    sigma5 = s5 + (1 / beta_att) * sign_ds5 * (abs_ds5 ** gamma_att)

    # Smooth sign for sigma
    sign_sigma1 = sigma1 / (np.abs(sigma1) + eps_att)
    sign_sigma3 = sigma3 / (np.abs(sigma3) + eps_att)
    sign_sigma5 = sigma5 / (np.abs(sigma5) + eps_att)

    term_phi = m_mu * lamb2**2 / lamb1 * np.abs(e1)**(2*m_mu - 1) * np.sign(e1)
    u2c = 1 / (N1 * lamb1) * (term_phi - beta_att / gamma_att * sign_ds1 * abs_ds1**(2 - gamma_att) - lamb1 * (M1_base * dtheta * dpsi + M2_base * dtheta + M3_base * dphi**2) + 0)
    term_theta = m_mu * lamb2**2 / lamb1 * np.abs(e3)**(2*m_mu - 1) * np.sign(e3)
    u3c = 1 / (N2 * lamb1) * (term_theta - beta_att / gamma_att * sign_ds3 * abs_ds3**(2 - gamma_att) - lamb1 * (M4_base * dphi * dpsi + M5_base * dphi + M6_base * dtheta**2) + 0)
    term_psi = m_mu * lamb2**2 / lamb1 * np.abs(e5)**(2*m_mu - 1) * np.sign(e5)
    u4c = 1 / (N3 * lamb1) * (term_psi - beta_att / gamma_att * sign_ds5 * abs_ds5**(2 - gamma_att) - lamb1 * (M7_base * dphi * dtheta + M8_base * dpsi**2) + 0)

    # Attitude switching -- FIXED to use sigma
    u2s = 1 / (N1 * lamb1) * (-k1_att * sigma1 - k2_att * sign_sigma1)
    u3s = 1 / (N2 * lamb1) * (-k1_att * sigma3 - k2_att * sign_sigma3)
    u4s = 1 / (N3 * lamb1) * (-k1_att * sigma5 - k2_att * sign_sigma5)

    u2 = u2c + u2s
    u3 = u3c + u3s
    u4 = u4c + u4s

    u1 = m * np.sqrt(vx**2 + vy**2 + (vz + g)**2)

    u1_list[i] = u1
    u2_list[i] = u2
    u3_list[i] = u3
    u4_list[i] = u4

# Extract state for plots
x, y, z, phi, theta, psi = state[:,0], state[:,2], state[:,4], state[:,6], state[:,8], state[:,10]

# Plots as before
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, x, 'b', label='x')
plt.plot(t, xdes_list, 'r--', label='x_des')
plt.legend()
plt.subplot(3,1,2)
plt.plot(t, y, 'b', label='y')
plt.plot(t, ydes_list, 'r--', label='y_des')
plt.legend()
plt.subplot(3,1,3)
plt.plot(t, z, 'b', label='z')
plt.plot(t, zdes_list, 'r--', label='z_des')
plt.legend()
plt.suptitle('Results of the position under the proposed controller')
plt.savefig('position_sim2.png')

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, s7_list, 'b')
plt.ylabel('s_x')
plt.subplot(3,1,2)
plt.plot(t, s9_list, 'b')
plt.ylabel('s_y')
plt.subplot(3,1,3)
plt.plot(t, s11_list, 'b')
plt.ylabel('s_z')
plt.suptitle('Results of the position sliding variables under the proposed controller')
plt.savefig('pos_sliding_sim2.png')

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, s1_list, 'b')
plt.ylabel('s_ϕ')
plt.subplot(3,1,2)
plt.plot(t, s3_list, 'b')
plt.ylabel('s_θ')
plt.subplot(3,1,3)
plt.plot(t, s5_list, 'b')
plt.ylabel('s_ψ')
plt.suptitle('Results of the attitude sliding variables under the proposed controller')
plt.savefig('att_sliding_sim2.png')

plt.figure(figsize=(10,8))
plt.subplot(4,1,1)
plt.plot(t, u1_list, 'b')
plt.ylabel('u1')
plt.subplot(4,1,2)
plt.plot(t, u2_list, 'b')
plt.ylabel('u2')
plt.subplot(4,1,3)
plt.plot(t, u3_list, 'b')
plt.ylabel('u3')
plt.subplot(4,1,4)
plt.plot(t, u4_list, 'b')
plt.ylabel('u4')
plt.suptitle('Results of the inputs under the proposed controller')
plt.savefig('inputs_sim2.png')

plt.figure(figsize=(8,6))
plt.plot(x, y, 'b', label='Actual')
plt.plot(xdes_list, ydes_list, 'r--', label='Desired')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Results of the path following in 2D space under the proposed controller')
plt.savefig('2d_path_sim2.png')

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b', label='Actual')
ax.plot(xdes_list, ydes_list, zdes_list, 'r--', label='Desired')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_title('Results of the path following in 3D space under the proposed controller')
plt.savefig('3d_path_sim2.png')

# Attitude tracking
plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(t, phi, 'b', label='ϕ')
plt.plot(t, phides_list, 'r--', label='ϕ_des')
plt.legend()
plt.subplot(3,1,2)
plt.plot(t, theta, 'b', label='θ')
plt.plot(t, thetades_list, 'r--', label='θ_des')
plt.legend()
plt.subplot(3,1,3)
plt.plot(t, psi, 'b', label='ψ')
plt.plot(t, psides_list, 'r--', label='ψ_des')
plt.legend()
plt.suptitle('Attitude under the proposed controller')
plt.savefig('attitude_sim2.png')

plt.show()