import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import sympy as sp
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy import integrate

### Linear Advection using Upwind Euler ###

def Euler(u, f, t, x):
    
    delta_x = x[1] - x[0]
    delta_t = t[1] - t[0]
    delta_xt = delta_t/delta_x

    # Copy initial condition
    u = u.copy()
    u_big = np.zeros((len(t), len(x)))
    u_big[0] = u.copy()

    # Time stepping loop
    for i in range(1, len(t)):
        u_new = u.copy()
        v_vals = f(t[i], x)      # velocity at current time/space
        pos_mask = v_vals > 0    # upwind direction depends on velocity sign

        # backward diff (for v>0)
        backward = (u - np.roll(u, 1))

        # forward diff (for v<0)
        forward = (np.roll(u, -1) - u)

        # choose correct derivative for each point
        u_x = np.where(pos_mask, backward, forward)

        # update all points at once
        u_new = u - delta_xt * v_vals * u_x
        u = u_new

        # store solution
        u_big[i] = u.copy()
    return u_big

### Inviscid Burger's Equations ###

def Euler_Burgers(u, t, x):
    
    delta_x = x[1] - x[0]
    delta_t = np.diff(t)
    delta_xt = delta_t/delta_x
    
    # Copy initial condition
    u = u.copy()
    u_big = np.zeros((len(t), len(x)))
    u_big[0] = u.copy()

    # Time stepping loop
    for i in range(1, len(t)):
        v_vals = u      # characteristic speed = u
        pos_mask = v_vals > 0    # upwind direction depends on velocity sign

        # backward diff (for v>0)
        backward = (u - np.roll(u, 1))

        # forward diff (for v<0)
        forward = (np.roll(u, -1) - u)

        # choose correct derivative for each point
        u_x = np.where(pos_mask, backward, forward)

        # update all points at once
        u = u - delta_xt[i-1] * v_vals * u_x

        # store solution
        u_big[i] = u.copy()
        
    return u_big

# Shock Time

def ShockTime_Burgers(x, u0):
    dx = x[1] - x[0]
    du_dx = np.gradient(u0, dx) # Finds gradient
    ShockTime = -1/(min(du_dx))
    return ShockTime

# Mapped Time

def Mapped_Euler_Burgers(u, tau, x):
    dx = x[1] - x[0]
    dtau = np.diff(tau)

    ux_0 = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
    ux_inf = np.max(np.abs(ux_0))
    v0 = u / ux_inf
    
    # initial mapped field
    v = v0.copy()

    # storage
    v_big = np.zeros((len(tau), len(x)))
    v_big[0] = v.copy()

    for i in range(1, len(tau)):
        v_vals = v
        pos_mask = v_vals > 0 # upwind direction depends on velocity sign

        # upwind derivatives
        backward = (v - np.roll(v, 1)) / dx
        forward  = (np.roll(v, -1) - v) / dx
        # choose correct derivative for each point
        v_x = np.where(pos_mask, backward, forward)

        v = v - dtau[i-1] * (v_vals * v_x + v)

        # normalization solution
        v_x_full = (np.roll(v, -1) - np.roll(v, 1)) / (2 * dx)
        norm = np.max(np.abs(v_x_full))
        v /= norm

        # store solution
        v_big[i] = v.copy()

    return v_big

def interp(x0,x1,x2,x):
    return (x0[1]*(x-x1[0])*(x-x2[0])/((x0[0]-x1[0])*(x0[0]-x2[0])) +
            x1[1]*(x-x2[0])*(x-x0[0])/((x1[0]-x2[0])*(x1[0]-x0[0])) +
            x2[1]*(x-x0[0])*(x-x1[0])/((x2[0]-x0[0])*(x2[0]-x1[0])))

# Function to interpolate gradient to find accurate maximum
def interpdiv(x0,x1,x2):
    return -((x0[0] + x2[0])*x1[1]*(x2[0] - x0[0]) - (x1[0] + x2[0])*x0[1]*(x2[0] - x1[0]) - (x0[0] + x1[0])*x2[1]*(x1[0] - x0[0])) / (2 * (x0[1] * (x2[0] - x1[0]) - x1[1] * (x2[0] - x0[0]) + x2[1] * (x1[0] - x0[0])))

def ThreePointInterpolation(x):
    grid = len(x)
    if x.ndim == 2:
        idx = np.argmax(np.abs(x), axis = 1)
        t_idx = np.arange(len(idx))
    else:
        idx = np.argmax(np.abs(x))
    
    i0 = (idx - 1) % grid
    i1 = idx
    i2 = (idx + 1) % grid
    if x.ndim == 2:
        y0 = np.abs(x[t_idx, i0])
        y1 = np.abs(x[t_idx, i1])
        y2 = np.abs(x[t_idx, i2])
    else:
        y0 = np.abs(x[i0])
        y1 = np.abs(x[i1])
        y2 = np.abs(x[i2])
        
    return y1 - (y2 - y0)**2 / (8*(y2 - 2*y1 + y0))

# Pseudo Spectral Method using Fourier Basis

def SpectralMethodInviscidBurgers(u0, L, x, t):
    N = len(x)
    dt = np.diff(t)

    uhat = np.fft.fft(u0)
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N)

    u_big = np.zeros((len(t), len(x)))
    u_big[0] = u0
    
    def f(uhat, k):
        u = np.fft.ifft(uhat)
        ux = np.fft.ifft(1j*k*uhat)
        f = u * ux
        fhat = np.fft.fft(f)
        fhat[np.abs(k) > N / 3] = 0.0
        duhat = -fhat
        return duhat
    for i in range(1, len(t)):
        K1 =  f(uhat, k)
        K2 =  f(uhat + 0.5 * dt[i-1] * K1, k)
        K3 =  f(uhat + 0.5 * dt[i-1] * K2, k)
        K4 =  f(uhat + dt[i-1] * K3, k)
        uhat = uhat+(K1 + 2*K2 + 2*K3 + K4)*dt[i-1]/6
        u_big[i] = np.real(np.fft.ifft(uhat))
    return u_big

def Mapped_SpectralMethodInviscidBurgers(u0, L, x, t): 
    dx = x[1] - x[0] 
    N = len(x) 
    dt = np.diff(t)

    k = 2*np.pi*np.fft.fftfreq(N, d=L/N) 
    uhat0 = np.fft.fft(u0)
    ux0 = np.real(np.fft.ifft(1j * k * uhat0))

    ux_inf = np.max(np.abs(ux0))
    v0 = u0 / ux_inf 
    
    uhat = np.fft.fft(v0) 
    k = 2*np.pi*np.fft.fftfreq(N, d=L/N) 
    u_big = np.zeros((len(t), len(x))) 
    u_big[0] = v0

    def f(uhat, k): 
        u = np.fft.ifft(uhat) 
        ux = np.fft.ifft(1j*k*uhat) 
        f = u * ux
        fhat = np.fft.fft(f) + uhat
        fhat[np.abs(k) > N / 3] = 0.0
        duhat = -fhat
        
        return duhat

    for i in range(1, len(t)): 
        K1 = f(uhat, k) 
        K2 = f(uhat + 0.5 * dt[i-1] * K1, k) 
        K3 = f(uhat + 0.5 * dt[i-1] * K2, k) 
        K4 = f(uhat + dt[i-1] * K3, k) 
        uhat = uhat+(K1 + 2*K2 + 2*K3 + K4)*dt[i-1]/6

        # normalization solution 
        v = np.real(np.fft.ifft(uhat))
        vx = np.real(np.fft.ifft(1j * k * uhat))

        norm = ThreePointInterpolation(vx)
        uhat /= norm
    
        u_big[i] = np.real(np.fft.ifft(uhat)) 
    return u_big