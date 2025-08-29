# Lid-driven cavity 2D — vorticidade–função-corrente (Re=100)
# Implementação 100% NumPy (sem Numba). dt adaptativo por CFL.
# Uso: python cavity.py --N 256 --Re 100 --CFL 0.3 --tol 1e-6 --jiters 80

import argparse, os, time, csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--Re", type=float, default=100.0)
ap.add_argument("--CFL", type=float, default=0.3)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--jiters", type=int, default=80)   # iterações de Jacobi por passo
ap.add_argument("--maxiter", type=int, default=20000)
args = ap.parse_args()

N, Re, CFL, tol, JITERS, maxiter = args.N, args.Re, args.CFL, args.tol, args.jiters, args.maxiter
L = 1.0
dx = L/(N-1); dy = dx
nu = 1.0/Re
Utop = 1.0

omega = np.zeros((N,N), dtype=np.float64)
psi   = np.zeros((N,N), dtype=np.float64)
u     = np.zeros((N,N), dtype=np.float64)
v     = np.zeros((N,N), dtype=np.float64)
omega_new = np.zeros_like(omega)

def set_psi_bcs(psi):
    psi[:,0] = 0.0; psi[:,-1] = 0.0
    psi[0,:] = 0.0; psi[-1,:] = 0.0

def update_uv_from_psi(u, v, psi):
    # interior
    u[1:-1,1:-1] = (psi[1:-1,2:] - psi[1:-1,:-2])/(2*dy)
    v[1:-1,1:-1] = -(psi[2:,1:-1] - psi[:-2,1:-1])/(2*dx)
    # paredes (no-slip + tampa U=1)
    u[:,0] = 0.0; v[:,0] = 0.0
    u[:,-1] = Utop; v[:,-1] = 0.0
    u[0,:] = 0.0; v[0,:] = 0.0
    u[-1,:] = 0.0; v[-1,:] = 0.0

def update_wall_vorticity(omega, psi):
    omega[:,0]  = -2.0*psi[:,1]/(dy*dy)                         # bottom
    omega[:,-1] = -2.0*psi[:,-2]/(dy*dy) - 2.0*Utop/dy          # top
    omega[0,:]  = -2.0*psi[1,:]/(dx*dx)                         # left
    omega[-1,:] = -2.0*psi[-2,:]/(dx*dx)                        # right

def step_vorticity(omega, u, v, dt):
    dom = (slice(1,-1), slice(1,-1))
    om = omega
    adv = ( u[dom]*(om[2:,1:-1] - om[:-2,1:-1])/(2*dx)
          + v[dom]*(om[1:-1,2:] - om[1:-1,:-2])/(2*dy) )
    lap = ( (om[2:,1:-1] - 2*om[1:-1,1:-1] + om[:-2,1:-1])/(dx*dx)
          + (om[1:-1,2:] - 2*om[1:-1,1:-1] + om[1:-1,:-2])/(dy*dy) )
    new = om.copy()
    new[dom] = om[dom] + dt*(-adv + nu*lap)
    return new

def jacobi_psi(psi, omega, iters):
    dx2 = dx*dx; dy2 = dy*dy
    beta = 1.0/(2*(dx2+dy2))
    psi_new = np.empty_like(psi)
    for _ in range(iters):
        set_psi_bcs(psi)
        psi_new[1:-1,1:-1] = beta*( (psi[2:,1:-1] + psi[:-2,1:-1])*dy2
                                   + (psi[1:-1,2:] + psi[1:-1,:-2])*dx2
                                   + dx2*dy2*omega[1:-1,1:-1] )
        psi_new[:,0] = 0.0; psi_new[:,-1] = 0.0
        psi_new[0,:] = 0.0; psi_new[-1,:] = 0.0
        psi, psi_new = psi_new, psi
    return psi

t0 = time.perf_counter()
steps = 0
dt_acc = 0.0
res = np.inf

# warm-up leve
set_psi_bcs(psi); update_uv_from_psi(u,v,psi); update_wall_vorticity(omega,psi)
omega_new = step_vorticity(omega, u, v, 1e-6)
psi = jacobi_psi(psi, omega, 1)

for it in range(1, maxiter+1):
    set_psi_bcs(psi)
    update_uv_from_psi(u, v, psi)
    update_wall_vorticity(omega, psi)

    umax = max(np.max(np.abs(u)), 1e-12)
    vmax = max(np.max(np.abs(v)), 1e-12)
    dtc1 = dx/umax; dtc2 = dy/vmax
    dtd1 = dx*dx/(4*nu); dtd2 = dy*dy/(4*nu)
    dt = CFL * min(dtc1, dtc2, dtd1, dtd2)

    omega_new = step_vorticity(omega, u, v, dt)

    # alinhar bordas antes de medir resíduo
    omega_new[0,:]  = omega[0,:]
    omega_new[-1,:] = omega[-1,:]
    omega_new[:,0]  = omega[:,0]
    omega_new[:,-1] = omega[:,-1]
    res = np.max(np.abs(omega_new - omega))

    omega = omega_new
    psi = jacobi_psi(psi, omega, JITERS)

    steps += 1
    dt_acc += dt

    if it % 200 == 0:
        print(f"Iter {it}, dt={dt:.2e}, res={res:.2e}")
    if res < tol:
        print(f"Convergiu em {it} passos")
        break

elapsed = time.perf_counter() - t0
cells = (N-2)*(N-2)
mlups_vort = cells*steps/elapsed/1e6
avg_dt = dt_acc/max(steps,1)

print(f"N={N} Re={Re:.1f} steps={steps} time={elapsed:.3f}s MLUPS(vort)={mlups_vort:.2f} "
      f"avg_dt={avg_dt:.2e} res={res:.2e}")

# perfis centrais (nomes iguais aos da Julia/Python Numba para o run.py usar)
y = np.linspace(0.0, 1.0, N); x = y
np.savetxt(f"u_center_python_N{N}.dat", np.c_[y, u[N//2,:]])
np.savetxt(f"v_center_python_N{N}.dat", np.c_[x, v[:,N//2]])

# resumo CSV (append) — usa summary_python.csv para uniformizar
sumfile = "summary_python.csv"
newfile = not os.path.exists(sumfile)
with open(sumfile, "a", newline="") as f:
    import csv
    w = csv.writer(f)
    if newfile:
        w.writerow(["timestamp","N","Re","CFL","tol","jacobi_iters","steps","time_s","avg_dt","MLUPS_vort","res"])
    w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"),
                N, Re, CFL, tol, JITERS, steps, f"{elapsed:.6f}", f"{avg_dt:.6e}",
                f"{mlups_vort:.3f}", f"{res:.3e}"])
