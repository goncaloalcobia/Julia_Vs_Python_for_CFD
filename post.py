import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(".")
df = pd.read_csv("results_combined_st.csv")

# speedup Julia vs Python
dfp = df.pivot_table(index="N", columns="language", values="time_s")
dfp["speedup(Julia/Python)"] = dfp["Python"]/dfp["Julia"]
print(dfp)

plt.figure()
dfp["speedup(Julia/Python)"].plot(marker="o")
plt.xlabel("Grid size N")
plt.ylabel("Speedup Julia/Python")
plt.title("Julia vs Python speedup (single-thread)")
plt.grid(True, ls="--", alpha=0.4)
plt.savefig("figs/speedup.png", dpi=200)

# perfis centrais
for N in df["N"].unique():
    J_u = np.loadtxt(f"u_center_julia_N{N}.dat")
    P_u = np.loadtxt(f"u_center_python_N{N}.dat")
    plt.figure()
    plt.plot(J_u[:,0], J_u[:,1], label="Julia")
    plt.plot(P_u[:,0], P_u[:,1], "--", label="Python")
    plt.xlabel("y"); plt.ylabel("u(x=0.5,y)")
    plt.title(f"u-profile comparison N={N}")
    plt.legend(); plt.grid(True, ls="--", alpha=0.4)
    plt.savefig(f"figs/u_profile_compare_N{N}.png", dpi=200)

    # diferença L2
    err = np.linalg.norm(J_u[:,1]-P_u[:,1])/np.linalg.norm(P_u[:,1])
    print(f"N={N}: L2 error Julia vs Python u-profile = {err:.2e}")

