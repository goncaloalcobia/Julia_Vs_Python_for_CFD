#!/usr/bin/env python3
import argparse, os, sys, subprocess, shutil, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- paths ----------------
HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"; FIGS.mkdir(exist_ok=True)
LOGS = HERE / "logs"; LOGS.mkdir(exist_ok=True)
DATA = HERE / "data"   # novo: diretório onde os .dat podem estar

# --------------- helpers ---------------
def run_cmd(cmd, log_path: Path):
    print(">>", " ".join(cmd))
    with open(log_path, "w") as lf:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=HERE
        )
        for line in p.stdout:
            sys.stdout.write(line); lf.write(line)
    if p.wait() != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def tail_last_row(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return None if df.empty else df.iloc[-1].to_dict()

def append_run(runs, row, language, threads):
    if not row:
        return
    runs.append({
        "language": language,
        "threads": int(threads),
        "N": int(row["N"]),
        "Re": float(row["Re"]),
        "CFL": float(row["CFL"]),
        "tol": float(row["tol"]),
        "jacobi_iters": int(row["jacobi_iters"]),
        "steps": int(row["steps"]),
        "time_s": float(row["time_s"]),
        "avg_dt": float(row["avg_dt"]),
        "MLUPS_vort": float(row["MLUPS_vort"]),
        "res": float(row["res"]),
    })

def plot_time_mlups(df, out_prefix, note=""):
    df = df.copy()
    df["N"] = df["N"].astype(int)
    df = df.sort_values(["language", "N"])

    # time
    plt.figure()
    for lang, g in df.groupby("language"):
        plt.plot(g["N"], g["time_s"], marker="o", label=lang)
    plt.xlabel("Grid size N"); plt.ylabel("Time to solution [s]")
    #plt.title(f"Lid-driven cavity — Time {note}".strip())
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / f"{out_prefix}_time.png", dpi=200)

    # mlups
    plt.figure()
    for lang, g in df.groupby("language"):
        plt.plot(g["N"], g["MLUPS_vort"], marker="o", label=lang)
    plt.xlabel("Grid size N"); plt.ylabel("Throughput [MLUPS]")
    #plt.title(f"Lid-driven cavity — MLUPS {note}".strip())
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / f"{out_prefix}_mlups.png", dpi=200)

def plot_speedup_vs_numpy(df, out_prefix, note=""):
    pv = df.pivot_table(index="N", columns="language", values="time_s", aggfunc="min")
    if "Python-NumPy" not in pv.columns or "Julia" not in pv.columns:
        return
    sp = pv["Python-NumPy"] / pv["Julia"]  # >1 → Julia mais rápida
    plt.figure()
    plt.plot(sp.index, sp.values, marker="o", label="NumPy/Julia")
    plt.axhline(1.0, ls="--", alpha=0.5)
    plt.xlabel("Grid size N"); plt.ylabel("Speedup (NumPy/Julia)")
    #plt.title(f"Speedup vs Julia (NumPy/Julia) {note}".strip())
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / f"{out_prefix}_speedup_numpy_over_julia.png", dpi=200)

# preferir ficheiros em data/, com fallback para raiz
def _pick(prefer: Path, fallback: Path) -> Path:
    return prefer if prefer.exists() else fallback

def plot_profiles(N, out_prefix):
    # u-centerline
    u_files = [
        ("Julia",        _pick(DATA / f"u_center_julia_N{N}.dat",  HERE / f"u_center_julia_N{N}.dat")),
        ("Python-NumPy", _pick(DATA / f"u_center_python_N{N}.dat", HERE / f"u_center_python_N{N}.dat")),
    ]
    if all(p.exists() for _, p in u_files):
        plt.figure()
        for name, p in u_files:
            d = np.loadtxt(p)
            plt.plot(d[:, 0], d[:, 1], label=name, linestyle="-" if name == "Julia" else "--")
        plt.xlabel("y"); plt.ylabel("u(x=0.5,y)")
        #plt.title(f"Centerline u-profile (N={N})")
        plt.legend(); plt.tight_layout()
        plt.savefig(FIGS / f"{out_prefix}_u_profile_N{N}.png", dpi=200)

    # v-centerline
    v_files = [
        ("Julia",        _pick(DATA / f"v_center_julia_N{N}.dat",  HERE / f"v_center_julia_N{N}.dat")),
        ("Python-NumPy", _pick(DATA / f"v_center_python_N{N}.dat", HERE / f"v_center_python_N{N}.dat")),
    ]
    if all(p.exists() for _, p in v_files):
        plt.figure()
        for name, p in v_files:
            d = np.loadtxt(p)
            plt.plot(d[:, 0], d[:, 1], label=name, linestyle="-" if name == "Julia" else "--")
        plt.xlabel("x"); plt.ylabel("v(x,y=0.5)")
        #plt.title(f"Centerline v-profile (N={N})")
        plt.legend(); plt.tight_layout()
        plt.savefig(FIGS / f"{out_prefix}_v_profile_N{N}.png", dpi=200)

def archive_outputs(label: str):
    """Copia todos os outputs para runs/YYYYMMDD_HHMMSS/…"""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = HERE / "runs" / stamp
    figs_dst  = root / "figs"
    logs_dst  = root / "logs"
    perf_dst  = root / "performance_results"
    phys_dst  = root / "physical_results"
    for d in (figs_dst, logs_dst, perf_dst, phys_dst):
        d.mkdir(parents=True, exist_ok=True)

    # figuras
    for f in FIGS.glob("*.png"):
        shutil.copy2(f, figs_dst / f.name)

    # logs
    for f in LOGS.glob("*.log"):
        shutil.copy2(f, logs_dst / f.name)

    # csv de performance
    for pat in ("summary_julia.csv", "summary_python.csv", f"results_combined_{label}.csv"):
        fp = HERE / pat
        if fp.exists():
            shutil.copy2(fp, perf_dst / fp.name)

    # resultados físicos (perfis) — raiz e data/
    for f in HERE.glob("u_center_*.dat"):
        shutil.copy2(f, phys_dst / f.name)
    for f in HERE.glob("v_center_*.dat"):
        shutil.copy2(f, phys_dst / f.name)
    if DATA.exists():
        for f in DATA.glob("u_center_*.dat"):
            shutil.copy2(f, phys_dst / f.name)
        for f in DATA.glob("v_center_*.dat"):
            shutil.copy2(f, phys_dst / f.name)

    print(f"\n==> Arquivo criado em: {root}")
    print(f"   - {figs_dst}")
    print(f"   - {logs_dst}")
    print(f"   - {perf_dst}")
    print(f"   - {phys_dst}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[128, 256, 384])
    ap.add_argument("--Re", type=float, default=100.0)
    ap.add_argument("--CFL", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--jiters", type=int, default=80)
    ap.add_argument("--label", type=str, default="st")
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    jl = HERE / "cavity.jl"
    py = HERE / "cavity.py"
    if not jl.exists() or not py.exists():
        print("Precisas de cavity.jl e cavity.py nesta pasta.", file=sys.stderr)
        sys.exit(1)

    julia_thr = os.environ.get("JULIA_NUM_THREADS", "1")
    numpy_thr = os.environ.get("OPENBLAS_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))

    runs = []
    if not args.skip_run:
        for N in args.Ns:
            # Julia
            jl_log = LOGS / f"julia_N{N}_{args.label}.log"
            run_cmd(
                ["julia", "-O3", "--check-bounds=no", str(jl),
                 str(N), str(args.Re), str(args.CFL), f"{args.tol:.1e}", str(args.jiters)],
                jl_log
            )
            row = tail_last_row(HERE / "summary_julia.csv"); append_run(runs, row, "Julia", julia_thr)

            # Python NumPy
            py_log = LOGS / f"python_numpy_N{N}_{args.label}.log"
            run_cmd(
                [sys.executable, str(py),
                 "--N", str(N), "--Re", str(args.Re), "--CFL", str(args.CFL),
                 "--tol", f"{args.tol:.1e}", "--jiters", str(args.jiters)],
                py_log
            )
            row = tail_last_row(HERE / "summary_python.csv"); append_run(runs, row, "Python-NumPy", numpy_thr)

    # juntar tudo (inclui runs antigos dos summaries)
    def safe_load(p, lang, thr):
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        if df.empty:
            return df
        df["language"] = lang
        df["threads"] = int(thr)
        return df

    df_all = []
    df_all.append(safe_load(HERE / "summary_julia.csv", "Julia", julia_thr))
    df_all.append(safe_load(HERE / "summary_python.csv", "Python-NumPy", numpy_thr))
    if runs:
        df_all.append(pd.DataFrame(runs))
    dfc = pd.concat([d for d in df_all if d is not None and not d.empty], ignore_index=True)
    # filtra Ns pedidos
    dfc = dfc[dfc["N"].isin(args.Ns)].copy()

    # normalizar tipos
    for c in ["N", "threads", "jacobi_iters", "steps"]:
        dfc[c] = dfc[c].astype(int)
    for c in ["Re", "CFL", "tol", "time_s", "avg_dt", "MLUPS_vort", "res"]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce")

    out_csv = HERE / f"results_combined_{args.label}.csv"
    dfc.to_csv(out_csv, index=False)

    note = f"(threads: Julia={julia_thr}, NumPy={numpy_thr})"
    plot_time_mlups(dfc, out_prefix=f"bench_{args.label}", note=note)
    plot_speedup_vs_numpy(dfc, out_prefix=f"bench_{args.label}", note=note)
    for N in sorted(dfc["N"].unique()):
        plot_profiles(N, out_prefix=f"profiles_{args.label}")

    # arquivar tudo numa pasta com timestamp
    archive_outputs(args.label)

    print("\n==> Feito.")
    print(f"- CSV combinado: {out_csv}")
    print(f"- Figuras: {FIGS.resolve()}")
    print(f"- Logs: {LOGS.resolve()}")

if __name__ == "__main__":
    main()
