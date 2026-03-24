#morris.py
import matlab
import matlab.engine
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import pandas as pd
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
# from SALib.test_functions import sobol_G as evaluate_model

# print("Starting Matlab")
# eng=matlab.engine.start_matlab()
# eng.addpath('/Users/bryanbarrios/Desktop/MastersResearch/MathmaticalModeling/modelDriver', nargout=0)

# Integrate from t0 to tf
T0, Tf = 0.0, 5000
paraLog=True
rel_bound = .15 # Plus/Minus around each base parameter
n=100 # Should increase for more stable results
num_levels=4
out_dir="SensitivityAnalysis/Morris/3_22"
max_run_time_sec=30
log_epsilon =1e-8


load_mat="/Users/bryanbarrios/Desktop/MastersResearch/DataCopeland.mat"
def build_bounds(base, rel=.2, eps=1e-12):
    base=np.asarray(base,float).ravel()
    lo=np.maximum(0,base*(1.0-rel))
    hi=base*(1.0+rel)
    hi=np.where( (hi<= lo) & (np.abs(base)<= eps), lo + 1e-6,hi)
    return lo,hi

def ensure_col(x):
    x=np.asarray(x,float)
    if x.ndim==1:
        x=x.reshape(-1,1)
    return x

def main():
    print("Starting Matlab")
    eng=matlab.engine.start_matlab()
    eng.addpath('/Users/bryanbarrios/Desktop/MastersResearch/MathmaticalModeling/ODE_Model', nargout=0)
    eng.eval(f"addpath(genpath('/Users/bryanbarrios/Desktop/MastersResearch/MathmaticalModeling/ODE_Model'));", nargout=0)
    
    load_mat = "/Users/bryanbarrios/Desktop/MastersResearch/MathmaticalModeling/ODE_Model/DataCopeland.mat"
    eng.load(load_mat, nargout=0)   
    eng.eval("global BPo Tm ODE_TOL; ODE_TOL = 1e-8;", nargout=0)
    eng.load(load_mat, nargout=0)

    eng.eval(f"""

        data.BP = BPm;    data.hr = HRm;
        data.TNF = TNFm;  data.IL6 = IL6m;  data.IL8 = IL8m;
        data.temp = TEMPm(1:7);
        data.age = 29; data.weight = 79.9; data.height = 177;
        data.HM = 207 - 0.7 * data.age;
        global BPo Tm ODE_TOL;
        BPo = data.BP(1); Tm = data.temp(1); ODE_TOL = 1e-8;
        [pars, Init] = load_pars_Init_Copeland_Edited(data);""" ,nargout=0)
    base_pars=np.array(eng.workspace["pars"], float).ravel()
    Init=eng.workspace["Init"]
    k=base_pars.size # K is the number of par to analyze
    
    par_names = [
            'k10', 'k10m', 'k6', 'k6m', 'k8', 'k8m', 'ktnf', 'ktnfm', 'kma', 'kmpe',# 1–10
            'kmr', 'kpe', 'x610', 'x66', 'x6tnf', 'x810', 'x8tnf', 'x106', 'xtnf10', 'xtnf6',# 11–20
            'xmpe', 'xm10', 'xmtnf', 'h106', 'h6tnf', 'h66', 'h610', 'h8tnf', 'h810', 'htnf10',            # 21–30
            'htnf6', 'hm10', 'hmtnf', 'hmpe', 'stnf', 's10', 's8', 's6', 'sm', 'mmax',            # 31–40
            'k6tnf', 'k8tnf', 'k106', 'kmtnf',            # 41–44
            'tau1', 'TM', 'Tm', 'kt', 'kttnf', 'kt6', 'kt10', 'xttnf', 'xt6', 'xt10',            # 45–54
            'httnf', 'ht6', 'ht10',            # 55–57
            'tau2', 'HM', 'HI', 'kh', 'xht', 'hht',            # 58–63
            'ppM', 'kpepp', 'kpp',            # 64–66
            'Ra', 'Rv', 'Rs', 'Cla', 'Csa', 'Clv', 'Csv', 'Em', 'EM', # 67–75
            'knom', 'kno', 'xntnf', 'xn10', 'hntnf', 'hn10',            # 76–81
            'krpp', 'krno', 'kr', 'xrpp', 'hrpp', 'xhp', 'hhp',            # 82–88
            'kpg', 'peinf', 'kpm', 'xI10', 'muno', 'kpn',            # 89–94
            'kdn', 'mud', 'xdn', 'alpha', 'hmI10',            # 95–99
            'sM', 'kmp',            # 100–101
        'kD', 'xDam', 'hmDa', 'xm10D', 'hm10D', 'hmD', 'knod' ,'xnDl10' ,'hnDl10', 'xn10D', 'hn10D' ,'ktnfhr'
        ]
    assert len(par_names) == k, "param_names length must match number of parameters k"

    ignore_pars=[
        10,12,
        35,37,38,40,90,
        46, 47,
        59,60,
        64,
        67,68,69,70,71,72,73,74,75,]
    ignore_pars_0=[idx-1 for idx in ignore_pars]
    
    
    active_mask = np.ones(k, dtype=bool)
    active_mask[ignore_pars_0] = False

    # Indices of active parameters
    active_idx = np.where(active_mask)[0]
    
    lo,hi=build_bounds(base_pars,rel_bound) # Creates Morris bounds 
    
    # Log Space Bounds
    if paraLog:
        eps=1e-12
        lo=np.maximum(lo,eps)
        hi=np.maximum(hi,lo*(1.0+1e-6))
        
        lo=np.log(lo)
        hi=np.log(hi)
    # Subset bounds and names to only para being perturb
    lo_perturb=lo[active_idx]
    hi_perturb=hi[active_idx]
    active_names = [par_names[i] for i in active_idx]

    problem={
        "num_vars":len(active_idx),
        "names": active_names,
        "bounds": np.c_[lo_perturb,hi_perturb].tolist(),
    }
    # Generates the morris sample
    # N=n is the number of trajectories, each trajectory has k+1 points -> total model runs
    # num_levels controls the resolution of the grid on each par
    # optimal_tracjectories=None -> random trajectories
    ## Or we can set it to True to use the optimal algorithm (better space filling)
    X=morris_sample(problem, N=n, num_levels=num_levels, optimal_trajectories=None)
    n_runs=X.shape[0]
    print(f"Morris sample:{n_runs}= N*(k+1) with N={n}, k={k}")
    time=matlab.double([float(T0),float(Tf)])
    
    # number of states
    print("Initilizing the model to get number of states")
    t0_m, sol0_m = eng.modelDriver(matlab.double(base_pars.tolist()), Init, time, nargout=2)
    t0 = np.array(t0_m).ravel()
    sol0 = ensure_col(np.array(sol0_m, float))
    n_states = sol0.shape[1]
    print(f"Detected n_states = {n_states}")
    
    # Evaluate all morris samples
    Y_all = np.zeros((n_runs, n_states), float)
    Y_all[:] = np.nan 
    print("Running model for all samples")
    
    timed_out = 0
    failed=0

    
    # For each sampled par set, it integrates the ODEs
    for i, row in enumerate(X, start=1): # Recall that X is the morris design matrix, with shape (n_runs, k)
        ## Each row is one par vector with length K
        
        # Converts the par vector for matlab
        # pars = matlab.double(np.asarray(row, float).tolist())# python list of floats -> matlab.double
        
        row=np.asarray(row,float)
        if paraLog:
            active_par_vals=np.exp(row)
        else:
            active_par_vals=row
        par_vec_full=base_pars.copy()
        par_vec_full[active_idx] = active_par_vals

        pars = matlab.double(par_vec_full.tolist())
        
        # Will run with timeout using asynchroous call
        try:
            solving=eng.modelDriver(pars,Init,time,nargout=2,background=True)
            t_m, sol_m = solving.result(timeout=max_run_time_sec)  # seconds
        except matlab.engine.TimeoutError:
            # Now cancel if still running
            try:
                solving.cancel()
            except Exception:
                pass
            timed_out += 1
            print(f"  Run {i}/{n_runs} exceeded {max_run_time_sec}")
            continue
        except Exception as e:
            failed += 1
            print(f"  Run {i}/{n_runs} failed with error: {e} so ignore.")
            continue


        # Runs the ODE once for that par set
        ## Pars: current par
        ## Init: initial conditions(kept fixed across runs)
        ## nargout=2: return both t (time grid) and soln 
        # t_m, sol_m = eng.modelDriver(pars, Init, time, nargout=2)
        
        # Normalizuing Matlab output into numpy arrays
        t = np.array(t_m).ravel() # With shape (nt,); ravel() flattens t to 1d 
        sol = ensure_col(np.array(sol_m, float)) # With shape (nt, n_states);and ensure_col ensure that sol is 2d
        if sol.shape[1] != n_states: # Just ensures that the model dim didnt change
            print(f"  Run {i}: state count changed ({sol.shape[1]} vs {n_states}) so ignore")
            failed += 1
            continue
            # raise RuntimeError(f"State count changed on run {i}: {sol.shape[1]} vs {n_states}")
        # Then summariesthe trajectory by AUC per state. Thats the scalar response Y for morris
        # Y_all[i-1, :] = [float(np.trapezoid(sol[:, j], x=t)) for j in range(n_states)]
        
        # For log,
        # sol_clipped=np.clip(sol,log_epsilon,None)
        # Y_all[i-1, :] = [float(np.trapezoid(np.log(sol_clipped[:, j]), x=t)) for j in range(n_states)]

        # treats nonpositive/invalid AUC as invalid run
        auc = np.trapezoid(sol, x=t, axis=0)  
        log_auc = np.where(np.isfinite(auc) & (auc > 0.0), np.log(auc), np.nan)
        Y_all[i-1, :] = log_auc
        

        if i % max(1, n_runs // 10) == 0:
            print(f"  {i}/{n_runs}")
            
    
    D = problem["num_vars"]              # number of active parameters
    traj_size = D + 1                    # points per trajectory
    if n_runs % traj_size != 0:
        raise RuntimeError(
            f"Sample size {n_runs} is not a multiple of traj_size={traj_size}"
        )
    n_traj_total = n_runs // traj_size

    # A run is valid if all state AUCs are finite
    run_valid_mask = np.all(np.isfinite(Y_all), axis=1)

    traj_valid = np.ones(n_traj_total, dtype=bool)
    for t in range(n_traj_total):
        start = t * traj_size
        end = start + traj_size
        if not np.all(run_valid_mask[start:end]):
            traj_valid[t] = False

    # Expand traj_valid back to a run-level mask
    keep_mask = np.repeat(traj_valid, traj_size)

    n_valid_runs = int(keep_mask.sum())
    n_valid_traj = int(traj_valid.sum())

    print(f" total runs: {n_runs}")
    print(f"total trajectories: {n_traj_total}")
    print(f" valid trajectories used: {n_valid_traj}")
    print(f" valid runs used: {n_valid_runs}")
    print(f" timed out runs: {timed_out}")
    print(f" failed runs: {failed}")

    if n_valid_traj == 0:
        eng.quit()
        return

    X_valid = X[keep_mask, :]
    Y_all_valid = Y_all[keep_mask, :]

    
    
    os.makedirs(out_dir, exist_ok=True)
    names = problem["names"]

    state_eqn_names = ["tnf","il10","il8","il6","ma","mr","pe","temp","pp",
                   "Vla","Vsa","Vlv","Vsv","hr","no","rs","D"]
    state_labels = state_eqn_names if len(state_eqn_names) == n_states else [f"state_{j+1}" for j in range(n_states)]

    mu_star_matrix = [] 
    state_png_paths = []

    for j in range(n_states):
        Y = Y_all_valid[:, j]
        
        
        
        Si = morris_analyze(
            problem, 
            
            X_valid, Y, conf_level=0.95, print_to_console=False, num_levels=num_levels)
        
        mu_star_matrix.append(np.asarray(Si["mu_star"], float))
      
        
        df = pd.DataFrame({
            "param": names,
            "mu": Si["mu"],# Signed mean (direction + magniture; cancels with sign flips)
            "mu_star": Si["mu_star"],# mean absolute effect (robust importance ranking)
            "mu_star_conf": Si["mu_star_conf"], # 95 percent CI
            "sigma": Si["sigma"],# spread of effects (nonlinear/interactions)
            ##  The bigger  -> more nonlinear/interactive
        }).sort_values("mu_star", ascending=False)
        
        df["rank"] = np.arange(1, len(df) + 1)

        df_plot=df

        # df_plot=df.head(10)
        
        fig_height = max(8, 0.18 * len(df_plot) + 2)
        fig, ax = plt.subplots(figsize=(8, fig_height))
        ax.barh(range(len(df_plot)), df_plot["mu_star"])
        ax.set_yticks(range(len(df_plot)))
        
        ax.set_yticklabels([f"{r}. {p}" for r, p in zip(df_plot["rank"], df_plot["param"])])
        ax.invert_yaxis()  # biggest on top
        
        ax.set_xlabel(r"$\mu^\ast$-importance")
        ax.set_title(f"{state_labels[j]} - Morris ")
        ax.tick_params(axis="y", labelsize=7)   # small labels so 106 fit
        ax.tick_params(axis="x", labelsize=8)
        
        fig.tight_layout()
        out_png = os.path.join(out_dir, f"{state_labels[j]}.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        
        state_png_paths.append(out_png)



        
       
    mu_star_matrix = np.vstack(mu_star_matrix)  # shape: (n_states, D)

    mu_star_mean   = np.nanmean(mu_star_matrix, axis=0)
    mu_star_median = np.nanmedian(mu_star_matrix, axis=0)
    mu_star_max    = np.nanmax(mu_star_matrix, axis=0)


    df_overall = pd.DataFrame({
        "param": names,
        "mu_star_mean": mu_star_mean,
        "mu_star_median": mu_star_median,
        "mu_star_max": mu_star_max,
    }).sort_values("mu_star_mean", ascending=False).reset_index(drop=False)

    # df_overall["rank"] = np.arange(1, len(df_overall) + 1)

    
    df_plot = df_overall.head(50).copy()
    df_plot["rank"] = np.arange(1, len(df_plot) + 1)

    fig_height = max(8, 0.18 * len(df_plot) + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(range(len(df_plot)), df_plot["mu_star_mean"])
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels([f"{r}. {p}" for r, p in zip(df_plot["rank"], df_plot["param"])])
    ax.invert_yaxis()

    ax.set_xlabel(r"Overall $\mu^\ast$ (mean across states)")
    ax.set_title("Morris Ranking - Log(Auc[y]) | 15%")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    out_png = os.path.join(out_dir, "Overall_ranked.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    pdf_path = os.path.join(out_dir, "Morris_Report.pdf")

    pages = [os.path.join(out_dir, "Overall_ranked.png")] + state_png_paths

    with PdfPages(pdf_path) as pdf:
        for img_path in pages:
            if not os.path.exists(img_path):
                continue

            img = mpimg.imread(img_path)

            fig = plt.figure(figsize=(11, 8.5))  # landscape letter-ish
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.imshow(img)
            pdf.savefig(fig, dpi=300)
            plt.close(fig)

    

    eng.quit()

if __name__ == "__main__":
    main()