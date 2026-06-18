# morris.py
# Morris sensitivity analysis updated to follow run_2.m data setup.
#
# Main changes from the old version:
#   1. Reads placebo_plotted_data(Survivor_Data).csv instead of DataCopeland.mat.
#   2. Builds the MATLAB data struct the same way as run_2.m:
#        data.hr, data.TNF, data.IL6, data.IL8, data.temp
#   3. Uses tspan = 0:0.1:50, matching run_2.m.
#   4. Computes cardiac output CO/Q from solved states and appends it as output 18.

import os
import matlab
import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SALib.sample.morris import sample as morris_sample
from SALib.analyze.morris import analyze as morris_analyze
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg



MATLAB_MODEL_DIR = "/Users/bryanbarrios/Desktop/MastersResearch/MathmaticalModeling/ODE_Model"

DATA_CSV = "placebo_plotted_data(Survivor_Data).csv"


T0 = 0.0
TF = 150
DT = 0.1

paraLog = True
rel_bound = 0.20          # plus/minus around each base parameter
n = 100                   # number of Morris trajectories
num_levels = 4
out_dir = "SensitivityAnalysis/Morris/5_22"
max_run_time_sec = 30


# -----------------------------
# Helpers
# -----------------------------

def matlab_quote(path: str) -> str:
    """Escape a Python string so it can be placed inside MATLAB single quotes."""
    return path.replace("'", "''")


def build_bounds(base, rel=0.20, eps=1e-12):
    """
    Build lower/upper bounds around each base parameter.
    Bounds are kept nonnegative because this model's parameters are expected to be nonnegative.
    """
    base = np.asarray(base, float).ravel()
    lo = np.maximum(0.0, base * (1.0 - rel))
    hi = base * (1.0 + rel)

    # If a parameter is exactly/near zero, give it a small positive upper range.
    hi = np.where((hi <= lo) & (np.abs(base) <= eps), lo + 1e-6, hi)

    return lo, hi


def ensure_col(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def append_cardiac_output(sol, pars):
    """
    Match run_2.m cardiac output calculation:

        Cla = pars(70);
        Clv = pars(72);
        Em  = pars(74);
        EM  = pars(75);

        Vla = sol(:,10);
        Vlv = sol(:,12);
        hr  = sol(:,14);

        pla  = Vla ./ Cla;
        plv  = Vlv ./ Clv;
        Vstr = -(pla./EM - plv./Em);
        Q    = ((1/6)*Vstr .* hr) / 60;

    Python uses 0-based indexing, so pars(70) -> pars[69], sol(:,10) -> sol[:,9].
    """
    pars = np.asarray(pars, float).ravel()
    sol = ensure_col(sol)

    if sol.shape[1] < 14:
        raise ValueError(f"Need at least 14 model states to compute CO/Q, got {sol.shape[1]}")
    if pars.size < 75:
        raise ValueError(f"Need at least 75 parameters to compute CO/Q, got {pars.size}")

    Cla = pars[69]
    Clv = pars[71]
    Em = pars[73]
    EM = pars[74]

    Vla = sol[:, 9]
    Vlv = sol[:, 11]
    hr = sol[:, 13]

    pla = Vla / Cla
    plv = Vlv / Clv
    Vstr = -(pla / EM - plv / Em)
    Q = ((1.0 / 6.0) * Vstr * hr) / 60.0

    return np.column_stack([sol, Q])


def make_tspan():
    """Create the same time grid as MATLAB 0:0.1:50."""
    return np.round(np.arange(T0, TF + 0.5 * DT, DT), 10)


def initialize_matlab_model(eng):
    """
    Start from the same data setup as run_2.m.

    run_2.m reads:
        HR       = T.HR_Mean;
        TEMP     = T.TEMP_Mean;
        TNF      = T.TNF_Mean;
        IL6      = T.IL_6_Mean;
        IL8      = T.IL_8_Mean;

    Then:
        data.hr   = HR;
        data.TNF  = TNF;
        data.IL6  = IL6;
        data.IL8  = IL8;
        data.temp = TEMP;
    """
    csv_path = DATA_CSV
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(MATLAB_MODEL_DIR, csv_path)

    csv_path_m = matlab_quote(csv_path)

    eng.eval(
        f"""
        clear data pars Init;

        csv_file = '{csv_path_m}';
        if ~isfile(csv_file)
            error('CSV file not found: %s', csv_file);
        end

        opts = detectImportOptions(csv_file, 'TreatAsMissing', {{'#DIV/0!'}});
        T = readtable(csv_file, opts);

        % Convert text-like table columns to numeric, same purpose as run_2.m.
        for j = 1:width(T)
            vname = T.Properties.VariableNames{{j}};
            col = T{{:, j}};

            if iscell(col) || isstring(col) || ischar(col)
                T.(vname) = str2double(string(col));
            end
        end

        HR   = T.HR_Mean;
        TEMP = T.TEMP_Mean;
        TNF  = T.TNF_Mean;
        IL6  = T.IL_6_Mean;
        IL8  = T.IL_8_Mean;

        data = struct();
        data.hr   = HR;
        data.TNF  = TNF;
        data.IL6  = IL6;
        data.IL8  = IL8;
        data.temp = TEMP;

        global Init ODE_TOL;
        ODE_TOL = 1e-8;

        [pars, Init] = load_pars_Init_Copeland_Edited(data);
        """,
        nargout=0,
    )

    base_pars = np.array(eng.workspace["pars"], float).ravel()
    init = eng.workspace["Init"]

    return base_pars, init


# -----------------------------
# Main Morris run
# -----------------------------

def main():
    print("Starting MATLAB")
    eng = matlab.engine.start_matlab()

    try:
        eng.addpath(MATLAB_MODEL_DIR, nargout=0)
        eng.eval(f"addpath(genpath('{matlab_quote(MATLAB_MODEL_DIR)}'));", nargout=0)
        eng.cd(MATLAB_MODEL_DIR, nargout=0)

        base_pars, Init = initialize_matlab_model(eng)
        k = base_pars.size

        par_names = [
            "k10", "k10m", "k6", "k6m", "k8", "k8m", "ktnf", "ktnfm", "kma", "kmpe",
            "kmr", "kpe", "x610", "x66", "x6tnf", "x810", "x8tnf", "x106", "xtnf10", "xtnf6",
            "xmpe", "xm10", "xmtnf", "h106", "h6tnf", "h66", "h610", "h8tnf", "h810", "htnf10",
            "htnf6", "hm10", "hmtnf", "hmpe", "stnf", "s10", "s8", "s6", "sm", "mmax",
            "k6tnf", "k8tnf", "k106", "kmtnf",
            "tau1", "TM", "Tm", "kt", "kttnf", "kt6", "kt10", "xttnf", "xt6", "xt10",
            "httnf", "ht6", "ht10",
            "tau2", "HM", "HI", "kh", "xht", "hht",
            "ppM", "kpepp", "kpp",
            "Ra", "Rv", "Rs", "Cla", "Csa", "Clv", "Csv", "Em", "EM",
            "knom", "kno", "xntnf", "xn10", "hntnf", "hn10",
            "krpp", "krno", "kr", "xrpp", "hrpp", "xhp", "hhp",
            "kpg", "peinf", "kpm", "xI10", "muno", "kpn",
            "kdn", "mud", "xdn", "alpha", "hmI10",
            "sM", "kmp",
            "kD", "xDam", "hmDa", "xm10D", "hm10D", "hmD", "knod", "xnDl10", "hnDl10", "xn10D", "hn10D", "ktnfhr","BPo",
        ]

        if len(par_names) != k:
            raise RuntimeError(f"par_names has length {len(par_names)}, but MATLAB pars has length {k}")

        # Kept from your original morris.py.
        # These are 1-based MATLAB parameter indices.
        ignore_pars = [
            10,12,103,104,109,110,59,60,46,47,35,37,38,40,67,68,69,70,71,72,73,74,75,90,114,
            
    
        ]
        ignore_pars_0 = [idx - 1 for idx in ignore_pars]

        active_mask = np.ones(k, dtype=bool)
        active_mask[ignore_pars_0] = False
        active_idx = np.where(active_mask)[0]

        lo, hi = build_bounds(base_pars, rel_bound)

        if paraLog:
            eps = 1e-12
            lo = np.maximum(lo, eps)
            hi = np.maximum(hi, lo * (1.0 + 1e-6))

            lo = np.log(lo)
            hi = np.log(hi)

        lo_perturb = lo[active_idx]
        hi_perturb = hi[active_idx]
        active_names = [par_names[i] for i in active_idx]

        problem = {
            "num_vars": len(active_idx),
            "names": active_names,
            "bounds": np.c_[lo_perturb, hi_perturb].tolist(),
        }

        X = morris_sample(problem, N=n, num_levels=num_levels, optimal_trajectories=None)
        n_runs = X.shape[0]
        D = problem["num_vars"]

        print(f"Morris sample: {n_runs} = N*(D+1), with N={n}, D={D}")
        print(f"Full parameter count k={k}; active parameter count D={D}")

        tspan_np = make_tspan()
        tspan = matlab.double(tspan_np.tolist())

        # Baseline run to detect number of outputs.
        print("Initializing the model to get number of model outputs")
        t0_m, sol0_m = eng.modelDriver(matlab.double(base_pars.tolist()), Init, tspan, nargout=2)
        t0 = np.array(t0_m).ravel()
        sol0 = ensure_col(np.array(sol0_m, float))
        sol0_plot = append_cardiac_output(sol0, base_pars)

        n_model_states = sol0.shape[1]
        n_outputs = sol0_plot.shape[1]

        print(f"Detected model states = {n_model_states}")
        print(f"Detected Morris outputs = {n_outputs} including appended CO/Q")

        # Evaluate all Morris samples.
        Y_all = np.full((n_runs, n_outputs), np.nan, dtype=float)
        print("Running model for all Morris samples")

        timed_out = 0
        failed = 0

        for i, row in enumerate(X, start=1):
            row = np.asarray(row, float)

            if paraLog:
                active_par_vals = np.exp(row)
            else:
                active_par_vals = row

            par_vec_full = base_pars.copy()
            par_vec_full[active_idx] = active_par_vals
            pars = matlab.double(par_vec_full.tolist())

            try:
                solving = eng.modelDriver(pars, Init, tspan, nargout=2, background=True)
                t_m, sol_m = solving.result(timeout=max_run_time_sec)
            except matlab.engine.TimeoutError:
                try:
                    solving.cancel()
                except Exception:
                    pass
                timed_out += 1
                print(f"  Run {i}/{n_runs} exceeded {max_run_time_sec} seconds, so ignored.")
                continue
            except Exception as e:
                failed += 1
                print(f"  Run {i}/{n_runs} failed with error: {e}; ignored.")
                continue

            t = np.array(t_m).ravel()
            sol = ensure_col(np.array(sol_m, float))

            if sol.shape[1] != n_model_states:
                failed += 1
                print(f"  Run {i}: state count changed ({sol.shape[1]} vs {n_model_states}), so ignored.")
                continue

            try:
                sol_plot = append_cardiac_output(sol, par_vec_full)
            except Exception as e:
                failed += 1
                print(f"  Run {i}: failed to compute CO/Q: {e}; ignored.")
                continue

            # Morris response = log(AUC) for each model output.
            auc = np.trapezoid(sol_plot, x=t, axis=0)
            log_auc = np.where(np.isfinite(auc) & (auc > 0.0), np.log(auc), np.nan)
            Y_all[i - 1, :] = log_auc

            if i % max(1, n_runs // 10) == 0:
                print(f"  {i}/{n_runs}")

        traj_size = D + 1
        if n_runs % traj_size != 0:
            raise RuntimeError(f"Sample size {n_runs} is not a multiple of traj_size={traj_size}")

        n_traj_total = n_runs // traj_size

        # A trajectory is valid only if all runs in that trajectory were valid.
        run_valid_mask = np.all(np.isfinite(Y_all), axis=1)

        traj_valid = np.ones(n_traj_total, dtype=bool)
        for traj_i in range(n_traj_total):
            start = traj_i * traj_size
            end = start + traj_size
            if not np.all(run_valid_mask[start:end]):
                traj_valid[traj_i] = False

        keep_mask = np.repeat(traj_valid, traj_size)

        n_valid_runs = int(keep_mask.sum())
        n_valid_traj = int(traj_valid.sum())

        print(f"total runs: {n_runs}")
        print(f"total trajectories: {n_traj_total}")
        print(f"valid trajectories used: {n_valid_traj}")
        print(f"valid runs used: {n_valid_runs}")
        print(f"timed out runs: {timed_out}")
        print(f"failed runs: {failed}")

        if n_valid_traj == 0:
            print("No valid Morris trajectories. Nothing to analyze.")
            return

        X_valid = X[keep_mask, :]
        Y_all_valid = Y_all[keep_mask, :]

        os.makedirs(out_dir, exist_ok=True)

        names = problem["names"]

        state_eqn_names = [
            "tnf", "il10", "il8", "il6", "Macrophages", "mr", "Pathogens", "temp", "pp",
            "Vla", "Vsa", "Vlv", "Vsv", "hr", "no", "rs", "Damage", "CO",
        ]
        state_labels = state_eqn_names if len(state_eqn_names) == n_outputs else [f"output_{j + 1}" for j in range(n_outputs)]

        mu_star_matrix = []
        state_png_paths = []

        for j in range(n_outputs):
            Y = Y_all_valid[:, j]

            Si = morris_analyze(
                problem,
                X_valid,
                Y,
                conf_level=0.95,
                print_to_console=False,
                num_levels=num_levels,
            )

            mu_star_matrix.append(np.asarray(Si["mu_star"], float))

            df = pd.DataFrame({
                "param": names,
                "mu": Si["mu"],
                "mu_star": Si["mu_star"],
                "mu_star_conf": Si["mu_star_conf"],
                "sigma": Si["sigma"],
            }).sort_values("mu_star", ascending=False)

            df["rank"] = np.arange(1, len(df) + 1)

            # Save numeric Morris results too.
            csv_out = os.path.join(out_dir, f"{state_labels[j]}_morris_results.csv")
            df.to_csv(csv_out, index=False)

            df_plot = df

            fig_height = max(8, 0.18 * len(df_plot) + 2)
            fig, ax = plt.subplots(figsize=(8, fig_height))
            ax.barh(range(len(df_plot)), df_plot["mu_star"])
            ax.set_yticks(range(len(df_plot)))
            ax.set_yticklabels([f"{r}. {p}" for r, p in zip(df_plot["rank"], df_plot["param"])])
            ax.invert_yaxis()

            ax.set_xlabel(r"$\mu^\ast$-importance")
            ax.set_title(f"{state_labels[j]} - Morris")
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelsize=8)

            fig.tight_layout()
            out_png = os.path.join(out_dir, f"{state_labels[j]}.png")
            fig.savefig(out_png, dpi=300)
            plt.close(fig)

            state_png_paths.append(out_png)

        mu_star_matrix = np.vstack(mu_star_matrix)

        mu_star_mean = np.nanmean(mu_star_matrix, axis=0)
        mu_star_median = np.nanmedian(mu_star_matrix, axis=0)
        mu_star_max = np.nanmax(mu_star_matrix, axis=0)

        df_overall = pd.DataFrame({
            "param": names,
            "mu_star_mean": mu_star_mean,
            "mu_star_median": mu_star_median,
            "mu_star_max": mu_star_max,
        }).sort_values("mu_star_mean", ascending=False).reset_index(drop=True)

        df_overall["rank"] = np.arange(1, len(df_overall) + 1)
        df_overall.to_csv(os.path.join(out_dir, "Overall_morris_results.csv"), index=False)

        df_plot = df_overall.head(50).copy()

        fig_height = max(8, 0.18 * len(df_plot) + 2)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.barh(range(len(df_plot)), df_plot["mu_star_mean"])
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels([f"{r}. {p}" for r, p in zip(df_plot["rank"], df_plot["param"])])
        ax.invert_yaxis()

        ax.set_xlabel(r"Overall $\mu^\ast$ (mean across outputs)")
        ax.set_title(f"Morris Ranking - Log(AUC[y]) | ±{int(rel_bound * 100)}%")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)

        fig.tight_layout()
        overall_png = os.path.join(out_dir, "Overall_ranked.png")
        fig.savefig(overall_png, dpi=300)
        plt.close(fig)

        pdf_path = os.path.join(out_dir, "Morris_Report.pdf")
        pages = [overall_png] + state_png_paths

        with PdfPages(pdf_path) as pdf:
            for img_path in pages:
                if not os.path.exists(img_path):
                    continue

                img = mpimg.imread(img_path)

                fig = plt.figure(figsize=(11, 8.5))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                ax.imshow(img)
                pdf.savefig(fig, dpi=300)
                plt.close(fig)

        print(f"Saved Morris figures, CSVs, and PDF report in: {out_dir}")
        print(f"PDF report: {pdf_path}")

    finally:
        try:
            eng.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
