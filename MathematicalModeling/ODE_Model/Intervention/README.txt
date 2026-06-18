ANTIMICROBIAL + NITRIC-OXIDE INTERVENTION

This package adds two independently controlled interventions to the MATLAB
ODE model.

1. Antimicrobial treatment, implemented as simple PK/PD

   dAbx/dt = doseRate(t) - kabx*Abx
   uPathogen = kabxpe*Abx/(xabxpe + Abx)
   dPe/dt = original Pe dynamics - uPathogen*Pe

2. Nitric-oxide inhibition, implemented as added NO clearance

   dNO/dt = original NO production - kno*NO - uNO(t)*NO

FILES

- model.m
- modelDriver.m
- load_pars_Init_Copeland_Edited.m
- run_antimicrobial_no_intervention.m
- run_antimicrobial_no_intervention.py
- rlModelStep.m
- run_mechanistic_rl_antibiotic.py

MATLAB

Place all files in the same folder and run:

    run_antimicrobial_no_intervention

PYTHON LAUNCHER

    python run_antimicrobial_no_intervention.py

The experiment compares no treatment, antimicrobial only, NO inhibition only,
and the combined treatment. It plots Pe, NO, Ma, and D and writes a CSV table.

EDIT THESE SETTINGS IN run_antimicrobial_no_intervention.m

    pathogenStart
    pathogenEnd
    abxDoseRate
    noStart
    noEnd
    noRate

EDIT THESE PK/PD PARAMETERS IN load_pars_Init_Copeland_Edited.m

    kabx
    kabxpe
    xabxpe

The supplied values are simulation examples, not clinical doses.

FUTURE RL ACTION

PYTORCH MECHANISTIC RL

The PyTorch DQN controller uses the existing MATLAB ODE model as the
mechanistic environment. The Python code does not rewrite the equations.
Instead, it calls:

    rlModelStep.m -> modelDriver.m -> model.m

At every decision interval, the DQN agent chooses an antibiotic dose-rate
action:

    u(t) in {0, 3.5, 7, 14}

MATLAB integrates the ODE over that interval, including the Abx PK state:

    dAbx/dt = u(t) - kabx*Abx

The reward penalizes pathogen burden, inflammation, damage, antibiotic dose,
and antibiotic concentration exposure. After treatment, the script turns
antibiotics off and checks for sustained recovery without rebound.

Run:

    python3 run_mechanistic_rl_antibiotic.py

Useful options:

    python3 run_mechanistic_rl_antibiotic.py --episodes 150
    python3 run_mechanistic_rl_antibiotic.py --actions 0,3.5,7,14
    python3 run_mechanistic_rl_antibiotic.py --randomize-initial-pe

Outputs:

    mechanistic_rl_training_summary.csv
    mechanistic_rl_policy_results.csv
    mechanistic_rl_antibiotic_policy.png
    mechanistic_rl_dqn_policy.pt

Requirement:

    MATLAB Engine API for Python must be installed for the same Python
    executable used to run the script.
