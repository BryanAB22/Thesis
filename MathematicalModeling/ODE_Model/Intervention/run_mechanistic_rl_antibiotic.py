"""PyTorch RL controller that calls the MATLAB sepsis ODE model.

The ODE equations stay in the existing .m files. Python only provides the RL
loop: a DQN agent chooses antibiotic dose-rate actions, MATLAB integrates the
mechanistic model for one decision interval, and the agent is rewarded for
sustained recovery with minimum antibiotic exposure.

Requirements:
  - MATLAB with the MATLAB Engine API for Python installed
  - PyTorch, NumPy, and Matplotlib
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


TNF, IL10, CXCL8, IL6, MA, MR, PE, TEMP, PAIN = range(9)
VLA, VSA, VLV, VSV, HR, NO, RS, DAMAGE, ABX = range(9, 18)


@dataclass(frozen=True)
class Thresholds:
    pe_clear: float = 0.01
    ma_safe: float = 0.05
    damage_safe: float = 0.10
    pe_rebound: float = 0.05
    max_damage: float = 100.0


class MatlabModelClient:
    """Persistent MATLAB Engine client for the existing .m model."""

    def __init__(self, folder: Path, matlab_app: Path | None = None) -> None:
        matlab_bin = find_matlab_library_folder(matlab_app)
        if matlab_bin is not None:
            prepend_dynamic_library_path(matlab_bin)

        try:
            import matlab.engine  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MATLAB Engine for Python is not installed. Install it from "
                "your MATLAB installation, for example:\n"
                "  cd /Applications/MATLAB_R20XXx.app/extern/engines/python\n"
                "  python3 -m pip install ."
            ) from exc

        self.matlab = __import__("matlab")
        self.engine = matlab.engine.start_matlab()
        self.engine.addpath(str(folder), nargout=0)

    def load_initial_state(self) -> Tuple[object, np.ndarray]:
        data = self.engine.struct()
        pars, init = self.engine.load_pars_Init_Copeland_Edited(data, nargout=2)
        return pars, np.asarray(init, dtype=float).reshape(-1)

    def step(
        self,
        pars: object,
        state: np.ndarray,
        start_time: float,
        dt: float,
        dose_rate: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        state_mat = self.matlab.double(state.reshape(1, -1).tolist())
        t_mat, sol_mat = self.engine.rlModelStep(
            pars,
            state_mat,
            float(start_time),
            float(start_time + dt),
            float(dose_rate),
            nargout=2,
        )
        t = np.asarray(t_mat, dtype=float).reshape(-1)
        sol = np.asarray(sol_mat, dtype=float)
        return t, sol

    def close(self) -> None:
        self.engine.quit()


def find_matlab_library_folder(matlab_app: Path | None) -> Path | None:
    if matlab_app is not None:
        candidate = matlab_app.expanduser().resolve()
        if candidate.name.endswith(".app"):
            candidate = candidate / "bin" / "maca64"
        elif candidate.name in {"matlab", "MATLAB"}:
            candidate = candidate.parent.parent / "bin" / "maca64"
        return candidate if candidate.is_dir() else None

    candidates = sorted(
        glob.glob(str(Path.home() / "Desktop" / "MATLAB_R*.app"))
        + glob.glob("/Applications/MATLAB_R*.app"),
        reverse=True,
    )
    for app in candidates:
        library_folder = Path(app) / "bin" / "maca64"
        if library_folder.is_dir():
            return library_folder.resolve()
    return None


def prepend_dynamic_library_path(folder: Path) -> None:
    # MATLAB R2025a bundles its own Abseil libraries. Put them first so the
    # Engine does not accidentally load incompatible Homebrew copies.
    for variable in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        existing = os.environ.get(variable, "")
        paths = [str(folder)]
        if existing:
            paths.extend(path for path in existing.split(os.pathsep) if path and path != str(folder))
        os.environ[variable] = os.pathsep.join(paths)


def reexec_with_matlab_libraries_if_needed(folder: Path | None) -> None:
    """Restart on macOS so dyld sees MATLAB libraries before Python imports."""
    if folder is None or platform.system() != "Darwin":
        return

    folder_text = str(folder)
    needs_reexec = False
    env = os.environ.copy()
    for variable in ("DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        existing = env.get(variable, "")
        current_paths = [path for path in existing.split(os.pathsep) if path]
        if not current_paths or current_paths[0] != folder_text:
            needs_reexec = True
            env[variable] = os.pathsep.join(
                [folder_text] + [path for path in current_paths if path != folder_text]
            )

    if needs_reexec:
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.storage: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        transition = (
            observation.astype(np.float32),
            int(action),
            float(reward),
            next_observation.astype(np.float32),
            bool(done),
        )
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        observations, actions, rewards, next_observations, dones = zip(
            *(self.storage[int(index)] for index in indices)
        )
        return (
            np.vstack(observations).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.vstack(next_observations).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class DQNAgent:
    def __init__(
        self,
        actions: Sequence[float],
        rng: np.random.Generator,
        seed: int,
        gamma: float = 0.96,
        learning_rate: float = 2e-3,
        batch_size: int = 64,
        replay_capacity: int = 20000,
    ) -> None:
        self.actions = tuple(float(action) for action in actions)
        self.rng = rng
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        self.observation_size = 5
        self.policy_net = QNetwork(self.observation_size, len(self.actions)).to(self.device)
        self.target_net = QNetwork(self.observation_size, len(self.actions)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(replay_capacity, rng)

    def observe(self, state: np.ndarray, time: float, max_time: float) -> np.ndarray:
        return np.array(
            [
                math.log1p(max(state[PE], 0.0)) / math.log1p(20.0),
                math.log1p(max(state[MA], 0.0)) / math.log1p(30000.0),
                math.log1p(max(state[DAMAGE], 0.0)) / math.log1p(100.0),
                math.log1p(max(state[ABX], 0.0)) / math.log1p(40.0),
                min(max(time / max_time, 0.0), 1.0),
            ],
            dtype=np.float32,
        )

    def choose_action(self, observation: np.ndarray, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, len(self.actions)))
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs).squeeze(0).cpu().numpy()
        best = np.flatnonzero(np.isclose(q_values, np.max(q_values)))
        return int(self.rng.choice(best))

    def remember(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.add(observation, action, reward, next_observation, done)

    def learn(self, updates: int) -> float:
        if len(self.replay) < self.batch_size:
            return math.nan

        last_loss = math.nan
        for _ in range(updates):
            observations, actions, rewards, next_observations, dones = self.replay.sample(self.batch_size)
            obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            act = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
            rew = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            next_obs = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
            done = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

            q_values = self.policy_net(obs).gather(1, act).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_obs).max(dim=1).values
                target = rew + self.gamma * (1.0 - done) * next_q

            loss = self.loss_fn(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        return last_loss

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


class AntibioticEnvironment:
    def __init__(
        self,
        matlab_client: MatlabModelClient,
        pars: object,
        initial_state: np.ndarray,
        thresholds: Thresholds,
        dt: float,
        max_treatment_time: float,
        followup: float,
        actions: Sequence[float],
    ) -> None:
        self.matlab_client = matlab_client
        self.pars = pars
        self.base_initial_state = initial_state.copy()
        self.thresholds = thresholds
        self.dt = dt
        self.max_treatment_time = max_treatment_time
        self.followup = followup
        self.actions = tuple(float(action) for action in actions)
        self.reset()

    def reset(self, initial_pe: float | None = None) -> np.ndarray:
        self.time = 0.0
        self.state = self.base_initial_state.copy()
        if initial_pe is not None:
            self.state[PE] = initial_pe
        self.history_t = [self.time]
        self.history_y = [self.state.copy()]
        self.history_actions: List[float] = []
        self.dose_exposure = 0.0
        self.abx_auc = 0.0
        self.positive_dose_steps = 0
        return self.state.copy()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool]:
        dose_rate = self.actions[action_index]
        t_step, y_step = self.matlab_client.step(
            self.pars,
            self.state,
            self.time,
            self.dt,
            dose_rate,
        )
        next_state = y_step[-1].copy()

        pe_auc = float(np.trapezoid(y_step[:, PE], t_step))
        ma_auc = float(np.trapezoid(y_step[:, MA], t_step))
        damage_auc = float(np.trapezoid(y_step[:, DAMAGE], t_step))
        abx_auc = float(np.trapezoid(y_step[:, ABX], t_step))
        dose_exposure = dose_rate * self.dt

        reward = (
            -0.40 * math.log1p(pe_auc)
            -0.03 * math.log1p(ma_auc)
            -0.08 * math.log1p(damage_auc)
            -0.04 * dose_exposure
            -0.01 * abx_auc
        )
        failed = bool(np.max(y_step[:, DAMAGE]) >= self.thresholds.max_damage)
        if failed:
            reward -= 100.0

        self.time += self.dt
        self.state = next_state
        self.history_t.extend(t_step[1:].tolist())
        self.history_y.extend(y_step[1:])
        self.history_actions.append(dose_rate)
        self.dose_exposure += dose_exposure
        self.abx_auc += abx_auc
        if dose_rate > 0:
            self.positive_dose_steps += 1

        return next_state, reward, failed

    def trajectory(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.history_t), np.vstack(self.history_y)


def sustained_recovery(env: AntibioticEnvironment, state: np.ndarray, current_time: float) -> Tuple[bool, np.ndarray, np.ndarray]:
    t_follow, y_follow = env.matlab_client.step(
        env.pars,
        state,
        current_time,
        env.followup,
        0.0,
    )
    pe_ok = np.max(y_follow[:, PE]) <= env.thresholds.pe_rebound
    ma_ok = y_follow[-1, MA] <= env.thresholds.ma_safe
    damage_ok = y_follow[-1, DAMAGE] <= env.thresholds.damage_safe
    no_failure = np.max(y_follow[:, DAMAGE]) < env.thresholds.max_damage
    return bool(pe_ok and ma_ok and damage_ok and no_failure), t_follow, y_follow


def epsilon_for_episode(episode: int, episodes: int) -> float:
    epsilon_start = 0.90
    epsilon_end = 0.05
    decay = max(episodes / 3.0, 1.0)
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / decay)


def train_agent(
    env: AntibioticEnvironment,
    agent: DQNAgent,
    episodes: int,
    rng: np.random.Generator,
    randomize_initial_pe: bool,
) -> List[Dict[str, float | bool]]:
    summaries: List[Dict[str, float | bool]] = []
    steps_per_episode = int(env.max_treatment_time // env.dt)

    for episode in range(episodes):
        initial_pe = float(rng.uniform(0.15, 3.0)) if randomize_initial_pe else None
        state = env.reset(initial_pe=initial_pe)
        episode_transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        failed = False
        loss = math.nan

        for _ in range(steps_per_episode):
            observation = agent.observe(state, env.time, env.max_treatment_time)
            action = agent.choose_action(observation, epsilon_for_episode(episode, episodes))
            next_state, reward, failed = env.step(action)
            next_observation = agent.observe(next_state, env.time, env.max_treatment_time)
            episode_transitions.append((observation, action, reward, next_observation, failed))
            state = next_state
            if failed:
                break

        success = False
        if not failed:
            success, _, _ = sustained_recovery(env, env.state, env.time)

        terminal_reward = 180.0 if success else -80.0
        terminal_reward -= 0.03 * env.dose_exposure
        if episode_transitions:
            obs, action, reward, next_obs, _ = episode_transitions[-1]
            episode_transitions[-1] = (obs, action, reward + terminal_reward, next_obs, True)

        for transition in episode_transitions:
            agent.remember(*transition)
        loss = agent.learn(updates=max(1, len(episode_transitions)))

        if (episode + 1) % 10 == 0:
            agent.update_target()

        summaries.append(
            {
                "episode": episode + 1,
                "success": success,
                "failed": failed,
                "dose_exposure": env.dose_exposure,
                "abx_auc": env.abx_auc,
                "positive_dose_steps": env.positive_dose_steps,
                "final_pe": float(env.state[PE]),
                "final_ma": float(env.state[MA]),
                "final_damage": float(env.state[DAMAGE]),
                "loss": loss,
            }
        )

        if (episode + 1) % 10 == 0:
            recent = summaries[-10:]
            success_rate = sum(1 for row in recent if row["success"]) / len(recent)
            print(
                f"Episode {episode + 1:4d}/{episodes}: "
                f"recent success={success_rate:.0%}, "
                f"last exposure={env.dose_exposure:.3f}, loss={loss:.4g}"
            )

    return summaries


def evaluate_greedy_policy(env: AntibioticEnvironment, agent: DQNAgent) -> Dict[str, object]:
    state = env.reset()
    steps = int(env.max_treatment_time // env.dt)
    recovered = False
    follow_t = None
    follow_y = None

    for _ in range(steps):
        observation = agent.observe(state, env.time, env.max_treatment_time)
        action = agent.choose_action(observation, epsilon=0.0)
        state, _, failed = env.step(action)
        if failed:
            break

        success, candidate_t, candidate_y = sustained_recovery(env, env.state, env.time)
        if success:
            recovered = True
            follow_t = candidate_t
            follow_y = candidate_y
            break

    if follow_t is None or follow_y is None:
        recovered, follow_t, follow_y = sustained_recovery(env, env.state, env.time)

    t_policy, y_policy = env.trajectory()
    positive_action_times = [
        index * env.dt for index, dose_rate in enumerate(env.history_actions) if dose_rate > 0
    ]
    if positive_action_times:
        treatment_duration = positive_action_times[-1] - positive_action_times[0] + env.dt
        first_dose_time = positive_action_times[0]
        last_dose_time = positive_action_times[-1] + env.dt
    else:
        treatment_duration = 0.0
        first_dose_time = math.nan
        last_dose_time = math.nan

    return {
        "success": recovered,
        "t_policy": t_policy,
        "y_policy": y_policy,
        "t_follow": follow_t,
        "y_follow": follow_y,
        "actions": list(env.history_actions),
        "dose_exposure": env.dose_exposure,
        "abx_auc": env.abx_auc,
        "treatment_duration": treatment_duration,
        "first_dose_time": first_dose_time,
        "last_dose_time": last_dose_time,
        "final_pe": float(follow_y[-1, PE]),
        "final_ma": float(follow_y[-1, MA]),
        "final_damage": float(follow_y[-1, DAMAGE]),
    }


def constant_duration_search(
    env: AntibioticEnvironment,
    dose_rate: float,
) -> Dict[str, float | bool]:
    candidate_durations = np.arange(0.0, env.max_treatment_time + env.dt, env.dt)
    zero_action_index = int(np.argmin(np.abs(np.asarray(env.actions) - 0.0)))
    treatment_action_index = int(np.argmin(np.abs(np.asarray(env.actions) - dose_rate)))
    for duration in candidate_durations:
        state = env.reset()
        while env.time < env.max_treatment_time - 1e-9:
            action_index = zero_action_index
            if env.time < duration - 1e-9:
                action_index = treatment_action_index
            state, _, failed = env.step(action_index)
            if failed:
                break

        success, _, y_follow = sustained_recovery(env, state, env.time)
        if success:
            return {
                "success": True,
                "duration": float(duration),
                "dose_rate": float(dose_rate),
                "dose_exposure": float(env.dose_exposure),
                "abx_auc": float(env.abx_auc),
                "final_pe": float(y_follow[-1, PE]),
                "final_ma": float(y_follow[-1, MA]),
                "final_damage": float(y_follow[-1, DAMAGE]),
            }
    return {"success": False}


def write_training_summary(path: Path, summaries: Iterable[Dict[str, float | bool]]) -> None:
    rows = list(summaries)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_policy_results(path: Path, evaluation: Dict[str, object], constant_best: Dict[str, float | bool]) -> None:
    rows = [
        {
            "method": "PyTorch DQN greedy policy",
            "success": evaluation["success"],
            "treatment_duration": evaluation["treatment_duration"],
            "dose_exposure": evaluation["dose_exposure"],
            "abx_auc": evaluation["abx_auc"],
            "first_dose_time": evaluation["first_dose_time"],
            "last_dose_time": evaluation["last_dose_time"],
            "final_pe": evaluation["final_pe"],
            "final_ma": evaluation["final_ma"],
            "final_damage": evaluation["final_damage"],
        },
        {
            "method": "constant-dose duration search",
            "success": constant_best.get("success", False),
            "treatment_duration": constant_best.get("duration", math.nan),
            "dose_exposure": constant_best.get("dose_exposure", math.nan),
            "abx_auc": constant_best.get("abx_auc", math.nan),
            "first_dose_time": 0.0 if constant_best.get("success", False) else math.nan,
            "last_dose_time": constant_best.get("duration", math.nan),
            "final_pe": constant_best.get("final_pe", math.nan),
            "final_ma": constant_best.get("final_ma", math.nan),
            "final_damage": constant_best.get("final_damage", math.nan),
        },
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_evaluation(path: Path, evaluation: Dict[str, object], dt: float) -> None:
    t_policy = evaluation["t_policy"]
    y_policy = evaluation["y_policy"]
    t_follow = evaluation["t_follow"]
    y_follow = evaluation["y_follow"]
    actions = evaluation["actions"]

    figure, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()
    plot_specs = [
        (PE, "Pathogen Pe"),
        (MA, "Activated macrophages Ma"),
        (DAMAGE, "Damage D"),
        (ABX, "Antibiotic Abx"),
        (NO, "Nitric oxide NO"),
    ]
    for axis, (state_index, label) in zip(axes, plot_specs):
        axis.plot(t_policy, y_policy[:, state_index], label="DQN treatment")
        axis.plot(t_follow, y_follow[:, state_index], "--", label="antibiotics-off follow-up")
        axis.set_xlabel("Time")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    action_axis = axes[-1]
    action_times = np.arange(len(actions)) * dt
    action_axis.step(action_times, actions, where="post")
    action_axis.set_xlabel("Time")
    action_axis.set_ylabel("Dose input u(t)")
    action_axis.set_title("Learned antibiotic actions")
    action_axis.grid(True, alpha=0.3)

    figure.suptitle("PyTorch DQN policy calling the MATLAB ODE model")
    figure.savefig(path, dpi=300)
    plt.close(figure)


def parse_actions(action_text: str) -> Tuple[float, ...]:
    actions = tuple(float(part.strip()) for part in action_text.split(",") if part.strip())
    if not actions:
        raise ValueError("At least one action is required.")
    if min(actions) < 0:
        raise ValueError("Dose actions must be nonnegative.")
    if 0.0 not in actions:
        actions = (0.0,) + actions
    return actions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--dt", type=float, default=6.0, help="RL decision interval.")
    parser.add_argument("--max-treatment-time", type=float, default=120.0)
    parser.add_argument("--followup", type=float, default=180.0)
    parser.add_argument(
        "--actions",
        default="0,3.5,7,14",
        help="Comma-separated antibiotic dose-rate actions.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--randomize-initial-pe", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--matlab-app",
        default=None,
        help="Optional path to MATLAB_R*.app or the MATLAB executable.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    actions = parse_actions(args.actions)
    folder = Path(__file__).resolve().parent
    matlab_app = Path(args.matlab_app) if args.matlab_app else None
    reexec_with_matlab_libraries_if_needed(find_matlab_library_folder(matlab_app))
    matlab_client = MatlabModelClient(folder, matlab_app=matlab_app)

    try:
        pars, initial = matlab_client.load_initial_state()
        thresholds = Thresholds()
        env = AntibioticEnvironment(
            matlab_client=matlab_client,
            pars=pars,
            initial_state=initial,
            thresholds=thresholds,
            dt=args.dt,
            max_treatment_time=args.max_treatment_time,
            followup=args.followup,
            actions=actions,
        )
        agent = DQNAgent(actions=actions, rng=rng, seed=args.seed)

        summaries = train_agent(
            env=env,
            agent=agent,
            episodes=args.episodes,
            rng=rng,
            randomize_initial_pe=args.randomize_initial_pe,
        )
        evaluation = evaluate_greedy_policy(env, agent)
        constant_best = constant_duration_search(env, dose_rate=max(actions))

        training_file = folder / "mechanistic_rl_training_summary.csv"
        results_file = folder / "mechanistic_rl_policy_results.csv"
        figure_file = folder / "mechanistic_rl_antibiotic_policy.png"
        model_file = folder / "mechanistic_rl_dqn_policy.pt"

        write_training_summary(training_file, summaries)
        write_policy_results(results_file, evaluation, constant_best)
        torch.save(agent.policy_net.state_dict(), model_file)
        if not args.no_plot:
            plot_evaluation(figure_file, evaluation, args.dt)

        recent = summaries[-min(20, len(summaries)) :]
        recent_success = sum(1 for row in recent if row["success"]) / max(len(recent), 1)

        print("PyTorch DQN antibiotic controller calling MATLAB model")
        print(f"Actions: {actions}")
        print(f"Episodes: {args.episodes}")
        print(f"Recent training success rate: {recent_success:.2%}")
        print("")
        print("Greedy DQN policy:")
        print(f"  Sustained recovery: {evaluation['success']}")
        print(f"  Treatment duration: {evaluation['treatment_duration']:.3f}")
        print(f"  Dose exposure integral: {evaluation['dose_exposure']:.3f}")
        print(f"  Antibiotic concentration AUC: {evaluation['abx_auc']:.3f}")
        print(
            "  Final Pe/Ma/D after follow-up: "
            f"{evaluation['final_pe']:.4g}, "
            f"{evaluation['final_ma']:.4g}, "
            f"{evaluation['final_damage']:.4g}"
        )
        print("")
        print("Constant-dose duration search:")
        if constant_best.get("success", False):
            print(f"  Minimum duration at dose {constant_best['dose_rate']}: {constant_best['duration']:.3f}")
            print(f"  Dose exposure integral: {constant_best['dose_exposure']:.3f}")
        else:
            print("  No successful duration found in the configured horizon.")
        print("")
        print(f"Saved training summary: {training_file}")
        print(f"Saved policy results: {results_file}")
        print(f"Saved DQN weights: {model_file}")
        if not args.no_plot:
            print(f"Saved policy figure: {figure_file}")
    finally:
        matlab_client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
