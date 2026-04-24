# **PPO-Driven Swarm Control**  

### End-to-End Rebuild of a Hybrid Multi-Robot Coverage Pipeline

### Motivation

This notebook develops a complete **end-to-end hybrid swarm-control pipeline** for vegetation-aware multi-agent exploration using satellite imagery. The central idea is simple but strong:

> **Use reinforcement learning to learn good local motion, and use classical multi-robot systems theory to impose swarm-level structure.**

That means this notebook is **not** treating reinforcement learning as the entire swarm solution. Instead, it treats PPO as the **microscopic local controller**, and then augments it with **potential-field shaping**, **graph-based consensus**, and **stochastic role switching** so that the final multi-agent behavior is scalable, interpretable, and more aligned with classical multi-robot systems theory.

The project begins with a user-provided **RGB satellite image**, converts it into a **vegetation-utility field** using a **VARI-based vegetation proxy**, trains or loads a PPO agent on local image crops, and then lifts that learned policy into a decentralized swarm setting. The final result is a hybrid controller capable of producing richer collective behaviors than raw PPO replication alone.

---

# 1. Abstract

This notebook develops a **hybrid multi-robot coverage framework** that combines **reinforcement learning** with **classical multi-robot systems (MRS) theory**. A satellite RGB image is transformed into a normalized scalar utility map using the **Visible Atmospherically Resistant Index (VARI)**, which serves as an RGB-only vegetation proxy when near-infrared measurements are unavailable. A single-agent **Proximal Policy Optimization (PPO)** policy is trained on **local $128 \times 128$** utility crops with a **first-visit reward**, producing a decentralized learned navigation primitive that prefers informative, vegetation-rich regions while discouraging revisits.

To scale this learned policy to a swarm, the notebook introduces three additional layers:

1. **Artificial potential fields** for collision avoidance, revisit repulsion, boundary handling, and utility-gradient attraction,
2. **Graph-based consensus dynamics** over a time-varying proximity graph for coordination and local disagreement reduction,
3. **CRN-inspired stochastic role switching** to induce adaptive heterogeneity among explorers, surveyors, defenders, and idle/recovery agents.

The final action of each robot is a **role-conditioned hybrid combination** of PPO, potential-field, and consensus terms. The framework is evaluated using **coverage ratio**, **utility gain**, **redundancy**, **consensus error**, and **spatial dispersion**, with comparisons across multiple controller regimes ranging from raw PPO to the full hybrid architecture.

# 2. Core Concept and Project Positioning

## 2.1 What this project is

This project is a **hierarchical hybrid swarm-control system** in which:

- the environment is represented as a scalar field $\phi(x,y)$,
- each agent sees only a **local crop** of that field,
- a PPO policy proposes a local motion tendency,
- classical swarm-control layers modify and regularize that tendency,
- the swarm collectively explores the field with better spacing, lower redundancy, and stronger coordination.

## 2.2 What this project is *not*

This is **not** a vague “RL-for-swarms” demo.
It does **not** claim that PPO alone solves scalable swarm coordination.
It does **not** claim global optimality or a complete proof of closed-loop convergence for the full hybrid system.

Instead, the honest claim is:

- **RL learns local instinct**
- **classical MRS theory enforces structure**
- **hybridization is the real contribution**

## 2.3 Application framing

Primary application narrative:

- **precision agriculture**
- **vegetation-aware multi-UAV coverage**
- **distributed inspection of scalar utility fields**

Secondary application narratives:

- environmental monitoring
- search and exploration
- utility-driven distributed inspection
- mapping-inspired multi-agent exploration

# 3. Research Idea in One Sentence

We learn a decentralized local controller from image patches using PPO, then augment it with interpretable swarm mechanisms so that **local intelligence + macroscopic coordination** produces better multi-agent coverage than raw policy replication.

# 4. Theoretical Foundations

This notebook is deliberately aligned with core MRS themes:

- **coverage in space and time**
- **agreement / consensus**
- **division of labor**
- **goal-oriented motion**
- **pattern formation**
- **decentralized control under local sensing**
- **graph-based interaction models**
- **potential-field-based coverage**
- **stochastic role/task transitions**

The architecture is best understood as a synthesis of:

1. **single-integrator mobile robot dynamics**,  
2. **scalar-field exploration**,  
3. **graph consensus**,  
4. **artificial potential functions**,  
5. **stochastic task allocation / role evolution**,  
6. **geometric swarm analysis**.

# 5. Mathematical Model

## 5.1 Environment as a scalar utility field

Let the user provide an RGB satellite image

$$
I(x,y) = \big(R(x,y), G(x,y), B(x,y)\big), \qquad R,G,B \in [0,1].
$$

Since we only assume RGB imagery, we do **not** compute true NDVI.
Instead, we compute the **VARI-based vegetation proxy**

$$
\text{VARI}(x,y) = \frac{G(x,y) - R(x,y)}{G(x,y) + R(x,y) - B(x,y) + \varepsilon},
$$

where $\varepsilon > 0$ prevents numerical instability.

This is then clipped and normalized into a utility field

$$
\phi(x,y) = \frac{\text{VARI}(x,y) - \text{VARI}_{\min}}
{\text{VARI}_{\max} - \text{VARI}_{\min} + \varepsilon},
\qquad \phi(x,y) \in [0,1].
$$

### Interpretation of $\phi(x,y)$

- high $\phi$: vegetation-rich / informative region  
- low $\phi$: low utility region  
- $\phi$ acts simultaneously as:
  - reward landscape,
  - perceptual substrate,
  - swarm-level coverage objective.

### Optional preprocessing

To stabilize downstream behavior, the pipeline may apply:

- Gaussian smoothing,
- percentile clipping,
- contrast normalization,
- downsampling,
- optional masking of text/borders/legends.

## 5.2 Agent dynamics

Each robot is modeled as a 2D point agent with **discrete-time single-integrator dynamics**

$$
p_i(k+1) = p_i(k) + \Delta t\, u_i(k),
$$

where:

- $p_i(k) \in \mathbb{R}^2$: position of agent $i$ at step $k$,
- $u_i(k) \in \mathbb{R}^2$: control input / velocity command,
- $\Delta t$: simulation timestep.

This is the correct abstraction for a swarm-layer notebook because it is simple, interpretable, and standard in multi-robot coordination theory.

## 5.3 Local observation model

Each agent does **not** observe the full map.
Instead, agent $i$ at position $p_i(k)$ receives a **local crop**

$$
o_i(k) \in \mathbb{R}^{1 \times P \times P},
$$

with $P=128$ by default.

This design is important because it:

- enforces **partial observability**,
- makes PPO genuinely **decentralized**,
- preserves a realistic gap between local sensing and global analysis.

## 5.4 Reward design for PPO

Let $V$ denote the visit map.
If the agent reaches cell $c_k$ at time step $k$, define

$$
r(k) =
\begin{cases}
\phi(c_k), & \text{if } c_k \text{ is visited for the first time},\\
0, & \text{otherwise}.
\end{cases}
$$

This reward is elegant because it jointly encourages:

- exploration,
- preference for useful regions,
- reduction of trivial revisits,
- implicit coverage behavior.

The baseline reward remains **first-visit utility reward**.  
That is the cleanest reward for the first version.

## 5.5 PPO microscopic controller

Let the trained PPO policy be

$$
\pi_\theta(o_i(k)) = a_i^{\text{ppo}}(k),
$$

where the discrete action is one of:

$$
\{\text{up}, \text{right}, \text{down}, \text{left}\}.
$$

For hybrid composition, this discrete action is mapped into a continuous direction vector

$$
d_i^{\text{ppo}}(k) \in \mathbb{R}^2.
$$

Then the PPO contribution to motion becomes

$$
u_i^{\text{ppo}}(k) = \alpha_{\text{ppo}}(i,k)\, d_i^{\text{ppo}}(k),
$$

where $\alpha_{\text{ppo}}(i,k)$ may depend on role and local context.

## 5.6 Proximity graph and consensus layer

At each timestep, define a time-varying proximity graph

$$
G(k) = (V, E(k)),
$$

with edge rule

$$
(i,j) \in E(k)
\quad \Longleftrightarrow \quad
\|p_i(k) - p_j(k)\| \le R_{\text{comm}}.
$$

Let:

- $A(k)$: adjacency matrix  
- $D(k)$: degree matrix  
- $L(k) = D(k) - A(k)$: graph Laplacian  

A classical consensus-style term is

$$
u_i^{\text{cons}}(k)
= k_{\text{cons}} \sum_{j \in \mathcal{N}_i(k)} \big(p_j(k)-p_i(k)\big).
$$

However, raw position consensus can collapse the swarm, so it must be used carefully.  
A more coverage-friendly alternative is to run consensus on a local information state \(y_i\), such as:

- local utility imbalance,
- frontier score,
- local under-coverage estimate,
- preferred heading.

Then

$$
y_i(k+1)
=
y_i(k) -
\epsilon
\sum_{j\in \mathcal{N}_i(k)}
\big(y_i(k)-y_j(k)\big).
$$

The role of consensus here is **coordination smoothing**, not rendezvous.

## 5.7 Potential-field layer

Define a potential-field contribution

$$
u_i^{\text{pf}} = F_i^{\text{att}} + F_i^{\text{rep}} + F_i^{\text{visit}} + F_i^{\text{bnd}}.
$$

### (a) Utility-gradient attraction

$$
F_i^{\text{att}} = k_{\text{att}} \nabla \phi(p_i).
$$

This encourages motion toward locally increasing utility.

### (b) Inter-agent repulsion

For neighbors inside repulsion radius \(R_{\text{rep}}\),

$$
F_i^{\text{rep}}
= \sum_j
k_{\text{rep}}\,
\psi(\|p_i-p_j\|)
\frac{p_i-p_j}{\|p_i-p_j\|+\varepsilon},
$$

where a simple choice is

$$
\psi(r) = \max\left(0,\frac{1}{r}-\frac{1}{R_{\text{rep}}}\right).
$$

This discourages clustering and near-collision behavior.

### (c) Visited-region repulsion

Let $M_{\text{visit}}$ be a visit-density map.
Then

$$
F_i^{\text{visit}} = -k_{\text{visit}} \nabla M_{\text{visit}}(p_i),
$$

which pushes agents away from heavily revisited regions.

### (d) Boundary repulsion

A soft inward force is added near image boundaries to prevent pathological edge-sticking without relying only on hard clipping.

## 5.8 CRN-inspired stochastic role switching

Each agent has a role

$$
\text{role}_i(k) \in
\{\text{Explorer}, \text{Surveyor}, \text{Defender}, \text{Idle}\}.
$$

Role evolution is modeled as a state-dependent stochastic process:

$$
\Pr\!\left(\text{role}_i(k+1)=r' \mid \text{role}_i(k)=r,\; z_i(k)\right)
= P_{r\to r'}(z_i(k)),
$$

where $z_i(k)$ is a local context vector that may include:

- local mean utility,
- local utility gradient magnitude,
- neighborhood density,
- revisit intensity,
- recent displacement,
- optional future battery/load placeholders.

This is **CRN-inspired**, not a claim of exact chemical-kinetics derivation.  
The right interpretation is:

- roles act like internal species,
- transitions act like stochastic reactions,
- aggregate role populations can be studied statistically over time.

## 5.9 Hybrid control law

The final swarm controller is

$$
u_i(k) =
w_{\text{ppo}}(i,k)\,u_i^{\text{ppo}}(k)
+
w_{\text{pf}}(i,k)\,u_i^{\text{pf}}(k)
+
w_{\text{cons}}(i,k)\,u_i^{\text{cons}}(k).
$$

The weights are role-dependent and context-dependent.

### Example role tendencies

**Explorer**
- high PPO weight,
- moderate utility attraction,
- weak consensus,
- moderate repulsion.

**Surveyor**
- balanced PPO and PF,
- moderate consensus,
- strong anti-revisit pressure.

**Defender**
- low PPO,
- strong repulsion,
- stronger local stabilization / spacing preservation.

**Idle / Recovery**
- small motion magnitude,
- weak random-walk or reset-like behavior,
- useful for reducing deterministic deadlocks.

After combination, the final control is clipped:

$$
u_i \leftarrow \text{sat}_{u_{\max}}(u_i).
$$

Then positions are updated via the single-integrator law.

## 5.10 Optional safety refinement

A version-2 extension may project the nominal hybrid control through a local optimization layer:

$$
\min_{u_i} \|u_i - u_i^{\text{nom}}\|^2
$$

subject to pairwise safety constraints or control-barrier-type inequalities.

This is an **optional refinement**, not a version-1 assumption.

# 6. Why a Hybrid Controller is Necessary

Pure PPO is expected to work well at the **single-agent** level, because it can learn local utility-seeking motion from image patches.

But raw PPO replication to many agents typically creates predictable failure modes:

- clustering,
- overlapping trajectories,
- repeated revisits,
- poor spatial partitioning,
- weak coordination as swarm size grows.

Classical swarm-control layers address exactly these weaknesses:

- **potential fields** improve spacing and anti-collision behavior,
- **consensus** improves coordination and local agreement,
- **role switching** adds adaptive heterogeneity,
- **geometric/statistical metrics** make the swarm behavior measurable.

So the notebook hypothesis is:

> **Pure RL learns local behavior; hybrid RL + MRS theory produces better swarm behavior.**

# 7. Notebook Roadmap

This notebook will be built as a logically staged pipeline.

## Section 0: Motivation and roadmap
We state the problem, the hybrid thesis, and the end-to-end pipeline.

## Section 1: Imports, paths, seeds, device, config
We set up reproducibility, device handling, directories, and global configuration.

## Section 2: Image ingestion and utility-map generation
We load a satellite image, compute the VARI field, normalize it, optionally smooth/downsample it, and inspect the resulting scalar field.

## Section 3: Single-agent Gymnasium environment
We construct the custom environment, define the observation crop, action space, first-visit reward, visit map, and rendering logic.

## Section 4: PPO setup
We define the CNN-based PPO model, vectorized environment wrapper, and training configuration.

## Section 5: PPO training
We train the single-agent policy and save checkpoints and logs.

## Section 6: PPO evaluation
We compare trained PPO against simple baselines and inspect rollout quality, trajectory structure, and utility harvested.

## Section 7: Transition to the swarm setting
We explain why naïve multi-agent PPO replication is insufficient and define the shared multi-agent bookkeeping.

## Section 8: Multi-agent initialization and shared maps
We initialize $N$ agents, shared visit memory, and optional communication graph visualization.

## Section 9: Potential-field module
We derive and implement utility attraction, inter-agent repulsion, revisit repulsion, and boundary handling.

## Section 10: Consensus module
We build the proximity graph, compute adjacency and Laplacian matrices, and implement a bounded coordination term.

## Section 11: Role-switching module
We define the roles, local context features, stochastic transition law, and diagnostics for role evolution.

## Section 12: Hybrid controller assembly
We combine PPO, PF, and consensus into a single role-conditioned controller.

## Section 13: Full swarm rollout
We run the full simulation, update states and roles, track metrics, and store swarm trajectories.

## Section 14: Visualization suite
We render:
- trajectory overlays,
- role-colored rollouts,
- visit-count heatmaps,
- coverage curves,
- consensus diagnostics,
- role-population dynamics,
- animation outputs.

## Section 15: Ablations
We compare:
- random multi-agent baseline,
- raw PPO replication,
- PPO + PF,
- PPO + PF + consensus,
- PPO + PF + consensus + roles.

## Section 16: Parameter sweeps
We study sensitivity to:
- number of agents,
- communication radius,
- repulsion gain,
- attraction gain,
- consensus gain,
- role-switch period.

## Section 17: Discussion and theory alignment
We identify what PPO learns, where it fails, which classical layers fix what, and which claims remain empirical rather than formally proved.

## Section 18: Export and handoff
We save plots, animations, metrics, configs, and backend outputs later reused by deployment code.

# 8. Expected Outcomes

## 8.1 Single-agent PPO
Expected behavior:

- trajectories drift toward higher-utility regions,
- returns improve over training,
- visited cells have higher average utility than random baseline,
- the policy learns a meaningful local exploration instinct.

## 8.2 Raw multi-agent PPO
Expected failure modes:

- agent clustering,
- overlap,
- revisit-heavy trajectories,
- poor spatial spread.

## 8.3 PPO + potential fields
Expected improvements:

- better dispersion,
- fewer close approaches,
- reduced revisit concentration,
- improved coverage geometry.

## 8.4 PPO + PF + consensus
Expected improvements:

- smoother coordination,
- lower local imbalance,
- stronger regional deployment consistency.

## 8.5 Full hybrid with roles
Expected improvements:

- best exploration/exploitation balance,
- adaptive behavior in dense vs sparse regions,
- emergent specialization,
- strongest overall coverage-quality tradeoff.

# 9. Metrics to be Reported

Primary metrics:

$$
\text{Coverage Ratio} = \frac{\text{unique visited cells}}{\text{total cells}}
$$

$$
\text{Utility Gain} = \sum_{\text{first visits}} \phi(c)
$$

$$
\text{Redundancy Index} = \frac{\text{revisit count}}{\text{total steps}}
$$

Additional metrics:

- mean pairwise distance,
- coverage entropy,
- consensus error,
- role occupancy fractions,
- connected-component statistics of the communication graph,
- optional algebraic connectivity $\lambda_2(L)$,
- optional Voronoi-based coverage proxies.

# 10. Expected Final Outputs of the Notebook

By the end of this notebook, we expect to produce:

- the original satellite image,
- the processed VARI/utility map,
- trained PPO checkpoints,
- single-agent evaluation plots,
- raw multi-agent PPO rollouts,
- hybrid swarm rollouts,
- trajectory overlays,
- visit-count heatmaps,
- coverage and redundancy curves,
- role-evolution plots,
- parameter-sweep visualizations,
- saved metrics in CSV/JSON form,
- reusable backend logic for deployment.

# 11. Honest Technical Boundaries

This notebook will stay technically honest about the following points:

1. **VARI is not true NDVI.**  
   It is an RGB-derived vegetation proxy.

2. **The full hybrid system is not claimed to have a complete global proof.**  
   Individual components are theory-grounded; the composed system is primarily justified empirically.

3. **CRN language is used as inspiration, not overclaimed as an exact derivation.**

4. **Consensus must be used carefully.**  
   Naïve position consensus can collapse the swarm, so bounded or information-based consensus is preferred.

5. **Potential fields can introduce local minima.**  
   Here they are used as shaping terms around PPO, not as the only controller.

# 12. Final Thesis of the Notebook

The main scientific message of this notebook is:

> **Pure reinforcement learning is strong at learning local navigation from raw local observations, but weak at scalable multi-agent coordination. Classical multi-robot systems theory is strong at providing structure, spacing, agreement, and interpretable collective behavior, but can be brittle when used alone. The most compelling controller is therefore hybrid: learned microscopic intelligence plus principled macroscopic coordination.**

That is the actual heart of **PPO-Driven Swarm Control**.