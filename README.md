# PPO-driven Swarm Control  

### A Hybrid Multi-Robot Framework Combining Reinforcement Learning, Consensus, Potential Fields, and CRN-Based Role Switching

**Author:** Ayushman Mishra  
**GitHub:** https://github.com/aymisxx  

---

## Project Overview

This project presents a **hybrid swarm control framework** that fuses **reinforcement learning** with **classical multi-robot systems (MRS) theory** to achieve scalable, stable, and efficient multi-agent coverage.

A microscopic **PPO-based learned controller** handles local navigation using vegetation information derived from satellite imagery, while macroscopic coordination is enforced through:

- Artificial **potential fields**
- **Graph-based consensus** dynamics
- **CRN-inspired stochastic role switching**

The result is a swarm that is **adaptive, decentralized, collision-free, and theoretically grounded**.

---

## Application Context

The motivating application is **precision agriculture**:

- A vegetation-rich field is represented as a **normalized NDVI proxy** (VARI-based).
- Multiple UAVs (modeled as single-integrator agents) must **explore and cover the field efficiently**.
- Each location yields reward **only on first visit**, discouraging redundant exploration.

This setup naturally emphasizes:
- Coverage efficiency  
- Spatial dispersion  
- Redundancy reduction  
- Robust decentralized coordination  

---

## Core Contributions

### 1. PPO-Based Local Navigation
- Each agent observes a **128×128 local NDVI patch**
- A CNN-based PPO policy outputs motion commands
- Learns vegetation-seeking behavior without global knowledge

### 2. Artificial Potential Fields
- **Attraction** to high-NDVI regions  
- **Repulsion** from nearby agents (collision avoidance)  
- **Revisit penalty** to discourage redundant paths  

### 3. Graph-Based Consensus
- Local communication graph induces Laplacian dynamics
- Reduces swarm imbalance and excessive dispersion
- Guarantees asymptotic consensus under standard connectivity assumptions

### 4. CRN-Inspired Role Switching
Agents stochastically transition between roles:
- **Explorer**: PPO-dominant, fast exploration  
- **Surveyor**: balanced behavior  
- **Defender**: strong repulsion and consensus  
- **Idle**: low activity / recovery mode  

Role transitions depend on local density and coverage metrics, inspired by **Chemical Reaction Networks (CRNs)**.

---

## Mathematical Model

- Robots are modeled as **single-integrator agents**  
- Hybrid control law:
  
  $$
  u_i = w_{rl}(r_i)u_i^{PPO} + w_{pf}(r_i)u_i^{PF} + w_{cons}(r_i)u_i^{cons}
  $$

- Potential fields guarantee collision avoidance and boundary invariance
- Consensus dynamics ensure cohesion and stability
- Mean-field CRN model captures swarm-level role distribution

Formal proofs are provided for:
- Consensus convergence  
- Collision avoidance  
- Boundedness and safety  

---

## Experimental Validation

Five regimes are evaluated:

1. Single-agent PPO  
2. Multi-agent PPO (no coordination)  
3. PPO + Potential Fields  
4. PPO + PF + Consensus  
5. **Full Hybrid: PPO + PF + Consensus + Role Switching**

### Key Findings:
- Raw PPO swarms cluster and revisit excessively  
- Potential fields improve dispersion  
- Consensus reduces coverage imbalance  
- **Full hybrid controller achieves best coverage, lowest redundancy, and strongest spatial organization**

Metrics include:
- Coverage ratio  
- NDVI harvested  
- Redundancy index  
- Consensus error  
- Role distribution dynamics  

---

## Folder Structure

```
PPO-driven Swarm Control
├── data/
│   ├── field_satellite.jpg
│   └── ndvi_field.npy
├── models/
│   └── ppo_ndvi_drone.zip
├── results/
│  (trajectory plots)
├── report/
│   └── PPO_driven_Swarm_Control_Report.pdf
├── PPO_Driven_Swarm_Control (Notebook).ipynb
├── PPO_Driven_Swarm_Control (PDF).pdf
├── requirements.txt
└── README.md
```

The notebook (**PPO_Driven_Swarm_Control (Notebook).ipynb**) is **fully standalone and reproducible**, starting from NDVI extraction and ending with full swarm simulations. '**results**' contains the hybrid rollout trajectory video. '**report**' folder contains the project report.

---

## Dependencies

- Python 3.9+  
- NumPy  
- OpenCV  
- Matplotlib  
- Gymnasium  
- PyTorch  
- Stable-Baselines3  

GPU acceleration (CUDA) is supported but optional.

---

## How to Run

1. Place a satellite image in `data/field_satellite.jpg`
2. Open the notebook:
   ```bash
   PPO_driven_Swarm_Control (Notebook).ipynb
   ```
3. Run all cells sequentially:
   - NDVI generation
   - PPO training
   - Multi-agent hybrid simulation
4. Outputs (plots, GIFs, metrics) are saved to `results/`

If you wish to look at the project without running the notebook/codes, kindly open **PPO_Driven_Swarm_Control (PDF).pdf**

---

## Key Takeaway

This project demonstrates that **reinforcement learning alone is insufficient for scalable swarm coordination**.  
By embedding PPO inside a **theoretically grounded MRS framework**, we obtain:

> Learning with structure.  
> Adaptivity with guarantees.  
> Emergence without chaos.

---

## Results & Analysis

The proposed hybrid swarm-control framework was evaluated through extensive simulations on a vegetation-driven coverage task. Performance was analyzed by progressively enabling coordination layers on top of a PPO-based local controller.

### Experimental Regimes

Five control configurations were compared:

- Single-Agent PPO

- Multi-Agent PPO (no coordination)

- PPO + Potential Fields (PF)

- PPO + PF + Consensus

- Full Hybrid: PPO + PF + Consensus + CRN Role Switching

All experiments used identical NDVI fields, swarm sizes, episode lengths, and initialization distributions to ensure fair comparison.

### Coverage Performance

- **Single-agent PPO** successfully learns vegetation-seeking behavior but is inherently limited in spatial coverage.
- **Multi-agent PPO without coordination** exhibits significant clustering and redundant trajectories, resulting in poor marginal gains as swarm size increases.
- **Potential field integration** improves agent dispersion and collision avoidance, increasing overall coverage.
- **Consensus dynamics** further balance spatial distribution, reducing over-exploration of local regions.
- The **full hybrid controller** achieves the highest coverage ratio by efficiently spreading agents across the environment while prioritizing high-NDVI regions.

### Redundancy & Dispersion

- **Raw PPO swarms** suffer from high revisit rates and overlapping trajectories.
- **Potential-field repulsion** significantly reduces close-proximity interactions.
- **Consensus terms** smooth swarm motion and prevent fragmentation.
- **CRN-based role switching** introduces functional heterogeneity, further reducing redundancy by dynamically reallocating agents to exploration-heavy or stabilization-focused roles.

**Overall, the full hybrid system consistently demonstrates the lowest redundancy index and the most uniform spatial dispersion.**

### Consensus Convergence

- Swarms with **consensus-enabled controllers** exhibit rapid decay of consensus error.
- Empirical convergence behavior aligns closely with **theoretical guarantees** derived from Laplacian-based analysis.
- Consensus improves **global coordination** without enforcing rigid formations, preserving exploration flexibility.

---

### Role Distribution Dynamics

- **CRN-inspired stochastic role switching** yields stable population-level role distributions.
- **Explorer agents** dominate early exploration phases, while **surveyor and defender roles** increase as local density and coverage rise.
- This adaptive redistribution improves **robustness** and prevents long-term stagnation.

---

### Qualitative Observations

Trajectory visualizations and time-lapse videos reveal clear qualitative differences:

- **PPO-only swarms** appear chaotic and locally greedy.
- **Hybrid swarms** exhibit smooth, structured, and interpretable collective motion.
- The **full hybrid controller** produces emergent behaviors such as territory splitting, wave-like dispersion, and coverage-front propagation.

> Minor boundary accumulation observed in earlier runs was found to be a transient effect of initialization and stochastic policy execution; upon rerunning the simulation with updated parameters, the swarm exhibited uniform coverage without persistent boundary clustering.

### Limitations

While the swarm exhibits strong dispersion and collision avoidance during early exploration, performance degrades over longer horizons. As coverage saturates and NDVI gradients weaken, PPO-driven actions can dominate the hybrid controller, reducing the effectiveness of fixed-gain potential-field repulsion. This occasionally leads to local clustering and near-collisions midway through the simulation.

Additionally, the learned PPO policy is not explicitly safety-aware and relies on classical control layers for collision avoidance. In dense regions, static potential-field gains and stochastic role switching may be insufficient to counter aggressive learned motions, suggesting the need for adaptive gain scheduling or safety-aware policy training in future work.

### Key Takeaways

- Reinforcement learning alone is insufficient for scalable swarm coordination.

- Classical MRS components provide structure, safety, and stability.

- PPO excels as a local intelligence module when embedded within a principled control architecture.

- The hybrid framework achieves robust, scalable, and interpretable swarm behavior.

---

# Citation

If you use or build upon this work / fork this work, please cite:

> Ayushman Mishra, *PPO-Driven Swarm Control: A Hybrid Multi-Robot Framework Combining Consensus, Potential Fields, and CRN-Based Role Switching*, github.com/aymisxx/PPO-driven-Swarm-Control

---