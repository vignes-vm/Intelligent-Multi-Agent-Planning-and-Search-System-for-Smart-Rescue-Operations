# Intelligent-Multi-Agent-Planning-and-Search-System-for-Smart-Rescue-Operations
Simulated smart rescue environment (like a disaster-struck building or terrain). Multiple autonomous agents must search for and rescue victims using heuristic search (A*), planning (STRIPS), and cooperative coordination, real-time re-planning and logical reasoning to complete rescue missions effectively.

## Literature & Research Background
### Multi-Agent Pathfinding (MAPF):
Multi-agent pathfinding frames the problem as agents navigating a shared graph from start to goal without collisions. Time-extended A* techniques like Spatio-Temporal A* augment the search by including temporal dimensions and wait actions to avoid collisions effectively. Classical literature lays out the formal problem setup, coordination constraints, and benchmarks.

### Centralized vs Decentralized Coordination
Centralized task planning paired with A* pathing can optimize globally but suffers in dynamic environments. Recent approaches favor decentralized coordination for better resilience and flexibility.

### Multi-Agent Task Planning & Synergies
Advanced frameworks consider conflicts and synergies — such as when agents share tasks or resources — and solve these via coupled planning algorithms like Iterative Inter‑Dependent Planning (IIDP).

### Search & Rescue-Specific Planning
Papers on collaborative search-and-rescue scenario planning discuss joint planning under uncertainty, with demonstrated response-time savings of ~20–25% over state-of-the-art task allocation techniques
ACM Digital Library.

### Receding-Horizon Coordination
Receding Horizon Planning (RHP) frameworks handle dynamic task assignment to heterogeneous agents by replanning over rolling windows for efficiency and robustness in rescue missions.

### MARL + Heuristics Hybrid
Emerging hybrid approaches combine heuristic planners (e.g., A*) for global guidance with Multi-Agent Deep RL for real-time adaptation — improving efficiency and flexibility in dynamic and conflict-heavy environments.

### Hierarchical Multi-Agent Learning
Hierarchical reinforcement learning structures multi-agent planning into subtasks and dynamic domain sections, optimizing coordination and learning efficiency at scale.

### Fuzzy/MPC Control for Real-Time Coordination
Recent work uses Model Predictive Fuzzy Control (MPFC) — combining centralized and fuzzy logic control — to achieve near-optimal coordination with low runtime cost in SaR missions.

### Adversarial and Robust Coordination in SAR
Approaches like AdverSAR train agents with adversarial communication noise and sensor faults to maintain robust coordination under uncertain or adversarial conditions.