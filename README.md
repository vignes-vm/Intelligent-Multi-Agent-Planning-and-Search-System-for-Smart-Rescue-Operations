# Intelligent-Multi-Agent-Planning-and-Search-System-for-Smart-Rescue-Operations
Simulated smart rescue environment (like a disaster-struck building or terrain). Multiple autonomous agents must search for and rescue victims using heuristic search (A*), planning (STRIPS), and cooperative coordination, real-time re-planning and logical reasoning to complete rescue missions effectively.

## 🧭 Problem Statement:
In real-world disaster scenarios—such as earthquakes, floods, or fires—timely and efficient search-and-rescue operations are critical to saving lives. Traditional rescue efforts often suffer from a lack of coordination, dynamic changes in the environment (e.g., debris, fire spread), and limited information about victims’ locations.

This project addresses the challenge of coordinated rescue planning in such dynamic settings by designing an intelligent multi-agent system capable of autonomous navigation, task allocation, and dynamic replanning. The system will integrate heuristic search algorithms (A*, IDA*), STRIPS-based planning, and logical reasoning mechanisms to enable multiple agents to cooperatively plan, adapt, and execute rescue missions in an evolving environment.

The objective is to develop a robust, hybrid AI framework that blends centralized task planning and decentralized coordination to maximize rescue efficiency, minimize path conflicts, and dynamically adjust to environmental changes.

## 📚 Literature & Research Background
The field of multi-agent systems (MAS) has seen rapid growth in applications ranging from autonomous vehicles to distributed robotics. In the context of disaster response and search-and-rescue (SAR), recent research has focused on combining multi-agent pathfinding (MAPF) with adaptive learning, planning, and coordination mechanisms.

Lu et al. (2023) proposed DrMaMP, a distributed real-time multi-agent planner capable of navigating cluttered and dynamic environments. Their approach uses decomposition and replanning to optimize rescue time while avoiding collisions.

Rahman et al. (2022) introduced AdverSAR, a MARL-based SAR system designed to be robust against communication noise and partial observability. This work emphasizes the need for decentralized and fault-tolerant coordination under uncertainty.

Liu et al. (2020) developed MAPPER, a hybrid planning system integrating global A* search with local RL-based adaptation, demonstrating strong performance in dynamic obstacle scenarios.

Nguyen et al. (2018) proposed ResQ, which combines heuristic reinforcement learning with task allocation for human-volunteer-based SAR operations.

Surveys like Felner et al. (2020) on MAPF highlight the trade-offs between centralized vs decentralized planning, the use of spatio-temporal graphs, and real-time collision avoidance mechanisms.

Despite these advances, most existing approaches either focus solely on learning or planning, with limited integration between classical AI planning (e.g., STRIPS) and modern adaptive coordination strategies. There is a clear gap in building a hybrid framework that unifies logical reasoning, task-level planning, and multi-agent cooperation under real-time dynamic constraints.

This project aims to fill that gap by combining STRIPS-based planning, A*/IDA* pathfinding, and logical inference mechanisms in a unified architecture that can adapt dynamically to environmental changes and coordinate multiple agents efficiently in disaster zones.

## 🧪 Abstract
In critical rescue scenarios such as natural disasters, timely coordination among multiple autonomous agents is essential for maximizing survivor recovery and minimizing response time. This project presents the design and development of an Intelligent Multi-Agent Planning and Search System for smart rescue operations in dynamic environments. The proposed system integrates heuristic search algorithms (A*, Greedy, IDA*), STRIPS-based action planning, and logical reasoning to enable agents to collaboratively explore, plan, and adapt in real time.

A grid-based simulation environment with dynamic obstacles and variable terrain conditions will be constructed to evaluate the system. The rescue agents will be capable of autonomous decision-making, dynamic pathfinding, conflict resolution, and task reassignment under environmental uncertainty. The hybrid architecture leverages centralized planning for mission initialization and decentralized execution with continuous replanning and local inference.

Comparative analysis of planning algorithms and coordination strategies will be conducted to assess performance across key metrics including rescue time, path optimality, and robustness under failure conditions. This project bridges classical AI paradigms with emerging adaptive strategies and offers potential extensions into real-world applications in robotics, emergency response, and autonomous search systems.

