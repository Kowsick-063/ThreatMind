# ThreatMind

## Predictive Cyber Attack World Model with Counterfactual Defense Planning

ThreatMind is a predictive cybersecurity decision-support platform designed to understand how a network evolves over time, predict possible future attack trajectories, simulate defensive interventions, and identify effective defense strategies.

Instead of only detecting attacks after suspicious activity occurs, ThreatMind focuses on the question:

> **"What is likely to happen next, and what defensive action would reduce the future risk?"**

The system models network behavior as a dynamic probabilistic world and uses temporal machine learning, graph-based network modeling, multi-step forecasting, MITRE ATT&CK mapping, explainability, and counterfactual simulation to support proactive cyber defense.

---

## Table of Contents

* [Overview](#overview)
* [Problem](#problem)
* [Solution](#solution)
* [Core Capabilities](#core-capabilities)
* [System Architecture](#system-architecture)
* [ThreatMind Pipeline](#threatmind-pipeline)
* [Key Components](#key-components)

  * [Telemetry Ingestion](#1-telemetry-ingestion)
  * [Feature Engineering](#2-feature-engineering)
  * [Temporal Network States](#3-temporal-network-states)
  * [Dynamic Network Graph](#4-dynamic-network-graph)
  * [Temporal Graph Model](#5-temporal-graph-model)
  * [Probabilistic World Model](#6-probabilistic-world-model)
  * [Multi-Step Attack Trajectories](#7-multi-step-attack-trajectories)
  * [MITRE ATT&CK Mapping](#8-mitre-attck-mapping)
  * [Explainability](#9-explainability)
  * [Counterfactual Defense Simulation](#10-counterfactual-defense-simulation)
  * [Defense Optimization](#11-defense-optimization)
  * [Real-Time Streaming](#12-real-time-streaming)
  * [SOC Dashboard](#13-soc-dashboard)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Data](#data)
* [Database Architecture](#database-architecture)
* [API](#api)
* [Installation](#installation)
* [Development](#development)
* [Model Development](#model-development)
* [Evaluation](#evaluation)
* [Security](#security)
* [Testing](#testing)
* [Deployment](#deployment)
* [Development Roadmap](#development-roadmap)
* [Research Direction](#research-direction)
* [Future Extensions](#future-extensions)
* [Team](#team)
* [Project Status](#project-status)
* [License](#license)

---

# Overview

Traditional intrusion detection systems primarily focus on identifying malicious activity from current or historical observations.

ThreatMind extends this idea toward **predictive and decision-oriented cybersecurity**.

The system receives network telemetry such as PCAP or NetFlow-derived information and converts it into temporal representations of network behavior.

These representations are used to:

1. Understand the current network state.
2. Predict possible future states.
3. Generate multiple possible attack trajectories.
4. Map predicted behaviors to MITRE ATT&CK techniques.
5. Explain why a trajectory is considered risky.
6. Simulate defensive interventions.
7. Estimate the effect of those interventions.
8. Compare defense strategies using risk reduction and operational cost.

---

# Problem

Modern networks are dynamic.

Hosts appear and disappear, communication patterns change, services become exposed, and attackers can move through multiple stages of an intrusion.

A conventional detection pipeline can identify suspicious activity, but detection alone does not answer several important questions:

* What could happen next?
* Which hosts could become compromised?
* How could an attacker move through the network?
* Which attack trajectory is most likely?
* What happens if a host is isolated?
* What happens if a port is blocked?
* Which defensive action provides the highest risk reduction?
* What operational cost or service disruption would each defense introduce?

ThreatMind is designed around these predictive and counterfactual questions.

---

# Solution

ThreatMind models the network as a changing state over time:

```text
S(t) → S(t+1) → S(t+2) → S(t+3) ...
```

The system learns a transition model:

```text
P(S(t+1) | S(t))
```

and uses it to generate possible future trajectories.

A simplified representation is:

```text
Current Network State
        ↓
Learned World Model
        ↓
Future State Distribution
        ↓
Multiple Possible Trajectories
        ↓
Attack Behaviour
        ↓
Defensive Intervention
        ↓
Counterfactual Future
        ↓
Defense Comparison
```

The goal is not simply to label the current traffic as malicious or benign.

The goal is to reason about **possible future network evolution** and evaluate defensive actions before applying them to a real network.

---

# Core Capabilities

ThreatMind is organized around four major capabilities:

## 1. Understand

Analyze network telemetry and construct a representation of the current network state.

## 2. Predict

Forecast possible future network states and attack trajectories.

## 3. Simulate

Apply hypothetical defensive interventions to simulated copies of the network state.

## 4. Recommend

Compare possible interventions and identify actions that provide strong risk reduction while considering operational cost and service disruption.

---

# System Architecture

```text
                 NETWORK TELEMETRY
                  PCAP / NetFlow
                        │
                        ▼
              ┌───────────────────┐
              │ Telemetry Ingestion│
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Feature Extraction│
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Temporal Windowing│
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Network State S(t)│
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Dynamic Network   │
              │ Graph             │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Temporal Graph    │
              │ Model             │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Probabilistic     │
              │ World Model       │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ K-Step Trajectory │
              │ Rollouts          │
              └─────────┬─────────┘
                        │
              ┌─────────┴──────────┐
              ▼                    ▼
      MITRE ATT&CK            Explainability
              │                    │
              └─────────┬──────────┘
                        ▼
              ┌───────────────────┐
              │ Counterfactual    │
              │ Engine            │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Intervention      │
              │ Simulator         │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ Defense Optimizer  │
              └─────────┬─────────┘
                        │
                        ▼
              ┌───────────────────┐
              │ ThreatMind SOC     │
              │ Dashboard          │
              └───────────────────┘
```

---

# ThreatMind Pipeline

The complete processing pipeline is:

```text
PCAP / NetFlow
      ↓
Ingestion
      ↓
Flow Aggregation
      ↓
Feature Extraction
      ↓
Data Cleaning
      ↓
Temporal Windowing
      ↓
Network State Construction
      ↓
Dynamic Graph Construction
      ↓
Temporal Graph Representation
      ↓
World Model
      ↓
Future State Prediction
      ↓
K-Step Rollout
      ↓
Attack Trajectory Generation
      ↓
MITRE ATT&CK Mapping
      ↓
Risk Explanation
      ↓
Counterfactual Simulation
      ↓
Defense Optimization
      ↓
SOC Dashboard
```

---

# Key Components

## 1. Telemetry Ingestion

ThreatMind is designed to work with network telemetry including:

* PCAP
* NetFlow-like flow records
* Network session information
* Simulated real-time telemetry

Initial development focuses on PCAP-based datasets.

The ingestion layer extracts information such as:

* Source IP
* Destination IP
* Source port
* Destination port
* Protocol
* Duration
* Packet count
* Byte count
* Packet rates
* Byte rates
* TCP flags

---

## 2. Feature Engineering

Raw network records are transformed into machine-learning-ready features.

The preprocessing pipeline includes:

* Duplicate removal
* Invalid record handling
* Missing-value handling
* Irrelevant-field removal
* Categorical encoding
* Numerical normalization
* Flow aggregation
* Statistical feature generation

Example derived features include:

```text
packets_per_second
bytes_per_second
connection_frequency
average_packet_size
unique_destination_ports
unique_destination_hosts
TCP_flag_patterns
```

The exact feature set will evolve during model development based on dataset characteristics and experimental evaluation.

---

## 3. Temporal Network States

Network activity is divided into time windows.

For example:

```text
0s ───────── 5s
     S(t)

5s ───────── 10s
     S(t+1)

10s ──────── 15s
     S(t+2)
```

Each window represents a network state.

The model learns relationships such as:

```text
S(t) → S(t+1)
```

and eventually:

```text
S(t) → S(t+1) → S(t+2) → ... → S(t+k)
```

The initial temporal window can be configured during experimentation rather than being permanently fixed.

---

## 4. Dynamic Network Graph

The network is represented as a graph.

### Nodes

Network hosts are represented as nodes.

```text
Host A
Host B
Host C
Host D
```

### Edges

Communication between hosts becomes an edge.

```text
Host A ───────→ Host B
```

Edges can contain attributes such as:

* Protocol
* Port
* Packets
* Bytes
* Duration
* Frequency
* TCP flags

A simplified graph can therefore represent:

```text
          ┌──────────┐
          │ Host A   │
          └────┬─────┘
               │
          TCP/443
               │
               ▼
          ┌──────────┐
          │ Host B   │
          └────┬─────┘
               │
          TCP/22
               │
               ▼
          ┌──────────┐
          │ Host C   │
          └──────────┘
```

NetworkX will initially be used for graph processing and experimentation, with Neo4j considered for more advanced graph storage and visualization.

---

## 5. Temporal Graph Model

A static graph only represents one network snapshot.

ThreatMind needs to understand how the graph changes over time.

The temporal graph model is therefore responsible for learning patterns such as:

```text
G(t) → G(t+1)
```

Possible model architectures include:

* GraphSAGE
* Graph Attention Networks
* Temporal GNN architectures

The implementation will select and evaluate an appropriate architecture rather than unnecessarily combining multiple models.

The primary goal is to learn useful temporal representations of network behavior.

---

## 6. Probabilistic World Model

The central concept of ThreatMind is the **network world model**.

Instead of producing only a single deterministic prediction, the system models possible future states.

Conceptually:

```text
P(S(t+1) | S(t))
```

The model can represent uncertainty about the future.

For example:

```text
Current State
     │
     ├── Future A ── 0.55
     │
     ├── Future B ── 0.30
     │
     └── Future C ── 0.15
```

These probabilities will be generated by the trained model rather than manually assigned.

---

## 7. Multi-Step Attack Trajectories

ThreatMind recursively rolls the world model forward.

Instead of predicting only:

```text
S(t+1)
```

it can generate:

```text
S(t+1)
S(t+2)
S(t+3)
...
S(t+k)
```

Multiple branches can be maintained:

```text
                    Current State
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
             S1         S2         S3
             │          │          │
          ┌──┴──┐    ┌──┴──┐    ┌──┴──┐
          ▼     ▼    ▼     ▼    ▼     ▼
         S11   S12  S21   S22  S31   S32
```

This allows ThreatMind to reason about multiple possible future trajectories rather than relying on one predicted path.

---

## 8. MITRE ATT&CK Mapping

Predicted attack-related behaviors can be mapped to the MITRE ATT&CK framework.

Conceptually:

```text
Model State
     ↓
Observed / Predicted Behaviour
     ↓
MITRE ATT&CK Technique
```

Each mapping can contain:

* Technique ID
* Technique name
* Confidence
* Supporting evidence

The mapping is intended to make predictions easier for security analysts to interpret.

---

## 9. Explainability

ThreatMind should not simply display:

```text
Risk = HIGH
```

It should provide evidence explaining why a prediction is considered risky.

Potential evidence includes:

* Unusual internal communication
* New host-to-host communication
* Unexpected port activity
* Abnormal packet rates
* Changes in communication frequency
* Suspicious network behavior

The explainability layer will expose the features and network relationships contributing to a prediction.

---

## 10. Counterfactual Defense Simulation

This is one of the key differentiating components of ThreatMind.

The system asks:

> **What would happen if we applied a particular defensive action?**

Possible interventions include:

```text
Block IP
Block Port
Isolate Host
Disable Connection
Segment Network
```

The intervention is applied to a **simulated copy** of the network state.

```text
                Current Network
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        Normal Future     Intervention
                              │
                              ▼
                     Counterfactual Future
```

The simulation does **not** automatically modify the real network.

---

## 11. Defense Optimization

Different interventions can have different costs.

For example:

```text
Isolate Host
    ↓
High Risk Reduction
    +
High Service Disruption
```

while:

```text
Block Port
    ↓
Moderate Risk Reduction
    +
Lower Operational Cost
```

ThreatMind uses a conceptual defense score:

```text
Defense Score =
Risk Reduction
− Operational Cost
− Service Disruption
```

The optimizer can compare simulated interventions and rank candidate defensive strategies.

---

## 12. Real-Time Streaming

The long-term architecture supports streaming telemetry.

Conceptually:

```text
Network
   ↓
Telemetry Stream
   ↓
Stream Processing
   ↓
Feature Extraction
   ↓
ThreatMind Inference
   ↓
World Model
   ↓
Prediction
   ↓
Defense Simulation
   ↓
SOC Dashboard
```

For development and demonstration, a simulated telemetry stream can be used before integrating with live infrastructure.

---

## 13. SOC Dashboard

The frontend will provide a security-operations view of the ThreatMind system.

The dashboard is intended to display:

### Network State

```text
Hosts
Connections
Protocols
Traffic
Network Graph
```

### Threat Prediction

```text
Current Risk
Predicted Risk
Future States
Attack Trajectories
```

### Explainability

```text
Why is this trajectory risky?
Which features contributed?
Which hosts are involved?
```

### MITRE ATT&CK

```text
Technique
Technique ID
Confidence
Evidence
```

### Counterfactual Simulation

```text
Intervention
Risk Before
Risk After
Risk Reduction
Operational Cost
Service Disruption
```

### Defense Recommendation

```text
Recommended Action
Expected Risk Reduction
Cost
Disruption
Confidence
```

---

# Technology Stack

## Backend

| Technology | Purpose          |
| ---------- | ---------------- |
| Python     | Core development |
| FastAPI    | REST API         |
| Uvicorn    | API server       |
| SQLAlchemy | Database ORM     |
| Pydantic   | Data validation  |

## Data Processing

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Pandas       | Tabular processing             |
| NumPy        | Numerical computation          |
| PyArrow      | Efficient data handling        |
| Scikit-learn | Classical ML and preprocessing |

## Machine Learning

| Technology        | Purpose                  |
| ----------------- | ------------------------ |
| PyTorch           | Deep learning            |
| PyTorch Geometric | Graph neural networks    |
| Scikit-learn      | Baselines and evaluation |

## Network / Graph Processing

| Technology | Purpose                         |
| ---------- | ------------------------------- |
| Scapy      | PCAP processing                 |
| NetworkX   | Graph construction and analysis |
| Neo4j      | Advanced graph storage          |

## Database

| Technology | Purpose                            |
| ---------- | ---------------------------------- |
| PostgreSQL | Persistent application data        |
| Redis      | Caching / streaming infrastructure |

## Frontend

| Technology | Purpose              |
| ---------- | -------------------- |
| React      | SOC dashboard        |
| TypeScript | Frontend development |

## Infrastructure

| Technology     | Purpose                        |
| -------------- | ------------------------------ |
| Docker         | Containerization               |
| Docker Compose | Local multi-service deployment |

---

# Project Structure

```text
ThreatMind/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── network.py
│   │
│   ├── schemas/
│   ├── api/
│   ├── services/
│   └── core/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── tsconfig.json
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── README.md
│
├── models/
│   ├── checkpoints/
│   ├── trained/
│   └── README.md
│
├── preprocessing/
│   ├── ingestion/
│   ├── cleaning/
│   ├── features/
│   └── temporal/
│
├── graph/
│   ├── construction/
│   ├── features/
│   └── temporal/
│
├── forecasting/
│   ├── baselines/
│   ├── temporal/
│   ├── world_model/
│   └── rollout/
│
├── counterfactual/
│   ├── interventions/
│   ├── simulator/
│   └── optimizer/
│
├── explainability/
│   ├── attribution/
│   └── evidence/
│
├── mitre/
│   ├── mapping/
│   └── knowledge/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── models/
│
├── notebooks/
│
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Data

## Initial Dataset

The initial development dataset is planned around:

**CIC-IDS2017**

The dataset provides labeled network traffic suitable for developing and evaluating intrusion-detection and predictive modeling components.

Potential future datasets include:

* CSE-CIC-IDS2018
* UNSW-NB15
* CIC-DDoS2019
* TON_IoT

Dataset selection may evolve as the modeling requirements become clearer.

---

# Database Architecture

ThreatMind uses PostgreSQL for persistent application data.

The planned core entities include:

```text
network_sessions
network_events
network_states
predictions
attack_trajectories
mitre_techniques
interventions
simulation_results
```

Conceptual relationship:

```text
Network Sessions
       ↓
Network Events
       ↓
Network States
       ↓
Predictions
       ↓
Attack Trajectories
       ↓
MITRE Techniques
       ↓
Interventions
       ↓
Simulation Results
```

Sensitive or unnecessary raw payload information should not be stored by default.

---

# API

The backend exposes REST APIs for the ThreatMind platform.

Planned endpoints include:

| Method | Endpoint                           | Purpose                         |
| ------ | ---------------------------------- | ------------------------------- |
| POST   | `/api/telemetry/upload`            | Upload telemetry                |
| GET    | `/api/network/state`               | Retrieve current network state  |
| GET    | `/api/forecast`                    | Retrieve future predictions     |
| GET    | `/api/trajectory`                  | Retrieve attack trajectories    |
| GET    | `/api/explanation/{prediction_id}` | Retrieve prediction explanation |
| POST   | `/api/simulation`                  | Run counterfactual simulation   |
| POST   | `/api/defense/compare`             | Compare defensive interventions |

Health endpoints:

```text
GET /
GET /health
```

API documentation is available through FastAPI during development.

---

# Installation

## Requirements

Recommended development environment:

* Windows / Linux
* Python 3.11
* PostgreSQL
* Git
* Node.js and npm
* Docker Desktop
* VS Code

---

## Clone the Repository

```bash
git clone https://github.com/<your-username>/ThreatMind.git
cd ThreatMind
```

---

## Create Python Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

---

## Install Backend Dependencies

```powershell
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create:

```text
.env
```

Example:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/threatmind
REDIS_URL=redis://localhost:6379/0
```

Never commit `.env` to GitHub.

Use `.env.example` for safe configuration templates.

---

# Database Setup

Create the PostgreSQL database:

```sql
CREATE DATABASE threatmind;
```

Then configure:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/threatmind
```

Database migrations will be managed as the schema develops.

---

# Running the Backend

From the project root:

```powershell
uvicorn backend.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/health
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Running the Frontend

After the React application is initialized:

```powershell
cd frontend
npm install
npm run dev
```

The frontend development server will normally be available at:

```text
http://localhost:5173
```

---

# Development

ThreatMind is being developed incrementally.

The development principle is:

```text
Real Data
   ↓
Real Processing
   ↓
Real Model
   ↓
Real Prediction
   ↓
Real Simulation
   ↓
Real Dashboard
```

Mock predictions should not be presented as real model results.

Where a component is still under development, the implementation and UI should clearly distinguish between:

* implemented functionality
* experimental functionality
* planned functionality

---

# Model Development

The model-development pipeline will initially establish classical baselines before moving toward the full temporal graph world model.

Planned progression:

```text
Random Forest
     ↓
Logistic Regression
     ↓
LSTM / GRU
     ↓
Temporal Graph Model
     ↓
Probabilistic World Model
     ↓
Multi-Step Rollout
```

Models will be evaluated using time-aware data splitting to reduce temporal leakage.

---

# Evaluation

ThreatMind will evaluate both predictive performance and usefulness for defense planning.

## Classification / Prediction Metrics

Potential metrics include:

* Precision
* Recall
* F1-score
* Macro F1
* Confusion Matrix
* Top-1 accuracy
* Top-3 accuracy

## Temporal Evaluation

Forecasting performance will also be evaluated across prediction horizons.

Example:

```text
Horizon 1
Horizon 2
Horizon 3
...
Horizon K
```

The objective is to understand how prediction quality changes as the system forecasts further into the future.

## Counterfactual Evaluation

The defense simulator will evaluate:

```text
Risk Before Intervention
Risk After Intervention
Risk Reduction
Operational Cost
Service Disruption
```

No performance numbers will be claimed until they are obtained experimentally.

---

# Ablation Study

The project will compare increasingly sophisticated model configurations.

A planned comparison is:

```text
Random Forest
     ↓
LSTM
     ↓
Temporal Model
     ↓
Temporal GNN
     ↓
Full ThreatMind World Model
```

This allows the contribution of temporal modeling, graph modeling, and probabilistic trajectory generation to be investigated independently.

---

# Security

ThreatMind is designed as a **decision-support and simulation system**.

The counterfactual engine should operate on simulated copies of network states.

The system should not automatically perform real-world network blocking or host isolation without explicit authorization and appropriate operational safeguards.

Security considerations include:

* API authentication
* Role-based access control
* Secure configuration
* Database access control
* Audit logging
* Input validation
* Model endpoint protection
* Avoiding unnecessary storage of raw payloads
* Secure handling of credentials
* Simulation isolation

---

# Testing

Testing will cover individual components and complete workflows.

## Unit Tests

Examples:

```text
PCAP parser
Feature extraction
Temporal state generation
Graph construction
Prediction formatting
Intervention logic
```

## Integration Tests

Examples:

```text
Telemetry → Database
Database → Prediction
Prediction → Trajectory
Trajectory → Simulation
Simulation → API
API → Dashboard
```

## Model Tests

Examples:

```text
Input shape validation
Output probability validation
Temporal consistency
Rollout stability
```

---

# Deployment

## Local Development

Docker Compose can provide:

```text
ThreatMind
├── FastAPI
├── PostgreSQL
└── Redis
```

The React frontend can run independently during development.

---

## Advanced Architecture

A future deployment architecture can evolve toward:

```text
Network
   ↓
Telemetry
   ↓
Kafka / Streaming Layer
   ↓
Stream Processing
   ↓
Feature Extraction
   ↓
ML Inference
   ↓
ThreatMind World Model
   ↓
PostgreSQL / Graph Storage
   ↓
API Gateway
   ↓
SOC Dashboard
```

GPU acceleration can be introduced when model complexity and workload require it.

---

# Development Roadmap

## Phase 1 — Foundation

* [x] Create repository
* [x] Python environment
* [x] FastAPI backend
* [x] Health endpoint
* [ ] PostgreSQL integration
* [ ] Database models
* [ ] Environment configuration

## Phase 2 — Data Pipeline

* [ ] Dataset acquisition
* [ ] PCAP ingestion
* [ ] Flow extraction
* [ ] Data cleaning
* [ ] Feature engineering
* [ ] Processed dataset generation

## Phase 3 — Temporal Modeling

* [ ] Temporal window generation
* [ ] Network state representation
* [ ] Baseline models
* [ ] Time-aware evaluation
* [ ] Forecasting API

## Phase 4 — Dynamic Graph

* [ ] Network graph construction
* [ ] Node features
* [ ] Edge features
* [ ] Temporal graph representation
* [ ] Graph analysis

## Phase 5 — Temporal World Model

* [ ] Temporal GNN
* [ ] State transition modeling
* [ ] Probabilistic predictions
* [ ] Multi-step rollout
* [ ] Trajectory storage

## Phase 6 — Cybersecurity Intelligence

* [ ] MITRE ATT&CK mapping
* [ ] Prediction explanations
* [ ] Evidence generation
* [ ] Risk scoring

## Phase 7 — Counterfactual Engine

* [ ] Intervention representation
* [ ] Block IP simulation
* [ ] Block port simulation
* [ ] Host isolation
* [ ] Connection disabling
* [ ] Network segmentation
* [ ] Counterfactual rollout

## Phase 8 — Defense Optimization

* [ ] Risk reduction calculation
* [ ] Operational cost modeling
* [ ] Service disruption modeling
* [ ] Defense scoring
* [ ] Intervention ranking

## Phase 9 — SOC Dashboard

* [ ] Network graph visualization
* [ ] Current network state
* [ ] Future trajectories
* [ ] Risk visualization
* [ ] MITRE view
* [ ] Explainability view
* [ ] Counterfactual comparison
* [ ] Defense recommendations

## Phase 10 — Streaming and Deployment

* [ ] Simulated telemetry stream
* [ ] Real-time inference
* [ ] Redis integration
* [ ] Docker Compose
* [ ] Authentication
* [ ] RBAC
* [ ] Audit logging
* [ ] End-to-end testing

---

# Research Direction

ThreatMind explores the intersection of:

* Cybersecurity
* Network intrusion detection
* Temporal machine learning
* Graph neural networks
* Probabilistic modeling
* Predictive analytics
* Counterfactual reasoning
* Decision support
* MITRE ATT&CK

The central research direction is:

> **Can a learned model of network evolution be used not only to predict future attack behavior, but also to compare hypothetical defensive interventions before they are applied?**

---

# Future Extensions

Potential extensions include:

* Larger network environments
* Additional cybersecurity datasets
* Online model updating
* Continual learning
* Advanced temporal graph architectures
* Reinforcement-learning-based defense planning
* Digital-twin-style network simulation
* Multi-agent attack simulation
* Real SOC integrations
* Automated threat-intelligence enrichment
* Advanced graph databases
* Distributed inference
* GPU-accelerated world models

These are future directions and are not necessarily part of the current implementation.

---

# Team

ThreatMind follows a modular team structure covering:

* Data Engineering
* Machine Learning
* Graph / AI Engineering
* Cybersecurity
* Backend Engineering
* Frontend Engineering

Each module can be developed independently while communicating through defined APIs and data schemas.

---

# Project Status

🚧 **Active Development**

Current implementation:

```text
Foundation
    ↓
FastAPI Backend              ✅
PostgreSQL Setup             🔄
Database Architecture        🔄
PCAP Pipeline                ⏳
Temporal Modeling            ⏳
Dynamic Graph                ⏳
World Model                  ⏳
Attack Trajectories          ⏳
MITRE Mapping                ⏳
Explainability               ⏳
Counterfactual Simulation    ⏳
Defense Optimization         ⏳
SOC Dashboard                ⏳
Real-Time Streaming          ⏳
```

The project will progressively replace each planned component with tested, working implementations.

---

# Design Principle

ThreatMind is built around a simple progression:

```text
DETECT
   ↓
UNDERSTAND
   ↓
PREDICT
   ↓
SIMULATE
   ↓
COMPARE
   ↓
DEFEND
```

The long-term objective is to move cybersecurity decision-making from:

```text
"What is happening?"
```

toward:

```text
"What is likely to happen next?"
```

and ultimately:

```text
"What can we do now to reduce the future risk?"
```

---

# License

License information will be added as the project reaches its intended release stage.

---

## ThreatMind

**Predict the threat. Simulate the future. Choose the defense.**
