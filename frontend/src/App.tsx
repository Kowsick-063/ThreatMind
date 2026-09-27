import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Brain,
  ChevronRight,
  CircleDot,
  Clock3,
  Crosshair,
  Database,
  Network,
  Shield,
  ShieldAlert,
  Target,
  Zap,
} from "lucide-react";
import "./App.css";

const incidents = [
  {
    id: "INC-1499431310",
    status: "ANALYZED",
    risk: "MODERATE",
    probability: 40.5,
    category: "Network Service Discovery",
    time: "12:31:50",
  },
];

const forecast = [
  39.89, 39.96, 40.02, 40.08, 40.11, 40.15,
  40.18, 40.22, 40.27, 40.31, 40.36, 40.51,
];

const mitre = [
  { id: "T1046", name: "Network Service Discovery", confidence: 80 },
  { id: "T1595.001", name: "Active Scanning", confidence: 72 },
  { id: "T1059", name: "Command & Scripting", confidence: 45 },
  { id: "T1071.001", name: "Web Protocols", confidence: 42 },
];

const defenses = [
  {
    action: "BLOCK_SOURCE",
    target: "192.168.10.8",
    reduction: "-0.4192",
    cost: "0.15",
    disruption: "0.10",
    recommended: true,
  },
  {
    action: "BLOCK_DESTINATION",
    target: "55 affected hosts",
    reduction: "-0.4262",
    cost: "0.18",
    disruption: "0.12",
    recommended: false,
  },
  {
    action: "NO_ACTION",
    target: "Continue monitoring",
    reduction: "-1.3730",
    cost: "0.00",
    disruption: "0.00",
    recommended: false,
  },
];

function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Brain size={23} />
          </div>
          <div>
            <h1>ThreatMind</h1>
            <span>SOC INTELLIGENCE</span>
          </div>
        </div>

        <div className="nav-section">
          <span className="nav-title">OPERATIONS</span>

          <div className="nav-item active">
            <ShieldAlert size={18} />
            <span>Dashboard</span>
          </div>

          <div className="nav-item">
            <AlertTriangle size={18} />
            <span>Incidents</span>
            <b>1</b>
          </div>

          <div className="nav-item">
            <Network size={18} />
            <span>Network</span>
          </div>

          <div className="nav-item">
            <Target size={18} />
            <span>Threat Forecast</span>
          </div>

          <div className="nav-item">
            <Crosshair size={18} />
            <span>Counterfactuals</span>
          </div>
        </div>

        <div className="nav-section">
          <span className="nav-title">INTELLIGENCE</span>

          <div className="nav-item">
            <Database size={18} />
            <span>MITRE ATT&CK</span>
          </div>

          <div className="nav-item">
            <Brain size={18} />
            <span>Explainability</span>
          </div>
        </div>

        <div className="system-status">
          <div className="status-dot" />
          <div>
            <strong>SYSTEM ONLINE</strong>
            <span>All services operational</span>
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              SOC <ChevronRight size={14} /> INCIDENT MONITOR
            </div>
            <h2>Threat Intelligence Dashboard</h2>
          </div>

          <div className="topbar-right">
            <div className="live">
              <CircleDot size={13} />
              LIVE
            </div>

            <div className="timestamp">
              <Clock3 size={15} />
              14 Sep 2026 · 12:31:50
            </div>
          </div>
        </header>

        <section className="content">
          <div className="incident-banner">
            <div className="incident-left">
              <div className="warning-icon">
                <AlertTriangle size={22} />
              </div>

              <div>
                <span className="eyebrow">ACTIVE INCIDENT</span>
                <h3>INC-1499431310</h3>
                <p>Network Service Discovery activity detected</p>
              </div>
            </div>

            <div className="incident-meta">
              <div>
                <span>STATUS</span>
                <strong>ANALYZED</strong>
              </div>
              <div>
                <span>PEAK HORIZON</span>
                <strong>H12</strong>
              </div>
            </div>
          </div>

          <div className="metrics">
            <Metric
              title="CURRENT RISK"
              value="40.5%"
              label="MODERATE"
              icon={<ShieldAlert />}
              trend="stable"
            />

            <Metric
              title="ATTACK PROBABILITY"
              value="40.5%"
              label="H12 PEAK"
              icon={<Activity />}
              trend="up"
            />

            <Metric
              title="ATTACK SURFACE"
              value="43.4"
              label="SCORE"
              icon={<Network />}
              trend="up"
            />

            <Metric
              title="NETWORK NODES"
              value="62"
              label="59 EDGES"
              icon={<Database />}
              trend="stable"
            />
          </div>

          <div className="grid-main">
            <section className="card forecast-card">
              <CardHeader
                title="Future Risk Forecast"
                subtitle="12-step temporal prediction"
                icon={<Activity size={17} />}
              />

              <div className="forecast">
                <div className="y-axis">
                  <span>50%</span>
                  <span>40%</span>
                  <span>30%</span>
                  <span>20%</span>
                </div>

                <div className="chart">
                  <div className="threshold">
                    <span>Elevated threshold · 50%</span>
                  </div>

                  <div className="grid-line line-1" />
                  <div className="grid-line line-2" />
                  <div className="grid-line line-3" />

                  <svg
                    viewBox="0 0 600 220"
                    preserveAspectRatio="none"
                    className="chart-svg"
                  >
                    <defs>
                      <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#38bdf8" stopOpacity=".28" />
                        <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
                      </linearGradient>
                    </defs>

                    <path
                      d="M0 120 L55 118 L110 115 L165 112 L220 111 L275 108 L330 106 L385 103 L440 100 L495 97 L550 94 L600 88 L600 220 L0 220 Z"
                      fill="url(#riskFill)"
                    />

                    <polyline
                      points="0,120 55,118 110,115 165,112 220,111 275,108 330,106 385,103 440,100 495,97 550,94 600,88"
                      fill="none"
                      stroke="#38bdf8"
                      strokeWidth="3"
                    />

                    <circle cx="600" cy="88" r="5" fill="#38bdf8" />
                  </svg>

                  <div className="x-axis">
                    {forecast.map((_, i) => (
                      <span key={i}>H{i + 1}</span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="forecast-footer">
                <div>
                  <span>INITIAL</span>
                  <strong>39.89%</strong>
                </div>

                <div>
                  <span>PEAK</span>
                  <strong>40.51%</strong>
                </div>

                <div>
                  <span>TREND</span>
                  <strong className="stable">STABLE</strong>
                </div>
              </div>
            </section>

            <section className="card">
              <CardHeader
                title="Network State"
                subtitle="Current traffic snapshot"
                icon={<Network size={17} />}
              />

              <div className="network-stats">
                <NetworkStat label="Flows" value="105" />
                <NetworkStat label="Packets" value="426" />
                <NetworkStat label="Bytes" value="53,196" />
                <NetworkStat label="Sources" value="10" />
                <NetworkStat label="Destinations" value="55" />
                <NetworkStat label="Ports" value="105" />
              </div>

              <div className="network-visual">
                <div className="network-core">
                  <Shield size={22} />
                </div>

                {["A", "B", "C", "D", "E", "F"].map((node, i) => (
                  <div
                    className={`network-node node-${i}`}
                    key={node}
                  >
                    {node}
                  </div>
                ))}

                <div className="connection c1" />
                <div className="connection c2" />
                <div className="connection c3" />
                <div className="connection c4" />
                <div className="connection c5" />
              </div>
            </section>
          </div>

          <div className="grid-main">
            <section className="card">
              <CardHeader
                title="MITRE ATT&CK Mapping"
                subtitle="Observed behavioral evidence"
                icon={<Target size={17} />}
              />

              <div className="mitre-list">
                {mitre.map((item) => (
                  <div className="mitre-row" key={item.id}>
                    <div className="mitre-id">{item.id}</div>

                    <div className="mitre-name">
                      <strong>{item.name}</strong>
                      <div className="confidence-bar">
                        <div style={{ width: `${item.confidence}%` }} />
                      </div>
                    </div>

                    <div className="confidence">
                      {item.confidence}%
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="card">
              <CardHeader
                title="Attack Evidence"
                subtitle="Why ThreatMind considers this risky"
                icon={<Brain size={17} />}
              />

              <div className="evidence">
                <Evidence
                  title="Network discovery behavior"
                  text="10 unique sources contacted 55 destinations across 105 observed ports."
                />

                <Evidence
                  title="Temporal signal"
                  text="Attack probability remains elevated across the forecast horizon."
                />

                <Evidence
                  title="Graph exposure"
                  text="Current graph contains 62 nodes and 59 communication edges."
                />
              </div>
            </section>
          </div>

          <section className="card defense-card">
            <CardHeader
              title="Counterfactual Defense Simulation"
              subtitle="Compare modeled interventions before acting"
              icon={<Crosshair size={17} />}
            />

            <div className="defense-table">
              <div className="table-header">
                <span>INTERVENTION</span>
                <span>TARGET</span>
                <span>RISK Δ</span>
                <span>OP. COST</span>
                <span>DISRUPTION</span>
                <span>DECISION</span>
              </div>

              {defenses.map((defense) => (
                <div
                  className={`table-row ${
                    defense.recommended ? "recommended" : ""
                  }`}
                  key={defense.action}
                >
                  <strong>{defense.action}</strong>
                  <span>{defense.target}</span>
                  <span className="negative">
                    <ArrowDownRight size={14} />
                    {defense.reduction}
                  </span>
                  <span>{defense.cost}</span>
                  <span>{defense.disruption}</span>
                  <span>
                    {defense.recommended ? (
                      <b className="recommended-badge">
                        <Zap size={12} />
                        RECOMMENDED
                      </b>
                    ) : (
                      <span className="muted">SIMULATED</span>
                    )}
                  </span>
                </div>
              ))}
            </div>
          </section>

          <section className="recommendation">
            <div className="recommendation-icon">
              <Shield size={24} />
            </div>

            <div className="recommendation-text">
              <span>THREATMIND RECOMMENDATION</span>
              <h3>
                BLOCK_SOURCE{" "}
                <small>→ 192.168.10.8</small>
              </h3>
              <p>
                Recommended based on the simulated defense outcome and current
                network evidence.
              </p>
            </div>

            <div className="recommendation-score">
              <span>DEFENSE SCORE</span>
              <strong>-0.6692</strong>
            </div>
          </section>

          <section className="card lifecycle-card">
            <CardHeader
              title="Incident Lifecycle"
              subtitle="SOC investigation progress"
              icon={<Clock3 size={17} />}
            />

            <div className="lifecycle">
              {[
                "DETECTED",
                "ASSESSED",
                "FORECASTED",
                "SIMULATED",
                "RECOMMENDED",
              ].map((stage, index) => (
                <div className="stage-wrapper" key={stage}>
                  <div
                    className={`stage ${
                      index === 4
                        ? "current"
                        : index < 4
                        ? "complete"
                        : ""
                    }`}
                  >
                    {index < 4 ? "✓" : index === 4 ? "●" : ""}
                  </div>

                  <span>{stage}</span>

                  {index < 4 && <div className="stage-line" />}
                </div>
              ))}
            </div>
          </section>

          <div className="footer">
            ThreatMind · Predictive Cyber Attack World Model · SOC Intelligence Platform
          </div>
        </section>
      </main>
    </div>
  );
}

function Metric({
  title,
  value,
  label,
  icon,
  trend,
}: {
  title: string;
  value: string;
  label: string;
  icon: React.ReactNode;
  trend: "up" | "stable";
}) {
  return (
    <div className="metric card">
      <div className="metric-top">
        <span>{title}</span>
        <div className="metric-icon">{icon}</div>
      </div>

      <div className="metric-value">{value}</div>

      <div className="metric-bottom">
        <span>{label}</span>
        {trend === "up" ? (
          <ArrowUpRight size={15} />
        ) : (
          <span className="stable-dot">●</span>
        )}
      </div>
    </div>
  );
}

function CardHeader({
  title,
  subtitle,
  icon,
}: {
  title: string;
  subtitle: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="card-header">
      <div className="card-title">
        <div className="card-icon">{icon}</div>
        <div>
          <h3>{title}</h3>
          <p>{subtitle}</p>
        </div>
      </div>

      <button className="more">•••</button>
    </div>
  );
}

function NetworkStat({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="network-stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Evidence({
  title,
  text,
}: {
  title: string;
  text: string;
}) {
  return (
    <div className="evidence-row">
      <div className="evidence-dot" />
      <div>
        <strong>{title}</strong>
        <p>{text}</p>
      </div>
    </div>
  );
}

export default App;