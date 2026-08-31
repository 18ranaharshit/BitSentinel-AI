import React, { useState, useEffect } from 'react';
import NetworkBackground from './components/NetworkBackground';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8005';
const WS_URL = import.meta.env.VITE_WS_URL || (
  API_BASE.startsWith('https://') 
    ? API_BASE.replace('https://', 'wss://') + '/ws/stream'
    : API_BASE.replace('http://', 'ws://') + '/ws/stream'
);

export default function App() {
  // Theme state: defaults to 'light', persists in localStorage
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('bitsentinel_theme') || 'light';
  });

  const [activeTab, setActiveTab] = useState('dashboard');
  const [kpis, setKpis] = useState(null);
  const [streamData, setStreamData] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [addressClusterInfo, setAddressClusterInfo] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [walletClusters, setWalletClusters] = useState(null);
  const [expandedClusterId, setExpandedClusterId] = useState(null);

  // Network Correlation Alerts & Clusters State (Task 4)
  const [networkAlerts, setNetworkAlerts] = useState(null);
  const [networkClusters, setNetworkClusters] = useState(null);
  const [expandedNetworkTxid, setExpandedNetworkTxid] = useState(null);
  const [networkTxDetail, setNetworkTxDetail] = useState({});

  // Sync theme with HTML root attribute
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('bitsentinel_theme', theme);
  }, [theme]);

  // Load REST KPIs, Benchmarks, Wallet Clusters, and Network Alerts
  useEffect(() => {
    fetch(`${API_BASE}/api/kpis`)
      .then(res => res.json())
      .then(data => setKpis(data))
      .catch(err => console.log('API KPI Load Error:', err));

    fetch(`${API_BASE}/api/benchmarks`)
      .then(res => res.json())
      .then(data => setBenchmarks(data))
      .catch(err => console.log('API Benchmarks Load Error:', err));

    fetch(`${API_BASE}/api/clusters`)
      .then(res => res.json())
      .then(data => setWalletClusters(data))
      .catch(err => console.log('API Clusters Load Error:', err));

    fetch(`${API_BASE}/api/network-alerts?limit=50`)
      .then(res => res.json())
      .then(data => setNetworkAlerts(data))
      .catch(err => console.log('API Network Alerts Load Error:', err));

    fetch(`${API_BASE}/api/network-alerts/clusters`)
      .then(res => res.json())
      .then(data => setNetworkClusters(data))
      .catch(err => console.log('API Network Clusters Load Error:', err));
  }, []);

  // Connect WebSocket for Live Block Fraud Stream Ticker
  useEffect(() => {
    let ws;
    try {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setWsConnected(true);
      ws.onclose = () => setWsConnected(false);
      ws.onerror = () => setWsConnected(false);
      ws.onmessage = (event) => {
        const item = JSON.parse(event.data);
        setStreamData(prev => [item, ...prev.slice(0, 49)]); // Keep last 50 transactions
      };
    } catch (e) {
      setWsConnected(false);
    }
    return () => ws && ws.close();
  }, []);

  const handleSearch = (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    const query = searchQuery.trim();
    setSearchResult(null);
    setAddressClusterInfo(null);

    // 1. Search Risk Score & Category
    fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}`)
      .then(res => res.json())
      .then(data => setSearchResult(data))
      .catch(err => setSearchResult({ found: false, risk_score_status: 'not_found', message: 'Search failed' }));

    // 2. Lookup Co-Spend Cluster if it looks like an address
    if (/^(1|3|bc1)[a-zA-HJ-NP-Z0-9]{20,62}$/i.test(query)) {
      fetch(`${API_BASE}/api/clusters/${encodeURIComponent(query)}`)
        .then(res => res.json())
        .then(data => setAddressClusterInfo(data))
        .catch(err => setAddressClusterInfo(null));
    }
  };

  const toggleClusterExpand = (clusterId) => {
    setExpandedClusterId(prev => prev === clusterId ? null : clusterId);
  };

  const toggleNetworkTxExpand = (txid) => {
    if (expandedNetworkTxid === txid) {
      setExpandedNetworkTxid(null);
      return;
    }
    setExpandedNetworkTxid(txid);

    // Fetch forensic drilldown if not already in cache
    if (!networkTxDetail[txid]) {
      fetch(`${API_BASE}/api/network-alerts/${encodeURIComponent(txid)}`)
        .then(res => res.json())
        .then(data => {
          if (data.found) {
            setNetworkTxDetail(prev => ({ ...prev, [txid]: data }));
          }
        })
        .catch(err => console.log('Error fetching network alert detail:', err));
    }
  };

  const getTierBadgeStyle = (tier) => {
    switch (tier) {
      case 'critical':
        return { background: 'var(--alert-critical-bg)', color: 'var(--alert-critical)', border: '1px solid var(--alert-critical-border)' };
      case 'high':
        return { background: 'var(--alert-high-bg)', color: 'var(--alert-high)', border: '1px solid var(--alert-high-border)' };
      case 'medium':
        return { background: 'var(--alert-medium-bg)', color: 'var(--alert-medium)', border: '1px solid var(--alert-medium-border)' };
      default:
        return { background: 'var(--alert-low-bg)', color: 'var(--alert-low)', border: '1px solid var(--alert-low-border)' };
    }
  };

  return (
    <div className="app-container">
      {/* Animated 2D Network Graph Canvas Background */}
      <NetworkBackground theme={theme} />

      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-top">
          <div className="brand">
            <div className="brand-icon">⚡</div>
            <span>BitSentinel-AI</span>
          </div>
          <div className="nav-menu">
            <button className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
              📊 Dashboard KPIs
            </button>
            <button className={`nav-item ${activeTab === 'stream' ? 'active' : ''}`} onClick={() => setActiveTab('stream')}>
              📡 Live WebSocket Ticker
            </button>
            <button className={`nav-item ${activeTab === 'search' ? 'active' : ''}`} onClick={() => setActiveTab('search')}>
              🔍 Tx & Address Search
            </button>
            <button className={`nav-item ${activeTab === 'network' ? 'active' : ''}`} onClick={() => setActiveTab('network')}>
              🌐 Network Correlation
            </button>
            <button className={`nav-item ${activeTab === 'clusters' ? 'active' : ''}`} onClick={() => setActiveTab('clusters')}>
              🔗 Wallet Clusters (Co-Spend)
            </button>
            <button className={`nav-item ${activeTab === 'reports' ? 'active' : ''}`} onClick={() => setActiveTab('reports')}>
              📈 Model Benchmarks
            </button>
          </div>
        </div>

        {/* Sidebar Footer with Light/Dark Theme Switcher */}
        <div className="sidebar-footer">
          <button 
            className="theme-toggle-btn"
            onClick={() => setTheme(prev => prev === 'light' ? 'dark' : 'light')}
            title={`Switch to ${theme === 'light' ? 'Dark' : 'Light'} Mode`}
            aria-label="Toggle visual theme"
          >
            {theme === 'light' ? '🌙 Switch to Dark' : '☀️ Switch to Light'}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Header Bar */}
        <div className="header-bar">
          <div className="header-title">
            <h1>BitSentinel-AI Monitoring & Telemetry Platform</h1>
            <p>Multimodal Cross-Layer Correlation, Elliptic GraphSAGE GNN & BABD-13 Behavioral ML</p>
          </div>
          <div className="status-badge">
            <div className="status-dot" style={{ background: wsConnected ? 'var(--alert-low)' : 'var(--alert-critical)' }}></div>
            <span>{wsConnected ? 'WebSocket Live Feed Active' : 'REST Mode Active'}</span>
          </div>
        </div>

        {/* Top KPI Cards */}
        {kpis && kpis.status === 'no_predictions_yet' ? (
          <div className="glass-panel" style={{ marginBottom: '24px', borderLeft: '4px solid var(--accent-gold)' }}>
            <h3 style={{ color: 'var(--accent-gold)', margin: 0 }}>⚠️ No Raw Block Predictions Generated Yet</h3>
            <p style={{ color: 'var(--text-secondary)', marginTop: '4px', margin: 0 }}>
              Run <code>python parse_raw_blocks_and_predict.py</code> to extract honest features and populate real KPIs.
            </p>
          </div>
        ) : (
          <div className="kpi-grid">
            <div className="kpi-card kpi-total">
              <span className="kpi-title">Total Scored Txs</span>
              <span className="kpi-value">{kpis ? kpis.total_scored_transactions?.toLocaleString() : '...'}</span>
            </div>
            <div className="kpi-card kpi-alert">
              <span className="kpi-title">High-Risk Alerts</span>
              <span className="kpi-value" style={{ color: 'var(--alert-critical)' }}>
                {kpis ? kpis.high_risk_alerts?.toLocaleString() : '...'}
              </span>
              <span className="kpi-sub" style={{ color: 'var(--alert-critical)' }}>
                {kpis ? `${kpis.risk_ratio_pct}% Alert Ratio` : ''}
              </span>
            </div>
            <div className="kpi-card kpi-monitored">
              <span className="kpi-title">Monitored Volume</span>
              <span className="kpi-value">{kpis ? `${kpis.total_monitored_btc_volume} BTC` : '...'}</span>
            </div>
            <div className="kpi-card kpi-flagged">
              <span className="kpi-title">Flagged Volume</span>
              <span className="kpi-value" style={{ color: 'var(--accent-gold)' }}>
                {kpis ? `${kpis.flagged_high_risk_btc_volume} BTC` : '...'}
              </span>
            </div>
            {kpis && kpis.coverage_pct !== null && kpis.coverage_pct !== undefined && (
              <div className="kpi-card kpi-coverage">
                <span className="kpi-title">Dataset Coverage</span>
                <span className="kpi-value" style={{ color: 'var(--accent-purple)' }}>
                  {kpis.coverage_pct}%
                </span>
                <span className="kpi-sub">
                  {kpis.blocks_processed} / {kpis.blocks_available} blocks
                </span>
              </div>
            )}
          </div>
        )}

        {/* Tab 1: Dashboard */}
        {activeTab === 'dashboard' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>Executive Fraud Risk & Intelligence Summary</h2>
            </div>
            <p style={{ color: 'var(--text-secondary)', marginBottom: '20px', lineHeight: '1.5' }}>
              Real-time cross-layer monitoring across raw blocks (600000–605999), 203,769 Elliptic graph transactions, 46k+ network telemetry records, and 798,934 BABD-13 Bitcoin addresses.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '20px' }}>
              <div className="feature-card">
                <h3 style={{ fontSize: '1rem', color: 'var(--accent-gold)', marginBottom: '8px', fontWeight: 700 }}>
                  🌐 Network↔Blockchain Fusion
                </h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.45' }}>
                  Fuses on-chain graph suspicion with BGP ASN & /24 subnet peer densities, lifting Test F1 to 0.8041 and filtering 100% of legit exchange bursts.
                </p>
              </div>
              <div className="feature-card">
                <h3 style={{ fontSize: '1rem', color: 'var(--accent-cyan)', marginBottom: '8px', fontWeight: 700 }}>
                  ⚡ Elliptic GNN Classification
                </h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.45' }}>
                  Self-Supervised GraphSAGE pretrained on 234k directed edges → Random Forest downstream achieving 0.7905 Test F1 and 0.9219 ROC-AUC.
                </p>
              </div>
              <div className="feature-card">
                <h3 style={{ fontSize: '1rem', color: 'var(--accent-purple)', marginBottom: '8px', fontWeight: 700 }}>
                  🏷️ BABD-13 Multi-Class ML
                </h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: '1.45' }}>
                  Account-Disjoint Stratified Classifier across 10 clean address categories achieving 96.37% Test Accuracy and reduced 5-feature model for honest raw-block scoring.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Live Stream */}
        {activeTab === 'stream' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>📡 Live Blockchain WebSocket Feed</h2>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Streaming from Raw Blocks (Heuristic Score Proxy with Real-Time Explainability)
              </span>
            </div>

            {streamData.length > 0 && streamData[0].stream_note && (
              <div style={{ background: 'var(--bg-panel-nested)', borderLeft: '3px solid var(--accent-cyan)', padding: '10px 14px', borderRadius: '6px', marginBottom: '16px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                ℹ️ {streamData[0].stream_note}
              </div>
            )}

            <table className="stream-table">
              <thead>
                <tr>
                  <th>Block</th>
                  <th>Transaction Hash</th>
                  <th>Value (BTC)</th>
                  <th>Inputs/Outputs</th>
                  <th>Heuristic Score</th>
                  <th>Risk Reason / Explainability</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {streamData.length === 0 ? (
                  <tr>
                    <td colSpan="7" style={{ textAlign: 'center', padding: '28px', color: 'var(--text-muted)' }}>
                      Connecting to Live Fraud WebSocket Stream ...
                    </td>
                  </tr>
                ) : (
                  streamData.map((tx, idx) => (
                    <tr key={idx} className={tx.is_alert ? 'alert-row' : ''}>
                      <td>#{tx.block_height}</td>
                      <td>{tx.tx_hash ? tx.tx_hash.slice(0, 20) + '...' : 'N/A'}</td>
                      <td>{tx.value_btc} BTC</td>
                      <td>{tx.inputs_count} In / {tx.outputs_count} Out</td>
                      <td><strong>{tx.heuristic_score ?? tx.fraud_score}</strong></td>
                      <td style={{ fontSize: '0.82rem', color: tx.is_alert ? 'var(--alert-high)' : 'var(--text-secondary)', maxWidth: '340px' }}>
                        {tx.explanation ? tx.explanation.replace('Rule-Based Heuristic Evaluation: ', '') : 'Normal transfer bounds'}
                      </td>
                      <td>
                        <span className={`alert-tag ${tx.is_alert ? 'high' : 'normal'}`}>
                          {tx.is_alert ? 'HIGH RISK' : 'NORMAL'}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        )}

        {/* Tab 3: Search */}
        {activeTab === 'search' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>🔍 Transaction & Address Risk Search Engine</h2>
            </div>
            <form onSubmit={handleSearch} className="search-box">
              <input
                type="text"
                className="search-input"
                placeholder="Enter Transaction Hash (64 hex) or Bitcoin Address (1..., 3..., bc1...)"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              <button type="submit" className="search-btn">Search Risk Score</button>
            </form>

            {/* Address Co-Spend Cluster Alert Badge */}
            {addressClusterInfo && addressClusterInfo.found && (
              <div style={{ background: 'var(--alert-high-bg)', border: '1px solid var(--alert-high-border)', padding: '16px', borderRadius: '10px', marginBottom: '20px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '1.4rem' }}>🔗</span>
                    <div>
                      <h4 style={{ color: 'var(--alert-high)', margin: 0, fontSize: '0.95rem', fontWeight: 700 }}>
                        Co-Spend Entity Detected ({addressClusterInfo.cluster_id})
                      </h4>
                      <p style={{ color: 'var(--text-secondary)', margin: '2px 0 0 0', fontSize: '0.85rem' }}>
                        This address belongs to a <strong>{addressClusterInfo.size}-address</strong> co-spent wallet entity.
                      </p>
                    </div>
                  </div>
                  <button 
                    onClick={() => toggleClusterExpand(addressClusterInfo.cluster_id)}
                    className="btn-action"
                    style={{ color: 'var(--alert-high)', borderColor: 'var(--alert-high-border)' }}
                  >
                    {expandedClusterId === addressClusterInfo.cluster_id ? 'Hide Clustered Addresses' : `View All ${addressClusterInfo.size} Addresses`}
                  </button>
                </div>

                {expandedClusterId === addressClusterInfo.cluster_id && (
                  <div style={{ marginTop: '14px', paddingTop: '12px', borderTop: '1px solid var(--alert-high-border)' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                      Associated Entity Addresses:
                    </span>
                    <div style={{ maxHeight: '160px', overflowY: 'auto', marginTop: '8px', background: 'var(--bg-code)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                      {addressClusterInfo.addresses.map((addr, aIdx) => (
                        <div key={aIdx} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: addr === addressClusterInfo.address ? 'var(--accent-indigo)' : 'var(--text-primary)', padding: '3px 0' }}>
                          {addr} {addr === addressClusterInfo.address && <span style={{ color: 'var(--alert-high)', fontSize: '0.75rem', fontWeight: 700 }}>(Searched Address)</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {searchResult && (
              <div style={{ background: 'var(--bg-panel-nested)', padding: '22px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                {searchResult.found ? (
                  searchResult.type === 'address' ? (
                    searchResult.risk_score_status === 'scored' ? (
                      <div>
                        <h3 style={{ color: 'var(--accent-indigo)', marginBottom: '8px', fontWeight: 700 }}>
                          🏷️ Bitcoin Address Behavioral Intelligence
                        </h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '14px' }}>
                          Engine: {searchResult.scoring_engine}
                        </p>
                        <div style={{ display: 'flex', gap: '24px', marginBottom: '16px', flexWrap: 'wrap' }}>
                          <div>
                            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Predicted Category:</span>
                            <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--accent-gold)' }}>
                              {searchResult.predicted_category}
                            </div>
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Model Confidence:</span>
                            <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: 'var(--alert-low)' }}>
                              {(searchResult.model_confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* Explainability Block */}
                        {searchResult.explanation && (
                          <div style={{ background: 'var(--bg-card)', borderLeft: '3px solid var(--accent-indigo)', padding: '12px 16px', borderRadius: '6px', marginBottom: '16px', border: '1px solid var(--border-color)', borderLeftWidth: '3px', borderLeftColor: 'var(--accent-indigo)' }}>
                            <span style={{ color: 'var(--accent-indigo)', fontSize: '0.75rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                              🔍 Why Predicted (SHAP Attribution):
                            </span>
                            <p style={{ color: 'var(--text-primary)', fontSize: '0.88rem', margin: '4px 0 0 0', lineHeight: '1.4' }}>
                              {searchResult.explanation}
                            </p>
                          </div>
                        )}

                        <pre className="code-box">
                          {JSON.stringify(searchResult.details, null, 2)}
                        </pre>
                      </div>
                    ) : (
                      <div>
                        <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '8px' }}>
                          ℹ️ Individual Address ML Score: Not Scored
                        </h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', lineHeight: '1.5', margin: 0 }}>
                          {searchResult.message}
                        </p>
                      </div>
                    )
                  ) : (
                    <div>
                      <h3 style={{ color: searchResult.is_high_risk ? 'var(--alert-critical)' : 'var(--alert-low)', marginBottom: '8px', fontWeight: 700 }}>
                        {searchResult.is_high_risk ? '🚨 HIGH HEURISTIC RISK ALERT' : '✓ CLEAN TRANSACTION'}
                      </h3>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '12px' }}>
                        Engine: {searchResult.scoring_engine}
                      </p>
                      <p style={{ fontFamily: 'var(--font-mono)', marginBottom: '12px' }}>
                        Heuristic Score: <strong>{searchResult.heuristic_score}</strong>
                      </p>

                      {/* Explainability Block for Transaction */}
                      {searchResult.explanation && (
                        <div style={{
                          background: searchResult.is_high_risk ? 'var(--alert-critical-bg)' : 'var(--alert-low-bg)',
                          border: `1px solid ${searchResult.is_high_risk ? 'var(--alert-critical-border)' : 'var(--alert-low-border)'}`,
                          borderLeftWidth: '4px',
                          borderLeftColor: searchResult.is_high_risk ? 'var(--alert-critical)' : 'var(--alert-low)',
                          padding: '12px 16px',
                          borderRadius: '6px',
                          marginBottom: '16px'
                        }}>
                          <span style={{ color: searchResult.is_high_risk ? 'var(--alert-critical)' : 'var(--alert-low)', fontSize: '0.75rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            🔍 Why Flagged (Heuristic Rule Factors):
                          </span>
                          <p style={{ color: 'var(--text-primary)', fontSize: '0.88rem', margin: '4px 0 0 0', lineHeight: '1.4' }}>
                            {searchResult.explanation}
                          </p>
                        </div>
                      )}

                      <pre className="code-box">
                        {JSON.stringify(searchResult.details, null, 2)}
                      </pre>
                    </div>
                  )
                ) : (
                  (!addressClusterInfo || !addressClusterInfo.found) && (
                    <div>
                      <p style={{ color: 'var(--text-secondary)', margin: 0 }}>{searchResult.message}</p>
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Network Correlation Alerts (Task 4) */}
        {activeTab === 'network' && (
          <div>
            {/* Section 1: Cross-Layer Cluster Summary */}
            <div className="glass-panel" style={{ marginBottom: '24px' }}>
              <div className="panel-header">
                <h2>🌐 Cross-Layer Coordinated Subnet & ASN Clusters</h2>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Aggregated from BGP Routing Intelligence & Temporal Subnet Density
                </span>
              </div>

              {networkClusters && networkClusters.status === 'not_generated_yet' ? (
                <div style={{ background: 'var(--alert-high-bg)', borderLeft: '4px solid var(--accent-gold)', padding: '16px', borderRadius: '6px' }}>
                  <h3 style={{ color: 'var(--accent-gold)', margin: 0 }}>⚠️ Network Correlation Not Generated Yet</h3>
                  <p style={{ color: 'var(--text-secondary)', margin: '6px 0 0 0', fontSize: '0.9rem' }}>
                    Please run <code>python train_combined_risk_model.py</code> to generate <code>models/network_correlated_alerts.csv</code>.
                  </p>
                </div>
              ) : networkClusters && networkClusters.clusters ? (
                <div>
                  <div style={{ display: 'flex', gap: '20px', marginBottom: '16px' }}>
                    <div style={{ background: 'var(--bg-panel-nested)', padding: '12px 20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Coordinated Subnets Flagged:</span>
                      <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>
                        {networkClusters.total_clusters}
                      </div>
                    </div>
                  </div>

                  <table className="stream-table">
                    <thead>
                      <tr>
                        <th>Subnet (/24)</th>
                        <th>Country</th>
                        <th>BGP ASN / Organization</th>
                        <th>Tx Count</th>
                        <th>Avg Fused Risk</th>
                        <th>Max Fused Risk</th>
                        <th>Time Span</th>
                      </tr>
                    </thead>
                    <tbody>
                      {networkClusters.clusters.map((cl, i) => (
                        <tr key={i}>
                          <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: 'var(--accent-indigo)' }}>
                            {cl.subnet}
                          </td>
                          <td>{cl.country}</td>
                          <td>
                            <span style={{ fontWeight: '600' }}>{cl.asn}</span>
                            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginLeft: '6px' }}>({cl.asn_name})</span>
                          </td>
                          <td><strong>{cl.tx_count}</strong></td>
                          <td>{(cl.avg_fused_prob * 100).toFixed(1)}%</td>
                          <td>
                            <span style={{
                              fontWeight: 'bold',
                              color: cl.max_fused_prob >= 0.70 ? 'var(--alert-critical)' : cl.max_fused_prob >= 0.50 ? 'var(--alert-high)' : 'var(--alert-low)'
                            }}>
                              {(cl.max_fused_prob * 100).toFixed(1)}%
                            </span>
                          </td>
                          <td style={{ color: 'var(--text-secondary)' }}>{cl.time_span_hours} hrs</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: 'var(--text-secondary)' }}>Loading coordinated subnet clusters ...</p>
              )}
            </div>

            {/* Section 2: Top Fused Alerts */}
            <div className="glass-panel">
              <div className="panel-header">
                <h2>🚨 Ranked Multimodal Fused Risk Alerts</h2>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                  Fusing Blockchain Graph ML + BGP Network Telemetry (Click TXID to inspect SHAP explanation)
                </span>
              </div>

              {networkAlerts && networkAlerts.status === 'not_generated_yet' ? (
                <div style={{ background: 'var(--alert-high-bg)', borderLeft: '4px solid var(--accent-gold)', padding: '16px', borderRadius: '6px' }}>
                  <h3 style={{ color: 'var(--accent-gold)', margin: 0 }}>⚠️ Fused Alerts Not Generated Yet</h3>
                  <p style={{ color: 'var(--text-secondary)', margin: '6px 0 0 0', fontSize: '0.9rem' }}>
                    Please run <code>python train_combined_risk_model.py</code> to execute multimodal fusion and generate alerts.
                  </p>
                </div>
              ) : networkAlerts && networkAlerts.alerts ? (
                <div>
                  <div style={{ display: 'flex', gap: '20px', marginBottom: '16px' }}>
                    <div style={{ background: 'var(--bg-panel-nested)', padding: '12px 20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>High-Risk Multimodal Alerts:</span>
                      <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: 'var(--alert-critical)' }}>
                        {networkAlerts.total_matching?.toLocaleString()}
                      </div>
                    </div>
                    <div style={{ background: 'var(--bg-panel-nested)', padding: '12px 20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Displaying Top:</span>
                      <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: 'var(--accent-gold)' }}>
                        {networkAlerts.returned}
                      </div>
                    </div>
                  </div>

                  <table className="stream-table">
                    <thead>
                      <tr>
                        <th>TXID</th>
                        <th>Subnet (/24)</th>
                        <th>ASN / Org</th>
                        <th>Country</th>
                        <th>Blockchain Risk</th>
                        <th>Network Signal</th>
                        <th>Fused Risk</th>
                        <th>Tier</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {networkAlerts.alerts.map((al) => (
                        <React.Fragment key={al.txid}>
                          <tr>
                            <td 
                              onClick={() => toggleNetworkTxExpand(al.txid)}
                              style={{ fontFamily: 'var(--font-mono)', cursor: 'pointer', color: 'var(--accent-indigo)', textDecoration: 'underline', fontWeight: 600 }}
                              title="Click to view SHAP explanation"
                            >
                              {al.txid ? `${String(al.txid).slice(0, 16)}...` : 'N/A'}
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)' }}>{al.src_subnet24}</td>
                            <td>{al.src_asn} <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>({al.src_asn_name})</span></td>
                            <td>{al.src_country}</td>
                            <td>{(al.blockchain_risk_score * 100).toFixed(1)}%</td>
                            <td>
                              <span style={{
                                background: al.is_correlated_cluster ? 'var(--alert-critical-bg)' : 'var(--bg-panel-nested)',
                                color: al.is_correlated_cluster ? 'var(--alert-critical)' : 'var(--text-secondary)',
                                border: `1px solid ${al.is_correlated_cluster ? 'var(--alert-critical-border)' : 'var(--border-color)'}`,
                                padding: '2px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold'
                              }}>
                                {al.is_correlated_cluster ? '⚡ CORRELATED' : 'STANDARD'}
                              </span>
                            </td>
                            <td><strong>{(al.fused_prob * 100).toFixed(1)}%</strong></td>
                            <td>
                              <span style={{ ...getTierBadgeStyle(al.risk_tier), padding: '3px 8px', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold', textTransform: 'uppercase' }}>
                                {al.risk_tier}
                              </span>
                            </td>
                            <td>
                              <button
                                onClick={() => toggleNetworkTxExpand(al.txid)}
                                className="btn-action"
                              >
                                {expandedNetworkTxid === al.txid ? 'Collapse ▲' : 'Explain ▼'}
                              </button>
                            </td>
                          </tr>

                          {/* Expanded Inline XAI Explanation & Evidence */}
                          {expandedNetworkTxid === al.txid && (
                            <tr>
                              <td colSpan="9" style={{ background: 'var(--bg-panel-nested)', padding: '16px' }}>
                                {networkTxDetail[al.txid] ? (
                                  <div>
                                    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderLeftWidth: '4px', borderLeftColor: 'var(--accent-indigo)', padding: '12px 16px', borderRadius: '6px', marginBottom: '14px' }}>
                                      <span style={{ color: 'var(--accent-indigo)', fontSize: '0.75rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                        🔍 Forensic Decision Attribution ({networkTxDetail[al.txid].scoring_engine}):
                                      </span>
                                      <p style={{ color: 'var(--text-primary)', fontSize: '0.9rem', margin: '6px 0 0 0', lineHeight: '1.4' }}>
                                        {networkTxDetail[al.txid].explanation}
                                      </p>
                                    </div>

                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
                                      {/* Top Factors */}
                                      <div style={{ background: 'var(--bg-card)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                                          Top Feature Attributions:
                                        </span>
                                        <div style={{ marginTop: '8px' }}>
                                          {networkTxDetail[al.txid].top_factors?.map((fac, fIdx) => (
                                            <div key={fIdx} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.85rem' }}>
                                              <span style={{ color: 'var(--text-primary)' }}>{fac.label}:</span>
                                              <span style={{ color: fac.contribution > 0 ? 'var(--alert-critical)' : 'var(--alert-low)', fontWeight: 'bold', fontFamily: 'var(--font-mono)' }}>
                                                {fac.contribution > 0 ? '+' : ''}{fac.contribution.toFixed(4)} ({fac.direction})
                                              </span>
                                            </div>
                                          ))}
                                        </div>
                                      </div>

                                      {/* Network Evidence */}
                                      <div style={{ background: 'var(--bg-card)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                                          Network Forensic Evidence:
                                        </span>
                                        <div style={{ marginTop: '8px', fontSize: '0.85rem' }}>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>Source IP:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.src_ip}</span>
                                          </div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>Subnet Peer Count:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.src_subnet24_peer_count} peers</span>
                                          </div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>6h Temporal Cluster Count:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.time_cluster_peer_count} txs</span>
                                          </div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>ASN Infrastructure:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.src_asn} ({networkTxDetail[al.txid].network_evidence?.src_asn_name})</span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                ) : (
                                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Loading forensic attribution details ...</p>
                                )}
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p style={{ color: 'var(--text-secondary)' }}>Loading ranked multimodal alerts ...</p>
              )}
            </div>
          </div>
        )}

        {/* Tab 5: Wallet Clusters */}
        {activeTab === 'clusters' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>🔗 Multi-Input Co-Spend Wallet Entity Clusters</h2>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Union-Find Clustering on Raw Block Common-Input Heuristic
              </span>
            </div>

            {walletClusters && walletClusters.status === 'not_generated_yet' ? (
              <div style={{ background: 'var(--alert-high-bg)', borderLeft: '4px solid var(--accent-gold)', padding: '16px', borderRadius: '6px' }}>
                <h3 style={{ color: 'var(--accent-gold)', margin: 0 }}>⚠️ Wallet Clusters Not Generated Yet</h3>
                <p style={{ color: 'var(--text-secondary)', margin: '6px 0 0 0', fontSize: '0.9rem' }}>
                  Please run <code>python bitcoin_heuristics.py</code> to execute Union-Find co-spend analysis and generate <code>models/wallet_clusters.csv</code>.
                </p>
              </div>
            ) : walletClusters && walletClusters.clusters ? (
              <div>
                <div style={{ display: 'flex', gap: '20px', marginBottom: '16px' }}>
                  <div style={{ background: 'var(--bg-panel-nested)', padding: '12px 20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Total Discovered Multi-Address Entities:</span>
                    <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>
                      {walletClusters.total_clusters?.toLocaleString()}
                    </div>
                  </div>
                  <div style={{ background: 'var(--bg-panel-nested)', padding: '12px 20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>Displaying Top Largest:</span>
                    <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: 'var(--accent-gold)' }}>
                      {walletClusters.displayed_count}
                    </div>
                  </div>
                </div>

                <table className="stream-table">
                  <thead>
                    <tr>
                      <th>Cluster ID</th>
                      <th>Entity Size (Addresses)</th>
                      <th>Lead Clustered Address</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {walletClusters.clusters.map((c) => (
                      <React.Fragment key={c.cluster_id}>
                        <tr>
                          <td><strong>{c.cluster_id}</strong></td>
                          <td>
                            <span style={{ background: 'var(--bg-panel-nested)', color: 'var(--accent-indigo)', padding: '3px 8px', borderRadius: '4px', fontWeight: 'bold', fontSize: '0.85rem', border: '1px solid var(--border-color)' }}>
                              {c.size} Addresses
                            </span>
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>
                            {c.addresses[0] ? `${c.addresses[0].slice(0, 24)}...` : 'N/A'}
                          </td>
                          <td>
                            <button
                              onClick={() => toggleClusterExpand(c.cluster_id)}
                              className="btn-action"
                            >
                              {expandedClusterId === c.cluster_id ? 'Collapse ▲' : 'Expand Addresses ▼'}
                            </button>
                          </td>
                        </tr>
                        {expandedClusterId === c.cluster_id && (
                          <tr>
                            <td colSpan="4" style={{ background: 'var(--bg-panel-nested)', padding: '16px' }}>
                              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                                All Member Addresses in {c.cluster_id} ({c.size} Total):
                              </span>
                              <div style={{ maxHeight: '180px', overflowY: 'auto', marginTop: '8px', background: 'var(--bg-code)', padding: '10px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                                {c.addresses.map((addr, aIdx) => (
                                  <div key={aIdx} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--text-secondary)', padding: '3px 0' }}>
                                    • {addr}
                                  </div>
                                ))}
                              </div>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>Loading wallet clusters from FastAPI backend ...</p>
            )}
          </div>
        )}

        {/* Tab 6: Benchmarks */}
        {activeTab === 'reports' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>📈 Model Evaluation & Benchmarks</h2>
            </div>
            {benchmarks ? (
              <div>
                <h3 style={{ fontSize: '1rem', color: 'var(--accent-gold)', marginBottom: '12px', fontWeight: 700 }}>Elliptic Benchmark Results</h3>
                <table className="stream-table" style={{ marginBottom: '24px' }}>
                  <thead>
                    <tr><th>Model</th><th>Split</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>PR-AUC</th></tr>
                  </thead>
                  <tbody>
                    {benchmarks.elliptic_benchmarks.map((row, i) => (
                      <tr key={i}>
                        <td>{row.Model}</td><td>{row.Split}</td><td>{row.Precision?.toFixed(4)}</td>
                        <td>{row.Recall?.toFixed(4)}</td><td><strong>{row['F1-Score']?.toFixed(4)}</strong></td>
                        <td>{row['ROC-AUC']?.toFixed(4)}</td><td>{row['PR-AUC']?.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                <h3 style={{ fontSize: '1rem', color: 'var(--accent-cyan)', marginBottom: '12px', fontWeight: 700 }}>BABD-13 Benchmark Results</h3>
                <table className="stream-table">
                  <thead>
                    <tr><th>Model</th><th>Split</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th><th>Macro Precision</th><th>Macro Recall</th></tr>
                  </thead>
                  <tbody>
                    {benchmarks.babd13_benchmarks.map((row, i) => (
                      <tr key={i}>
                        <td>{row.Model}</td><td>{row.Split}</td><td>{row.Accuracy?.toFixed(4)}</td>
                        <td><strong>{row['Macro F1']?.toFixed(4)}</strong></td><td>{row['Weighted F1']?.toFixed(4)}</td>
                        <td>{row['Macro Precision']?.toFixed(4)}</td><td>{row['Macro Recall']?.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p style={{ color: 'var(--text-secondary)' }}>Loading benchmark results from FastAPI backend ...</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
