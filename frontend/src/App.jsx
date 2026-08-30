import React, { useState, useEffect } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8005';
const WS_URL = import.meta.env.VITE_WS_URL || (
  API_BASE.startsWith('https://') 
    ? API_BASE.replace('https://', 'wss://') + '/ws/stream'
    : API_BASE.replace('http://', 'ws://') + '/ws/stream'
);

export default function App() {
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

  // Load REST KPIs, Benchmarks, and Wallet Clusters
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

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand">
          <span style={{ fontSize: '1.5rem' }}>⚡</span>
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
          <button className={`nav-item ${activeTab === 'clusters' ? 'active' : ''}`} onClick={() => setActiveTab('clusters')}>
            🔗 Wallet Clusters (Co-Spend)
          </button>
          <button className={`nav-item ${activeTab === 'reports' ? 'active' : ''}`} onClick={() => setActiveTab('reports')}>
            📈 Model Benchmarks
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        {/* Header Bar */}
        <div className="header-bar">
          <div className="header-title">
            <h1>BitSentinel-AI Monitoring & Telemetry Platform</h1>
            <p>Elliptic GraphSAGE GNN, BABD-13 Address ML & Multi-Input Co-Spend Clustering</p>
          </div>
          <div className="status-badge">
            <div className="status-dot" style={{ background: wsConnected ? '#00e676' : '#ff4d4d' }}></div>
            <span>{wsConnected ? 'WebSocket Live Feed Active' : 'REST Mode Active'}</span>
          </div>
        </div>

        {/* Top KPI Cards */}
        {kpis && kpis.status === 'no_predictions_yet' ? (
          <div className="glass-panel" style={{ marginBottom: '24px', borderLeft: '4px solid #f7931a' }}>
            <h3 style={{ color: '#f7931a', margin: 0 }}>⚠️ No Raw Block Predictions Generated Yet</h3>
            <p style={{ color: '#8b98a5', marginTop: '4px', margin: 0 }}>
              Run <code>python parse_raw_blocks_and_predict.py</code> to extract honest features and populate real KPIs.
            </p>
          </div>
        ) : (
          <div className="kpi-grid">
            <div className="kpi-card">
              <span className="kpi-title">Total Scored Txs</span>
              <span className="kpi-value">{kpis ? kpis.total_scored_transactions?.toLocaleString() : '...'}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-title">High-Risk Alerts</span>
              <span className="kpi-value" style={{ color: '#ff4d4d' }}>
                {kpis ? kpis.high_risk_alerts?.toLocaleString() : '...'}
              </span>
              <span className="kpi-sub">{kpis ? `${kpis.risk_ratio_pct}% Alert Ratio` : ''}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-title">Monitored Volume</span>
              <span className="kpi-value">{kpis ? `${kpis.total_monitored_btc_volume} BTC` : '...'}</span>
            </div>
            <div className="kpi-card">
              <span className="kpi-title">Flagged Volume</span>
              <span className="kpi-value" style={{ color: '#f7931a' }}>
                {kpis ? `${kpis.flagged_high_risk_btc_volume} BTC` : '...'}
              </span>
            </div>
          </div>
        )}

        {/* Tab 1: Dashboard */}
        {activeTab === 'dashboard' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>Executive Fraud Risk & Intelligence Summary</h2>
            </div>
            <p style={{ color: '#8b98a5', marginBottom: '16px' }}>
              Real-time monitoring across raw blocks (600000–605999), 203,769 Elliptic graph transactions, and 798,934 BABD-13 Bitcoin addresses.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
              <div style={{ background: '#0a0d14', padding: '16px', borderRadius: '8px' }}>
                <h3 style={{ fontSize: '1rem', color: '#f7931a', marginBottom: '8px' }}>Elliptic GNN Classification</h3>
                <p style={{ fontSize: '0.85rem', color: '#8b98a5' }}>
                  Self-Supervised GraphSAGE pretrained on 234k directed edges → Random Forest downstream achieving 0.7905 Test F1 and 0.9219 ROC-AUC.
                </p>
              </div>
              <div style={{ background: '#0a0d14', padding: '16px', borderRadius: '8px' }}>
                <h3 style={{ fontSize: '1rem', color: '#00d2ff', marginBottom: '8px' }}>BABD-13 Multi-Class Classifier</h3>
                <p style={{ fontSize: '0.85rem', color: '#8b98a5' }}>
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
              <span style={{ fontSize: '0.85rem', color: '#8b98a5' }}>
                Streaming from Raw Blocks (Heuristic Score Proxy with Real-Time Explainability)
              </span>
            </div>
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
                    <td colSpan="7" style={{ textAlign: 'center', padding: '24px', color: '#8b98a5' }}>
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
                      <td style={{ fontSize: '0.8rem', color: tx.is_alert ? '#fb923c' : '#94a3b8', maxWidth: '320px' }}>
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
              <div style={{ background: '#1c1917', border: '1px solid #f97316', padding: '16px', borderRadius: '8px', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '1.4rem' }}>🔗</span>
                    <div>
                      <h4 style={{ color: '#f97316', margin: 0, fontSize: '0.95rem' }}>
                        Co-Spend Entity Detected ({addressClusterInfo.cluster_id})
                      </h4>
                      <p style={{ color: '#d6d3d1', margin: '2px 0 0 0', fontSize: '0.85rem' }}>
                        This address belongs to a <strong>{addressClusterInfo.size}-address</strong> co-spent wallet entity.
                      </p>
                    </div>
                  </div>
                  <button 
                    onClick={() => toggleClusterExpand(addressClusterInfo.cluster_id)}
                    style={{ background: '#292524', color: '#fb923c', border: '1px solid #f97316', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.8rem' }}
                  >
                    {expandedClusterId === addressClusterInfo.cluster_id ? 'Hide Clustered Addresses' : `View All ${addressClusterInfo.size} Addresses`}
                  </button>
                </div>

                {expandedClusterId === addressClusterInfo.cluster_id && (
                  <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #44403c' }}>
                    <span style={{ fontSize: '0.75rem', color: '#a8a29e', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      Associated Entity Addresses:
                    </span>
                    <div style={{ maxHeight: '160px', overflowY: 'auto', marginTop: '6px', background: '#0c0a09', padding: '8px', borderRadius: '4px' }}>
                      {addressClusterInfo.addresses.map((addr, aIdx) => (
                        <div key={aIdx} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: addr === addressClusterInfo.address ? '#38bdf8' : '#e2e8f0', padding: '3px 0' }}>
                          {addr} {addr === addressClusterInfo.address && <span style={{ color: '#f97316', fontSize: '0.75rem' }}>(Searched Address)</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {searchResult && (
              <div style={{ background: '#0a0d14', padding: '20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                {searchResult.found ? (
                  searchResult.type === 'address' ? (
                    searchResult.risk_score_status === 'scored' ? (
                      <div>
                        <h3 style={{ color: '#00d2ff', marginBottom: '8px' }}>
                          🏷️ Bitcoin Address Behavioral Intelligence
                        </h3>
                        <p style={{ color: '#8b98a5', fontSize: '0.85rem', marginBottom: '12px' }}>
                          Engine: {searchResult.scoring_engine}
                        </p>
                        <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
                          <div>
                            <span style={{ color: '#8b98a5', fontSize: '0.8rem' }}>Predicted Category:</span>
                            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#ff9800' }}>
                              {searchResult.predicted_category}
                            </div>
                          </div>
                          <div>
                            <span style={{ color: '#8b98a5', fontSize: '0.8rem' }}>Model Confidence:</span>
                            <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#00e676' }}>
                              {(searchResult.model_confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* Explainability Block */}
                        {searchResult.explanation && (
                          <div style={{ background: '#172554', borderLeft: '3px solid #38bdf8', padding: '10px 14px', borderRadius: '4px', marginBottom: '16px' }}>
                            <span style={{ color: '#93c5fd', fontSize: '0.75rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                              🔍 Why Predicted (SHAP Attribution):
                            </span>
                            <p style={{ color: '#e0f2fe', fontSize: '0.85rem', margin: '4px 0 0 0' }}>
                              {searchResult.explanation}
                            </p>
                          </div>
                        )}

                        <pre style={{ background: '#161b26', padding: '12px', borderRadius: '6px', fontSize: '0.85rem', overflowX: 'auto' }}>
                          {JSON.stringify(searchResult.details, null, 2)}
                        </pre>
                      </div>
                    ) : (
                      <div>
                        <h3 style={{ color: '#38bdf8', marginBottom: '8px' }}>
                          ℹ️ Individual Address ML Score: Not Scored
                        </h3>
                        <p style={{ color: '#94a3b8', fontSize: '0.9rem', lineHeight: '1.5', margin: 0 }}>
                          {searchResult.message}
                        </p>
                      </div>
                    )
                  ) : (
                    <div>
                      <h3 style={{ color: searchResult.is_high_risk ? '#ff4d4d' : '#00e676', marginBottom: '8px' }}>
                        {searchResult.is_high_risk ? '🚨 HIGH HEURISTIC RISK ALERT' : '✓ CLEAN TRANSACTION'}
                      </h3>
                      <p style={{ color: '#8b98a5', fontSize: '0.85rem', marginBottom: '12px' }}>
                        Engine: {searchResult.scoring_engine}
                      </p>
                      <p style={{ fontFamily: 'var(--font-mono)', marginBottom: '8px' }}>
                        Heuristic Score: <strong>{searchResult.heuristic_score}</strong>
                      </p>

                      {/* Explainability Block for Transaction */}
                      {searchResult.explanation && (
                        <div style={{ background: searchResult.is_high_risk ? '#450a0a' : '#052e16', borderLeft: `3px solid ${searchResult.is_high_risk ? '#ef4444' : '#22c55e'}`, padding: '10px 14px', borderRadius: '4px', marginBottom: '16px' }}>
                          <span style={{ color: searchResult.is_high_risk ? '#fca5a5' : '#86efac', fontSize: '0.75rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                            🔍 Why Flagged (Heuristic Rule Factors):
                          </span>
                          <p style={{ color: searchResult.is_high_risk ? '#fee2e2' : '#dcfce7', fontSize: '0.85rem', margin: '4px 0 0 0' }}>
                            {searchResult.explanation}
                          </p>
                        </div>
                      )}

                      <pre style={{ background: '#161b26', padding: '12px', borderRadius: '6px', fontSize: '0.85rem', overflowX: 'auto' }}>
                        {JSON.stringify(searchResult.details, null, 2)}
                      </pre>
                    </div>
                  )
                ) : (
                  (!addressClusterInfo || !addressClusterInfo.found) && (
                    <div>
                      <p style={{ color: '#8b98a5', margin: 0 }}>{searchResult.message}</p>
                    </div>
                  )
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Wallet Clusters */}
        {activeTab === 'clusters' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>🔗 Multi-Input Co-Spend Wallet Entity Clusters</h2>
              <span style={{ fontSize: '0.85rem', color: '#8b98a5' }}>
                Union-Find Clustering on Raw Block Common-Input Heuristic
              </span>
            </div>

            {walletClusters && walletClusters.status === 'not_generated_yet' ? (
              <div style={{ background: '#1c1917', borderLeft: '4px solid #f97316', padding: '16px', borderRadius: '6px' }}>
                <h3 style={{ color: '#f97316', margin: 0 }}>⚠️ Wallet Clusters Not Generated Yet</h3>
                <p style={{ color: '#a8a29e', margin: '6px 0 0 0', fontSize: '0.9rem' }}>
                  Please run <code>python bitcoin_heuristics.py</code> to execute Union-Find co-spend analysis and generate <code>models/wallet_clusters.csv</code>.
                </p>
              </div>
            ) : walletClusters && walletClusters.clusters ? (
              <div>
                <div style={{ display: 'flex', gap: '20px', marginBottom: '16px' }}>
                  <div style={{ background: '#0a0d14', padding: '12px 20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <span style={{ color: '#8b98a5', fontSize: '0.8rem' }}>Total Discovered Multi-Address Entities:</span>
                    <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: '#38bdf8' }}>
                      {walletClusters.total_clusters?.toLocaleString()}
                    </div>
                  </div>
                  <div style={{ background: '#0a0d14', padding: '12px 20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                    <span style={{ color: '#8b98a5', fontSize: '0.8rem' }}>Displaying Top Largest:</span>
                    <div style={{ fontSize: '1.3rem', fontWeight: 'bold', color: '#f7931a' }}>
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
                            <span style={{ background: '#1e293b', color: '#38bdf8', padding: '3px 8px', borderRadius: '4px', fontWeight: 'bold', fontSize: '0.85rem' }}>
                              {c.size} Addresses
                            </span>
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>
                            {c.addresses[0] ? `${c.addresses[0].slice(0, 24)}...` : 'N/A'}
                          </td>
                          <td>
                            <button
                              onClick={() => toggleClusterExpand(c.cluster_id)}
                              style={{ background: '#1e293b', color: '#e2e8f0', border: '1px solid #475569', padding: '4px 10px', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem' }}
                            >
                              {expandedClusterId === c.cluster_id ? 'Collapse ▲' : 'Expand Addresses ▼'}
                            </button>
                          </td>
                        </tr>
                        {expandedClusterId === c.cluster_id && (
                          <tr>
                            <td colSpan="4" style={{ background: '#0a0d14', padding: '16px' }}>
                              <span style={{ fontSize: '0.8rem', color: '#8b98a5', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                All Member Addresses in {c.cluster_id} ({c.size} Total):
                              </span>
                              <div style={{ maxHeight: '180px', overflowY: 'auto', marginTop: '8px', background: '#161b26', padding: '10px', borderRadius: '6px' }}>
                                {c.addresses.map((addr, aIdx) => (
                                  <div key={aIdx} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: '#94a3b8', padding: '3px 0' }}>
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
              <p style={{ color: '#8b98a5' }}>Loading wallet clusters from FastAPI backend ...</p>
            )}
          </div>
        )}

        {/* Tab 5: Benchmarks */}
        {activeTab === 'reports' && (
          <div className="glass-panel">
            <div className="panel-header">
              <h2>📈 Model Evaluation & Benchmarks</h2>
            </div>
            {benchmarks ? (
              <div>
                <h3 style={{ fontSize: '1rem', color: '#f7931a', marginBottom: '12px' }}>Elliptic Benchmark Results</h3>
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

                <h3 style={{ fontSize: '1rem', color: '#00d2ff', marginBottom: '12px' }}>BABD-13 Benchmark Results</h3>
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
              <p style={{ color: '#8b98a5' }}>Loading benchmark results from FastAPI backend ...</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
