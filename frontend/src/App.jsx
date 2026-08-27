import React, { useState, useEffect } from 'react';

const API_BASE = 'http://localhost:8005';
const WS_URL = 'ws://localhost:8005/ws/stream';

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [kpis, setKpis] = useState(null);
  const [streamData, setStreamData] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResult, setSearchResult] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);

  // Load REST KPIs
  useEffect(() => {
    fetch(`${API_BASE}/api/kpis`)
      .then(res => res.json())
      .then(data => setKpis(data))
      .catch(err => console.log('API KPI Load Error:', err));

    fetch(`${API_BASE}/api/benchmarks`)
      .then(res => res.json())
      .then(data => setBenchmarks(data))
      .catch(err => console.log('API Benchmarks Load Error:', err));
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
    if (!searchQuery) return;
    fetch(`${API_BASE}/api/search?q=${encodeURIComponent(searchQuery)}`)
      .then(res => res.json())
      .then(data => setSearchResult(data))
      .catch(err => setSearchResult({ found: false, message: 'Search failed' }));
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
            <p>Elliptic GraphSAGE GNN & BABD-13 Address Classification Engine</p>
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
                  Self-Supervised GraphSAGE pretrained on 234k directed edges -> Random Forest downstream achieving 0.7905 Test F1 and 0.9219 ROC-AUC.
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
                Streaming from Raw Blocks (Heuristic Score Proxy)
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
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {streamData.length === 0 ? (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', padding: '24px', color: '#8b98a5' }}>
                      Connecting to Live Fraud WebSocket Stream ...
                    </td>
                  </tr>
                ) : (
                  streamData.map((tx, idx) => (
                    <tr key={idx} className={tx.is_alert ? 'alert-row' : ''}>
                      <td>#{tx.block_height}</td>
                      <td>{tx.tx_hash ? tx.tx_hash.slice(0, 24) + '...' : 'N/A'}</td>
                      <td>{tx.value_btc} BTC</td>
                      <td>{tx.inputs_count} In / {tx.outputs_count} Out</td>
                      <td><strong>{tx.heuristic_score ?? tx.fraud_score}</strong></td>
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

            {searchResult && (
              <div style={{ background: '#0a0d14', padding: '20px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                {searchResult.found ? (
                  searchResult.type === 'address' ? (
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
                      <pre style={{ background: '#161b26', padding: '12px', borderRadius: '6px', fontSize: '0.85rem', overflowX: 'auto' }}>
                        {JSON.stringify(searchResult.details, null, 2)}
                      </pre>
                    </div>
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
                      <pre style={{ background: '#161b26', padding: '12px', borderRadius: '6px', fontSize: '0.85rem', overflowX: 'auto' }}>
                        {JSON.stringify(searchResult.details, null, 2)}
                      </pre>
                    </div>
                  )
                ) : (
                  <div>
                    <p style={{ color: '#8b98a5' }}>{searchResult.message}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Tab 4: Benchmarks */}
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
