import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  LayoutDashboard,
  Radio,
  Search,
  Network,
  Layers,
  BarChart3,
  Activity,
  AlertTriangle,
  Zap,
  Cpu,
  Tag,
  Info,
  Link2,
  Fingerprint,
  AlertOctagon,
  CheckCircle2,
  Sparkles,
  FileSearch,
  ChevronDown,
  ChevronUp,
  Share2,
  Database,
  Sliders,
  TrendingUp,
  PieChart as PieIcon,
  Filter,
  ArrowRight
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine
} from 'recharts';
import GraphView from './GraphView';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8005';
const WS_URL = import.meta.env.VITE_WS_URL || (
  API_BASE.startsWith('https://') 
    ? API_BASE.replace('https://', 'wss://') + '/ws/stream'
    : API_BASE.replace('http://', 'ws://') + '/ws/stream'
);

const DEFAULT_CHART_COLORS = {
  gold: '#f7931a',
  cyan: '#00d2ff',
  purple: '#c084fc',
  indigo: '#818cf8',
  critical: '#ff4d4d',
  high: '#fb923c',
  medium: '#facc15',
  low: '#00e676',
  textSecondary: '#8b98a5',
  borderColor: 'rgba(255, 255, 255, 0.08)',
  bgCard: 'rgba(22, 27, 38, 0.75)',
};

// Helper for formatting big numbers
const formatCompactNumber = (num) => {
  if (num === null || num === undefined) return '0';
  if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'k';
  return Number(num).toLocaleString();
};

const formatBtcVolume = (val) => {
  if (val === null || val === undefined) return '0 BTC';
  if (val >= 1e6) return (val / 1e6).toFixed(1) + 'M BTC';
  if (val >= 1e3) return (val / 1e3).toFixed(1) + 'k BTC';
  return Number(val).toFixed(2) + ' BTC';
};

// Reusable Segmented Pill Selector (Top 10 | Top 25 | All)
function PillSelector({ options = [10, 25, 'all'], value, onChange, label = 'Show:' }) {
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
      {label && <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)', fontWeight: 600 }}>{label}</span>}
      <div className="filter-pills">
        {options.map((opt) => {
          const val = opt === 'all' ? 99999 : Number(opt);
          const isSelected = value === val || (opt === 'all' && value >= 99999);
          const displayText = opt === 'all' ? 'All' : `Top ${opt}`;
          return (
            <button
              key={String(opt)}
              className={`pill-btn ${isSelected ? 'active' : ''}`}
              onClick={() => onChange(val)}
              type="button"
            >
              {displayText}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// Dark Glassmorphic Custom Tooltip Component
const DarkChartTooltip = ({ active, payload, label, valuePrefix = '', valueSuffix = '', formatter }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: 'rgba(14, 18, 26, 0.95)',
        border: '1px solid rgba(255, 255, 255, 0.12)',
        borderRadius: '8px',
        padding: '10px 14px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)',
        backdropFilter: 'blur(8px)',
        fontSize: '0.82rem',
        color: '#f0f4f8',
        zIndex: 100
      }}>
        {label && <div style={{ fontWeight: 'bold', marginBottom: '6px', color: '#ffffff' }}>{label}</div>}
        {payload.map((entry, index) => (
          <div key={`item-${index}`} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '3px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: entry.color || entry.fill }}></span>
            <span style={{ color: '#8b98a5' }}>{entry.name}:</span>
            <span style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
              {formatter ? formatter(entry.value, entry.name, entry) : `${valuePrefix}${typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}${valueSuffix}`}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

// Shared Factor Bar Chart Component
function FactorBarChart({ factors, chartColors }) {
  if (!factors || !Array.isArray(factors) || factors.length === 0) return null;

  const data = factors.map((f, idx) => {
    const label = f.label || f.feature || f.factor || `Factor #${idx + 1}`;
    const contrib = typeof f.contribution === 'number' ? f.contribution : 0;
    const isRisk = f.direction === 'risk' || contrib > 0;
    return {
      label,
      contribution: contrib,
      rawVal: f.value,
      isRisk,
      fill: isRisk ? chartColors.critical : chartColors.low
    };
  });

  return (
    <div style={{ marginTop: '14px', marginBottom: '14px', background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px 14px' }}>
      <div style={{ fontSize: '0.76rem', color: chartColors.textSecondary, fontWeight: 700, marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        📊 Decision Factor Attribution Delta (SHAP / Heuristic Impact)
      </div>
      <div style={{ width: '100%', height: Math.max(140, data.length * 36) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            layout="vertical"
            data={data}
            margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" horizontal={false} />
            <XAxis
              type="number"
              tick={{ fill: chartColors.textSecondary, fontSize: 11, fontFamily: 'var(--font-mono)' }}
              axisLine={{ stroke: chartColors.borderColor }}
              tickLine={{ stroke: chartColors.borderColor }}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={150}
              tick={{ fill: chartColors.textSecondary, fontSize: 11 }}
              axisLine={{ stroke: chartColors.borderColor }}
              tickLine={false}
            />
            <ReferenceLine x={0} stroke={chartColors.textSecondary} strokeWidth={1} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload;
                  return (
                    <div style={{
                      background: 'rgba(14, 18, 26, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.12)',
                      borderRadius: '8px',
                      padding: '8px 12px',
                      fontSize: '0.8rem',
                      color: '#f0f4f8'
                    }}>
                      <div style={{ fontWeight: 'bold', color: '#ffffff' }}>{d.label}</div>
                      {d.rawVal !== undefined && (
                        <div style={{ color: '#8b98a5', marginTop: '2px' }}>Feature Value: <span style={{ color: '#f0f4f8', fontFamily: 'var(--font-mono)' }}>{String(d.rawVal)}</span></div>
                      )}
                      <div style={{ color: d.isRisk ? chartColors.critical : chartColors.low, marginTop: '4px', fontWeight: 700 }}>
                        Attribution: {d.contribution > 0 ? '+' : ''}{Number(d.contribution).toFixed(4)} ({d.isRisk ? 'Pushed toward Risk' : 'Protective / Safe Signal'})
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="contribution" radius={[4, 4, 4, 4]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

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

  // Graph Investigation State
  const [graphEntityId, setGraphEntityId] = useState('37sczWKSEWeqJPXj9MijJa2zQhYVqoQ2AG');
  const [graphInput, setGraphInput] = useState('');

  // Network Correlation Alerts & Clusters State
  const [networkAlerts, setNetworkAlerts] = useState(null);
  const [networkClusters, setNetworkClusters] = useState(null);
  const [expandedNetworkTxid, setExpandedNetworkTxid] = useState(null);
  const [networkTxDetail, setNetworkTxDetail] = useState({});

  // Classification & Top Filter Limits (Top 10 | Top 25 | All)
  const [streamLimit, setStreamLimit] = useState(25);
  const [subnetLimit, setSubnetLimit] = useState(10);
  const [networkAlertsLimit, setNetworkAlertsLimit] = useState(10);
  const [clustersLimit, setClustersLimit] = useState(10);
  const [benchmarksLimit, setBenchmarksLimit] = useState(99999);

  // Memoized Chart Colors read dynamically from CSS Variables
  const [chartColors, setChartColors] = useState(DEFAULT_CHART_COLORS);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const style = getComputedStyle(document.documentElement);
      const getVal = (varName, fallback) => style.getPropertyValue(varName).trim() || fallback;
      setChartColors({
        gold: getVal('--accent-gold', DEFAULT_CHART_COLORS.gold),
        cyan: getVal('--accent-cyan', DEFAULT_CHART_COLORS.cyan),
        purple: getVal('--accent-purple', DEFAULT_CHART_COLORS.purple),
        indigo: getVal('--accent-indigo', DEFAULT_CHART_COLORS.indigo),
        critical: getVal('--alert-critical', DEFAULT_CHART_COLORS.critical),
        high: getVal('--alert-high', DEFAULT_CHART_COLORS.high),
        medium: getVal('--alert-medium', DEFAULT_CHART_COLORS.medium),
        low: getVal('--alert-low', DEFAULT_CHART_COLORS.low),
        textSecondary: getVal('--text-secondary', DEFAULT_CHART_COLORS.textSecondary),
        borderColor: getVal('--border-color', DEFAULT_CHART_COLORS.borderColor),
        bgCard: getVal('--bg-card', DEFAULT_CHART_COLORS.bgCard),
      });
    }
  }, []);

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

  const navigateToGraph = (entity) => {
    if (!entity) return;
    setGraphEntityId(entity);
    setGraphInput(entity);
    setActiveTab('graph');
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
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <ShieldAlert size={18} color="var(--accent-gold)" />
          </div>
          <span>BitSentinel-AI</span>
        </div>
        <div className="nav-menu">
          <button className={`nav-item ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <LayoutDashboard size={17} />
            <span>Dashboard KPIs</span>
          </button>
          <button className={`nav-item ${activeTab === 'stream' ? 'active' : ''}`} onClick={() => setActiveTab('stream')}>
            <Radio size={17} />
            <span>Live WebSocket Ticker</span>
          </button>
          <button className={`nav-item ${activeTab === 'search' ? 'active' : ''}`} onClick={() => setActiveTab('search')}>
            <Search size={17} />
            <span>Tx & Address Search</span>
          </button>
          <button className={`nav-item ${activeTab === 'graph' ? 'active' : ''}`} onClick={() => setActiveTab('graph')}>
            <Share2 size={17} />
            <span>Graph Investigation</span>
          </button>
          <button className={`nav-item ${activeTab === 'network' ? 'active' : ''}`} onClick={() => setActiveTab('network')}>
            <Network size={17} />
            <span>Network Correlation</span>
          </button>
          <button className={`nav-item ${activeTab === 'clusters' ? 'active' : ''}`} onClick={() => setActiveTab('clusters')}>
            <Layers size={17} />
            <span>Wallet Clusters</span>
          </button>
          <button className={`nav-item ${activeTab === 'reports' ? 'active' : ''}`} onClick={() => setActiveTab('reports')}>
            <BarChart3 size={17} />
            <span>Model Benchmarks</span>
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
            <Activity size={13} style={{ marginLeft: '2px' }} />
            <span>{wsConnected ? 'WebSocket Live Feed Active' : 'REST Mode Active'}</span>
          </div>
        </div>

        {/* Top KPI Cards (Compact) */}
        {kpis && kpis.status === 'no_predictions_yet' ? (
          <div className="glass-panel" style={{ marginBottom: '18px', borderLeft: '4px solid var(--accent-gold)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={18} color="var(--accent-gold)" />
              <h3 style={{ color: 'var(--accent-gold)', margin: 0, fontSize: '0.95rem' }}>No Raw Block Predictions Generated Yet</h3>
            </div>
            <p style={{ color: 'var(--text-secondary)', marginTop: '4px', margin: 0, fontSize: '0.85rem' }}>
              Run <code>python src/parse_raw_blocks_and_predict.py</code> to extract honest features and populate real KPIs.
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
              <span className="kpi-value">{kpis ? `${formatBtcVolume(kpis.total_monitored_btc_volume)}` : '...'}</span>
            </div>
            <div className="kpi-card kpi-flagged">
              <span className="kpi-title">Flagged Volume</span>
              <span className="kpi-value" style={{ color: 'var(--accent-gold)' }}>
                {kpis ? `${formatBtcVolume(kpis.flagged_high_risk_btc_volume)}` : '...'}
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
            <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', lineHeight: '1.45', fontSize: '0.88rem' }}>
              Real-time cross-layer monitoring across raw blocks (600000–605999), 203,769 Elliptic graph transactions, 46k+ network telemetry records, and 798,934 BABD-13 Bitcoin addresses.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
              <div className="feature-card">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <Share2 size={16} color="var(--accent-gold)" />
                  <h3 style={{ fontSize: '0.92rem', color: 'var(--accent-gold)', margin: 0, fontWeight: 700 }}>
                    Network↔Blockchain Fusion
                  </h3>
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  Fuses on-chain graph suspicion with BGP ASN & /24 subnet peer densities, lifting Test F1 to 0.8041 and filtering 100% of legit exchange bursts.
                </p>
              </div>
              <div className="feature-card">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <Cpu size={16} color="var(--accent-cyan)" />
                  <h3 style={{ fontSize: '0.92rem', color: 'var(--accent-cyan)', margin: 0, fontWeight: 700 }}>
                    Elliptic GNN Classification
                  </h3>
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  Self-Supervised GraphSAGE pretrained on 234k directed edges → Random Forest downstream achieving 0.7905 Test F1 and 0.9219 ROC-AUC.
                </p>
              </div>
              <div className="feature-card">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <Tag size={16} color="var(--accent-purple)" />
                  <h3 style={{ fontSize: '0.92rem', color: 'var(--accent-purple)', margin: 0, fontWeight: 700 }}>
                    BABD-13 Multi-Class ML
                  </h3>
                </div>
                <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
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
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Radio size={20} color="var(--accent-cyan)" />
                <h2>Live Blockchain WebSocket Feed</h2>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                <PillSelector
                  options={[10, 25, 'all']}
                  value={streamLimit}
                  onChange={setStreamLimit}
                  label="Display Limit:"
                />
              </div>
            </div>

            {streamData.length > 0 && streamData[0].stream_note && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-panel-nested)', borderLeft: '3px solid var(--accent-cyan)', padding: '8px 12px', borderRadius: '6px', marginBottom: '14px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <Info size={15} color="var(--accent-cyan)" style={{ flexShrink: 0 }} />
                <span>{streamData[0].stream_note}</span>
              </div>
            )}

            {/* Rolling Risk Line Chart for Filtered Window */}
            {streamData.length > 0 && (
              <div style={{ background: 'var(--bg-panel-nested)', border: '1px solid var(--border-color)', borderRadius: '10px', padding: '12px 14px', marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <TrendingUp size={15} color="var(--accent-cyan)" />
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-heading)', fontWeight: 700 }}>
                      Live Heuristic Risk Trajectory (Last {Math.min(streamLimit, streamData.length)} Txs)
                    </span>
                  </div>
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    Red Dots: High-Risk Threshold (≥ 0.70)
                  </span>
                </div>
                <div style={{ width: '100%', height: 180 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={streamData.slice(0, streamLimit).reverse().map((tx, idx) => ({
                        idx: `#${idx + 1}`,
                        txHash: tx.tx_hash ? tx.tx_hash.slice(0, 8) + '...' : `tx-${idx}`,
                        score: parseFloat(tx.heuristic_score ?? tx.fraud_score ?? 0),
                        isAlert: Boolean(tx.is_alert || (tx.heuristic_score ?? tx.fraud_score) >= 0.70),
                        valueBtc: tx.value_btc
                      }))}
                      margin={{ top: 10, right: 25, left: -10, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
                      <XAxis dataKey="idx" tick={{ fill: chartColors.textSecondary, fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: chartColors.borderColor }} />
                      <YAxis domain={[0, 1.0]} tick={{ fill: chartColors.textSecondary, fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={{ stroke: chartColors.borderColor }} />
                      <ReferenceLine y={0.70} stroke={chartColors.high} strokeDasharray="3 3" label={{ value: '0.70 Alert Threshold', fill: chartColors.high, fontSize: 10, position: 'insideTopRight' }} />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const d = payload[0].payload;
                            return (
                              <div style={{
                                background: 'rgba(14, 18, 26, 0.95)',
                                border: `1px solid ${d.isAlert ? chartColors.critical : 'rgba(255, 255, 255, 0.12)'}`,
                                borderRadius: '8px',
                                padding: '8px 12px',
                                fontSize: '0.8rem',
                                color: '#f0f4f8'
                              }}>
                                <div style={{ fontWeight: 'bold', color: '#ffffff' }}>Tx {d.txHash} ({d.idx})</div>
                                <div style={{ color: '#8b98a5', marginTop: '2px' }}>Value: <span style={{ color: '#f0f4f8' }}>{d.valueBtc} BTC</span></div>
                                <div style={{ color: d.isAlert ? chartColors.critical : chartColors.low, marginTop: '2px', fontWeight: 700 }}>
                                  Score: {d.score.toFixed(4)} ({d.isAlert ? 'HIGH RISK ALERT' : 'NORMAL'})
                                </div>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        name="Heuristic Score"
                        stroke={chartColors.cyan}
                        strokeWidth={2}
                        dot={(dotProps) => {
                          const { cx, cy, payload, key } = dotProps;
                          const isHigh = (payload.score ?? 0) >= 0.70;
                          return (
                            <circle
                              key={key}
                              cx={cx}
                              cy={cy}
                              r={isHigh ? 4.5 : 2.5}
                              fill={isHigh ? chartColors.critical : chartColors.cyan}
                              stroke={isHigh ? '#ffffff' : 'none'}
                              strokeWidth={isHigh ? 1.5 : 0}
                            />
                          );
                        }}
                        activeDot={{ r: 5 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Contained Scrollable Table with Sticky Header */}
            <div className="table-scroll-container">
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
                      <td colSpan="7" style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                        Connecting to Live Fraud WebSocket Stream ...
                      </td>
                    </tr>
                  ) : (
                    streamData.slice(0, streamLimit).map((tx, idx) => (
                      <tr key={idx} className={tx.is_alert ? 'alert-row' : ''}>
                        <td>#{tx.block_height}</td>
                        <td>{tx.tx_hash ? tx.tx_hash.slice(0, 20) + '...' : 'N/A'}</td>
                        <td>{tx.value_btc} BTC</td>
                        <td>{tx.inputs_count} In / {tx.outputs_count} Out</td>
                        <td><strong>{tx.heuristic_score ?? tx.fraud_score}</strong></td>
                        <td style={{ fontSize: '0.8rem', color: tx.is_alert ? 'var(--alert-high)' : 'var(--text-secondary)', maxWidth: '340px' }}>
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
          </div>
        )}

        {/* Tab 3: Search */}
        {activeTab === 'search' && (
          <div className="glass-panel">
            <div className="panel-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Search size={20} color="var(--accent-gold)" />
                <h2>Transaction & Address Risk Search Engine</h2>
              </div>
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
              <div style={{ background: 'var(--alert-high-bg)', border: '1px solid var(--alert-high-border)', padding: '14px', borderRadius: '10px', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Link2 size={22} color="var(--alert-high)" />
                    <div>
                      <h4 style={{ color: 'var(--alert-high)', margin: 0, fontSize: '0.92rem', fontWeight: 700 }}>
                        Co-Spend Entity Detected ({addressClusterInfo.cluster_id})
                      </h4>
                      <p style={{ color: 'var(--text-secondary)', margin: '2px 0 0 0', fontSize: '0.82rem' }}>
                        This address belongs to a <strong>{addressClusterInfo.size}-address</strong> co-spent wallet entity.
                      </p>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    <button
                      onClick={() => navigateToGraph(addressClusterInfo.address || searchQuery)}
                      className="btn-action"
                      style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
                    >
                      <span>Investigate Graph</span>
                      <ArrowRight size={13} />
                    </button>
                    <button 
                      onClick={() => toggleClusterExpand(addressClusterInfo.cluster_id)}
                      className="btn-action"
                      style={{ color: 'var(--alert-high)', borderColor: 'var(--alert-high-border)' }}
                    >
                      {expandedClusterId === addressClusterInfo.cluster_id ? 'Hide Clustered Addresses' : `View All ${addressClusterInfo.size} Addresses`}
                    </button>
                  </div>
                </div>

                {expandedClusterId === addressClusterInfo.cluster_id && (
                  <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid var(--alert-high-border)' }}>
                    <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                      Associated Entity Addresses:
                    </span>
                    <div style={{ maxHeight: '140px', overflowY: 'auto', marginTop: '6px', background: 'var(--bg-code)', padding: '8px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                      {addressClusterInfo.addresses.map((addr, aIdx) => (
                        <div key={aIdx} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: addr === addressClusterInfo.address ? 'var(--accent-indigo)' : 'var(--text-primary)', padding: '2px 0' }}>
                          {addr} {addr === addressClusterInfo.address && <span style={{ color: 'var(--alert-high)', fontSize: '0.72rem', fontWeight: 700 }}>(Searched Address)</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {searchResult && (
              <div style={{ background: 'var(--bg-panel-nested)', padding: '18px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
                {searchResult.found ? (
                  searchResult.type === 'address' ? (
                    searchResult.risk_score_status === 'scored' ? (
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Fingerprint size={18} color="var(--accent-indigo)" />
                            <h3 style={{ color: 'var(--accent-indigo)', margin: 0, fontWeight: 700, fontSize: '0.95rem' }}>
                              Bitcoin Address Behavioral Intelligence
                            </h3>
                          </div>
                          <button
                            onClick={() => navigateToGraph(searchResult.details?.account || searchQuery)}
                            className="btn-action"
                            style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
                          >
                            <span>Investigate Graph</span>
                            <ArrowRight size={13} />
                          </button>
                        </div>

                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '12px' }}>
                          Engine: {searchResult.scoring_engine}
                        </p>
                        <div style={{ display: 'flex', gap: '20px', marginBottom: '14px', flexWrap: 'wrap' }}>
                          <div>
                            <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', fontWeight: 600 }}>Predicted Category:</span>
                            <div style={{ fontSize: '1.15rem', fontWeight: 'bold', color: 'var(--accent-gold)' }}>
                              {searchResult.predicted_category}
                            </div>
                          </div>
                          <div>
                            <span style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', fontWeight: 600 }}>Model Confidence:</span>
                            <div style={{ fontSize: '1.15rem', fontWeight: 'bold', color: 'var(--alert-low)' }}>
                              {(searchResult.model_confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                        </div>

                        {/* Explainability Block */}
                        {searchResult.explanation && (
                          <div style={{ background: 'var(--bg-card)', borderLeft: '3px solid var(--accent-indigo)', padding: '10px 14px', borderRadius: '6px', marginBottom: '14px', border: '1px solid var(--border-color)', borderLeftWidth: '3px', borderLeftColor: 'var(--accent-indigo)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                              <Sparkles size={14} color="var(--accent-indigo)" />
                              <span style={{ color: 'var(--accent-indigo)', fontSize: '0.72rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                Why Predicted (SHAP Attribution):
                              </span>
                            </div>
                            <p style={{ color: 'var(--text-primary)', fontSize: '0.85rem', margin: '4px 0 0 0', lineHeight: '1.4' }}>
                              {searchResult.explanation}
                            </p>
                          </div>
                        )}

                        {/* Factor Bar Chart for Address Search */}
                        <FactorBarChart factors={searchResult.top_factors} chartColors={chartColors} />

                        <pre className="code-box">
                          {JSON.stringify(searchResult.details, null, 2)}
                        </pre>
                      </div>
                    ) : (
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Info size={18} color="var(--accent-cyan)" />
                            <h3 style={{ color: 'var(--accent-cyan)', margin: 0, fontSize: '0.95rem' }}>
                              Individual Address ML Score: Not Scored
                            </h3>
                          </div>
                          <button
                            onClick={() => navigateToGraph(searchQuery)}
                            className="btn-action"
                            style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
                          >
                            <span>Investigate Graph</span>
                            <ArrowRight size={13} />
                          </button>
                        </div>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: '1.45', margin: 0 }}>
                          {searchResult.message}
                        </p>
                      </div>
                    )
                  ) : (
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px', flexWrap: 'wrap', gap: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          {searchResult.is_high_risk ? (
                            <AlertOctagon size={20} color="var(--alert-critical)" />
                          ) : (
                            <CheckCircle2 size={20} color="var(--alert-low)" />
                          )}
                          <h3 style={{ color: searchResult.is_high_risk ? 'var(--alert-critical)' : 'var(--alert-low)', margin: 0, fontWeight: 700, fontSize: '0.95rem' }}>
                            {searchResult.is_high_risk ? 'HIGH HEURISTIC RISK ALERT' : 'CLEAN TRANSACTION'}
                          </h3>
                        </div>
                        <button
                          onClick={() => navigateToGraph(searchResult.details?.tx_hash || searchQuery)}
                          className="btn-action"
                          style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
                        >
                          <span>Investigate Graph</span>
                          <ArrowRight size={13} />
                        </button>
                      </div>

                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem', marginBottom: '10px' }}>
                        Engine: {searchResult.scoring_engine}
                      </p>
                      <p style={{ fontFamily: 'var(--font-mono)', marginBottom: '10px', fontSize: '0.85rem' }}>
                        Heuristic Score: <strong>{searchResult.heuristic_score}</strong>
                      </p>

                      {/* Explainability Block for Transaction */}
                      {searchResult.explanation && (
                        <div style={{
                          background: searchResult.is_high_risk ? 'var(--alert-critical-bg)' : 'var(--alert-low-bg)',
                          border: `1px solid ${searchResult.is_high_risk ? 'var(--alert-critical-border)' : 'var(--alert-low-border)'}`,
                          borderLeftWidth: '4px',
                          borderLeftColor: searchResult.is_high_risk ? 'var(--alert-critical)' : 'var(--alert-low)',
                          padding: '10px 14px',
                          borderRadius: '6px',
                          marginBottom: '14px'
                        }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                            <FileSearch size={14} color={searchResult.is_high_risk ? 'var(--alert-critical)' : 'var(--alert-low)'} />
                            <span style={{ color: searchResult.is_high_risk ? 'var(--alert-critical)' : 'var(--alert-low)', fontSize: '0.72rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                              Why Flagged (Heuristic Rule Factors):
                            </span>
                          </div>
                          <p style={{ color: 'var(--text-primary)', fontSize: '0.85rem', margin: '4px 0 0 0', lineHeight: '1.4' }}>
                            {searchResult.explanation}
                          </p>
                        </div>
                      )}

                      {/* Factor Bar Chart for Transaction Search */}
                      <FactorBarChart factors={searchResult.top_factors} chartColors={chartColors} />

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

        {/* Tab 4: Graph Investigation (Cytoscape) */}
        {activeTab === 'graph' && (
          <div className="glass-panel">
            <div className="panel-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Share2 size={20} color="var(--accent-cyan)" />
                <h2>Entity Graph & Co-Spend Cluster Visualizer</h2>
              </div>
              <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                Interactive Node-Link Topology on Real Blockchain Co-Spend Entities
              </span>
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (graphInput.trim()) {
                  setGraphEntityId(graphInput.trim());
                }
              }}
              className="search-box"
              style={{ marginBottom: '12px' }}
            >
              <input
                type="text"
                className="search-input"
                placeholder="Enter Bitcoin Address (e.g. 37sczWKSEWeqJPXj9MijJa2zQhYVqoQ2AG) or Transaction Hash"
                value={graphInput}
                onChange={(e) => setGraphInput(e.target.value)}
              />
              <button type="submit" className="search-btn">Investigate Entity</button>
            </form>

            {/* Quick Sample Chips for One-Click Demos */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Quick Samples:</span>
              <button
                type="button"
                onClick={() => {
                  setGraphInput('37sczWKSEWeqJPXj9MijJa2zQhYVqoQ2AG');
                  setGraphEntityId('37sczWKSEWeqJPXj9MijJa2zQhYVqoQ2AG');
                }}
                className="btn-action"
                style={{ fontSize: '0.74rem', padding: '3px 8px' }}
              >
                CL-00001 Lead Address (Co-Spend Cluster)
              </button>
              <button
                type="button"
                onClick={() => {
                  setGraphInput('3P2uLvCJdRn9DQvJvXAdoeHeikoJSPj2sX');
                  setGraphEntityId('3P2uLvCJdRn9DQvJvXAdoeHeikoJSPj2sX');
                }}
                className="btn-action"
                style={{ fontSize: '0.74rem', padding: '3px 8px' }}
              >
                Sibling Address in Cluster
              </button>
            </div>

            <GraphView
              entityId={graphEntityId}
              apiBase={API_BASE}
              chartColors={chartColors}
              FactorBarChartComponent={FactorBarChart}
            />
          </div>
        )}

        {/* Tab 5: Network Correlation Alerts */}
        {activeTab === 'network' && (
          <div>
            {/* Section 1: Cross-Layer Cluster Summary */}
            <div className="glass-panel" style={{ marginBottom: '18px' }}>
              <div className="panel-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Network size={20} color="var(--accent-cyan)" />
                  <h2>Cross-Layer Coordinated Subnet & ASN Clusters</h2>
                </div>
                <PillSelector
                  options={[10, 25, 'all']}
                  value={subnetLimit}
                  onChange={setSubnetLimit}
                  label="Subnets:"
                />
              </div>

              {networkClusters && networkClusters.status === 'not_generated_yet' ? (
                <div style={{ background: 'var(--alert-high-bg)', borderLeft: '4px solid var(--accent-gold)', padding: '14px', borderRadius: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AlertTriangle size={18} color="var(--accent-gold)" />
                    <h3 style={{ color: 'var(--accent-gold)', margin: 0, fontSize: '0.92rem' }}>Network Correlation Not Generated Yet</h3>
                  </div>
                  <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0', fontSize: '0.85rem' }}>
                    Please run <code>python src/train_combined_risk_model.py</code> to generate <code>models/network_correlated_alerts.csv</code>.
                  </p>
                </div>
              ) : networkClusters && networkClusters.clusters ? (
                <div>
                  {/* Dynamic Top N Subnets Horizontal Bar Chart */}
                  {networkClusters.clusters.length > 0 && (
                    <div style={{ background: 'var(--bg-panel-nested)', border: '1px solid var(--border-color)', borderRadius: '10px', padding: '12px 14px', marginBottom: '14px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <TrendingUp size={15} color="var(--accent-cyan)" />
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-heading)', fontWeight: 700 }}>
                            {subnetLimit >= 99999 ? 'All' : `Top ${subnetLimit}`} Coordinated Subnets Ranked by Peak Risk
                          </span>
                        </div>
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                          Critical ≥85%, High ≥70%, Med ≥50%, Low &lt;50%
                        </span>
                      </div>
                      <div style={{ width: '100%', height: Math.min(260, Math.max(160, Math.min(subnetLimit, networkClusters.clusters.length) * 26)) }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            layout="vertical"
                            data={[...networkClusters.clusters]
                              .sort((a, b) => (b.max_fused_prob || 0) - (a.max_fused_prob || 0))
                              .slice(0, subnetLimit)
                              .map(cl => {
                                const prob = cl.max_fused_prob || 0;
                                let fill = chartColors.low;
                                if (prob >= 0.85) fill = chartColors.critical;
                                else if (prob >= 0.70) fill = chartColors.high;
                                else if (prob >= 0.50) fill = chartColors.medium;
                                return {
                                  subnet: cl.subnet,
                                  country: cl.country,
                                  asn: cl.asn,
                                  max_fused_prob: prob,
                                  avg_fused_prob: cl.avg_fused_prob || 0,
                                  tx_count: cl.tx_count,
                                  fill
                                };
                              })}
                            margin={{ top: 5, right: 25, left: 10, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" horizontal={false} />
                            <XAxis
                              type="number"
                              domain={[0, 1.0]}
                              tickFormatter={(v) => `${Math.round(v * 100)}%`}
                              tick={{ fill: chartColors.textSecondary, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                              axisLine={{ stroke: chartColors.borderColor }}
                            />
                            <YAxis
                              type="category"
                              dataKey="subnet"
                              width={100}
                              tick={{ fill: chartColors.textSecondary, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                              axisLine={{ stroke: chartColors.borderColor }}
                            />
                            <Tooltip
                              content={({ active, payload }) => {
                                if (active && payload && payload.length) {
                                  const d = payload[0].payload;
                                  return (
                                    <div style={{
                                      background: 'rgba(14, 18, 26, 0.95)',
                                      border: '1px solid rgba(255, 255, 255, 0.12)',
                                      borderRadius: '8px',
                                      padding: '8px 12px',
                                      fontSize: '0.8rem',
                                      color: '#f0f4f8'
                                    }}>
                                      <div style={{ fontWeight: 'bold', color: '#ffffff' }}>{d.subnet} ({d.country})</div>
                                      <div style={{ color: '#8b98a5', marginTop: '2px' }}>BGP ASN: <span style={{ color: '#f0f4f8' }}>{d.asn}</span></div>
                                      <div style={{ color: '#8b98a5' }}>Tx Count: <span style={{ color: '#f0f4f8' }}>{d.tx_count}</span></div>
                                      <div style={{ color: d.fill, marginTop: '2px', fontWeight: 700 }}>
                                        Peak Fused Risk: {(d.max_fused_prob * 100).toFixed(1)}% (Avg: {(d.avg_fused_prob * 100).toFixed(1)}%)
                                      </div>
                                    </div>
                                  );
                                }
                                return null;
                              }}
                            />
                            <Bar dataKey="max_fused_prob" name="Peak Fused Risk" radius={[0, 4, 4, 0]}>
                              {[...networkClusters.clusters]
                                .sort((a, b) => (b.max_fused_prob || 0) - (a.max_fused_prob || 0))
                                .slice(0, subnetLimit)
                                .map((entry, index) => {
                                  const prob = entry.max_fused_prob || 0;
                                  let fill = chartColors.low;
                                  if (prob >= 0.85) fill = chartColors.critical;
                                  else if (prob >= 0.70) fill = chartColors.high;
                                  else if (prob >= 0.50) fill = chartColors.medium;
                                  return <Cell key={`cell-${index}`} fill={fill} />;
                                })}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}

                  {/* Contained Scrollable Subnet Table */}
                  <div className="table-scroll-container" style={{ maxHeight: '280px' }}>
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
                        {networkClusters.clusters.slice(0, subnetLimit).map((cl, i) => (
                          <tr key={i}>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 'bold', color: 'var(--accent-indigo)' }}>
                              {cl.subnet}
                            </td>
                            <td>{cl.country}</td>
                            <td>
                              <span style={{ fontWeight: '600' }}>{cl.asn}</span>
                              <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginLeft: '6px' }}>({cl.asn_name})</span>
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
                </div>
              ) : (
                <p style={{ color: 'var(--text-secondary)' }}>Loading coordinated subnet clusters ...</p>
              )}
            </div>

            {/* Section 2: Top Fused Alerts */}
            <div className="glass-panel">
              <div className="panel-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <ShieldAlert size={20} color="var(--alert-critical)" />
                  <h2>Ranked Multimodal Fused Risk Alerts</h2>
                </div>
                <PillSelector
                  options={[10, 25, 'all']}
                  value={networkAlertsLimit}
                  onChange={setNetworkAlertsLimit}
                  label="Alerts:"
                />
              </div>

              {networkAlerts && networkAlerts.status === 'not_generated_yet' ? (
                <div style={{ background: 'var(--alert-high-bg)', borderLeft: '4px solid var(--accent-gold)', padding: '14px', borderRadius: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <AlertTriangle size={18} color="var(--accent-gold)" />
                    <h3 style={{ color: 'var(--accent-gold)', margin: 0, fontSize: '0.92rem' }}>Fused Alerts Not Generated Yet</h3>
                  </div>
                  <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0', fontSize: '0.85rem' }}>
                    Please run <code>python src/train_combined_risk_model.py</code> to execute multimodal fusion and generate alerts.
                  </p>
                </div>
              ) : networkAlerts && networkAlerts.alerts ? (
                <div className="table-scroll-container" style={{ maxHeight: '380px' }}>
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
                      {networkAlerts.alerts.slice(0, networkAlertsLimit).map((al) => (
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
                            <td>{al.src_asn} <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>({al.src_asn_name})</span></td>
                            <td>{al.src_country}</td>
                            <td>{(al.blockchain_risk_score * 100).toFixed(1)}%</td>
                            <td>
                              <span style={{
                                display: 'inline-flex',
                                alignItems: 'center',
                                gap: '4px',
                                background: al.is_correlated_cluster ? 'var(--alert-critical-bg)' : 'var(--bg-panel-nested)',
                                color: al.is_correlated_cluster ? 'var(--alert-critical)' : 'var(--text-secondary)',
                                border: `1px solid ${al.is_correlated_cluster ? 'var(--alert-critical-border)' : 'var(--border-color)'}`,
                                padding: '2px 7px', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 'bold'
                              }}>
                                {al.is_correlated_cluster && <Zap size={11} />}
                                {al.is_correlated_cluster ? 'CORRELATED' : 'STANDARD'}
                              </span>
                            </td>
                            <td><strong>{(al.fused_prob * 100).toFixed(1)}%</strong></td>
                            <td>
                              <span style={{ ...getTierBadgeStyle(al.risk_tier), padding: '2px 7px', borderRadius: '4px', fontSize: '0.72rem', fontWeight: 'bold', textTransform: 'uppercase' }}>
                                {al.risk_tier}
                              </span>
                            </td>
                            <td>
                              <div style={{ display: 'flex', gap: '4px' }}>
                                <button
                                  onClick={() => navigateToGraph(al.txid)}
                                  className="btn-action"
                                  style={{ display: 'inline-flex', alignItems: 'center', gap: '2px', color: 'var(--accent-cyan)' }}
                                  title="Investigate in Node Graph"
                                >
                                  <span>Graph</span>
                                  <ArrowRight size={11} />
                                </button>
                                <button
                                  onClick={() => toggleNetworkTxExpand(al.txid)}
                                  className="btn-action"
                                  style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}
                                >
                                  <span>{expandedNetworkTxid === al.txid ? 'Collapse' : 'Explain'}</span>
                                  {expandedNetworkTxid === al.txid ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                                </button>
                              </div>
                            </td>
                          </tr>

                          {/* Expanded Inline XAI Explanation & Evidence */}
                          {expandedNetworkTxid === al.txid && (
                            <tr>
                              <td colSpan="9" style={{ background: 'var(--bg-panel-nested)', padding: '14px' }}>
                                {networkTxDetail[al.txid] ? (
                                  <div>
                                    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderLeftWidth: '4px', borderLeftColor: 'var(--accent-indigo)', padding: '10px 14px', borderRadius: '6px', marginBottom: '12px' }}>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                                        <FileSearch size={14} color="var(--accent-indigo)" />
                                        <span style={{ color: 'var(--accent-indigo)', fontSize: '0.72rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                          Forensic Decision Attribution ({networkTxDetail[al.txid].scoring_engine}):
                                        </span>
                                      </div>
                                      <p style={{ color: 'var(--text-primary)', fontSize: '0.85rem', margin: '4px 0 0 0', lineHeight: '1.4' }}>
                                        {networkTxDetail[al.txid].explanation}
                                      </p>
                                    </div>

                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '14px' }}>
                                      {/* Top Factors */}
                                      <div style={{ background: 'var(--bg-card)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                                        <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                                          Top Feature Attributions:
                                        </span>
                                        <div style={{ marginTop: '6px' }}>
                                          {networkTxDetail[al.txid].top_factors?.map((fac, fIdx) => (
                                            <div key={fIdx} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid var(--border-subtle)', fontSize: '0.8rem' }}>
                                              <span style={{ color: 'var(--text-primary)' }}>{fac.label}:</span>
                                              <span style={{ color: fac.contribution > 0 ? 'var(--alert-critical)' : 'var(--alert-low)', fontWeight: 'bold', fontFamily: 'var(--font-mono)' }}>
                                                {fac.contribution > 0 ? '+' : ''}{fac.contribution.toFixed(4)} ({fac.direction})
                                              </span>
                                            </div>
                                          ))}
                                        </div>
                                      </div>

                                      {/* Network Evidence */}
                                      <div style={{ background: 'var(--bg-card)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                                        <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                                          Network Forensic Evidence:
                                        </span>
                                        <div style={{ marginTop: '6px', fontSize: '0.8rem' }}>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>Source IP:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.src_ip}</span>
                                          </div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>Subnet Peer Count:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.src_subnet24_peer_count} peers</span>
                                          </div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>6h Temporal Cluster Count:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.time_cluster_peer_count} txs</span>
                                          </div>
                                          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                                            <span style={{ color: 'var(--text-secondary)' }}>ASN Infrastructure:</span>
                                            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{networkTxDetail[al.txid].network_evidence?.src_asn} ({networkTxDetail[al.txid].network_evidence?.src_asn_name})</span>
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                ) : (
                                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>Loading forensic attribution details ...</p>
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

        {/* Tab 6: Wallet Clusters */}
        {activeTab === 'clusters' && (
          <div className="glass-panel">
            <div className="panel-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={20} color="var(--accent-cyan)" />
                <h2>Multi-Input Co-Spend Wallet Entity Clusters</h2>
              </div>
              <PillSelector
                options={[10, 25, 'all']}
                value={clustersLimit}
                onChange={setClustersLimit}
                label="Entities:"
              />
            </div>

            {walletClusters && walletClusters.status === 'not_generated_yet' ? (
              <div style={{ background: 'var(--alert-high-bg)', borderLeft: '4px solid var(--accent-gold)', padding: '14px', borderRadius: '6px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <AlertTriangle size={18} color="var(--accent-gold)" />
                  <h3 style={{ color: 'var(--accent-gold)', margin: 0, fontSize: '0.92rem' }}>Wallet Clusters Not Generated Yet</h3>
                </div>
                <p style={{ color: 'var(--text-secondary)', margin: '4px 0 0 0', fontSize: '0.85rem' }}>
                  Please run <code>python src/bitcoin_heuristics.py</code> to execute Union-Find co-spend analysis and generate <code>models/wallet_clusters.csv</code>.
                </p>
              </div>
            ) : walletClusters && walletClusters.clusters ? (
              <div className="table-scroll-container" style={{ maxHeight: '420px' }}>
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
                    {walletClusters.clusters.slice(0, clustersLimit).map((c) => (
                      <React.Fragment key={c.cluster_id}>
                        <tr>
                          <td><strong>{c.cluster_id}</strong></td>
                          <td>
                            <span style={{ background: 'var(--bg-panel-nested)', color: 'var(--accent-indigo)', padding: '2px 8px', borderRadius: '4px', fontWeight: 'bold', fontSize: '0.82rem', border: '1px solid var(--border-color)' }}>
                              {c.size} Addresses
                            </span>
                          </td>
                          <td style={{ fontFamily: 'var(--font-mono)' }}>
                            {c.addresses[0] ? `${c.addresses[0].slice(0, 24)}...` : 'N/A'}
                          </td>
                          <td>
                            <div style={{ display: 'flex', gap: '6px' }}>
                              <button
                                onClick={() => navigateToGraph(c.addresses[0])}
                                className="btn-action"
                                style={{ display: 'inline-flex', alignItems: 'center', gap: '3px', color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
                                title="Visualize Cluster Topology in Node Graph"
                              >
                                <span>Graph</span>
                                <ArrowRight size={12} />
                              </button>
                              <button
                                onClick={() => toggleClusterExpand(c.cluster_id)}
                                className="btn-action"
                                style={{ display: 'inline-flex', alignItems: 'center', gap: '3px' }}
                              >
                                <span>{expandedClusterId === c.cluster_id ? 'Collapse' : 'Expand'}</span>
                                {expandedClusterId === c.cluster_id ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                              </button>
                            </div>
                          </td>
                        </tr>
                        {expandedClusterId === c.cluster_id && (
                          <tr>
                            <td colSpan="4" style={{ background: 'var(--bg-panel-nested)', padding: '14px' }}>
                              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 600 }}>
                                All Member Addresses in {c.cluster_id} ({c.size} Total):
                              </span>
                              <div style={{ maxHeight: '160px', overflowY: 'auto', marginTop: '6px', background: 'var(--bg-code)', padding: '8px', borderRadius: '6px', border: '1px solid var(--border-color)' }}>
                                {c.addresses.map((addr, aIdx) => (
                                  <div key={aIdx} style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', color: 'var(--text-secondary)', padding: '2px 0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span>• {addr}</span>
                                    <button
                                      onClick={() => navigateToGraph(addr)}
                                      className="btn-action"
                                      style={{ padding: '1px 6px', fontSize: '0.72rem', color: 'var(--accent-cyan)' }}
                                    >
                                      Graph →
                                    </button>
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

        {/* Tab 7: Benchmarks */}
        {activeTab === 'reports' && (
          <div className="glass-panel">
            <div className="panel-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <BarChart3 size={20} color="var(--accent-gold)" />
                <h2>Model Evaluation & Benchmarks</h2>
              </div>
              <PillSelector
                options={[10, 25, 'all']}
                value={benchmarksLimit}
                onChange={setBenchmarksLimit}
                label="Filter Views:"
              />
            </div>
            {benchmarks ? (
              <div>
                {/* Grouped BarChart for Elliptic Benchmarks */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                  <Database size={16} color="var(--accent-gold)" />
                  <h3 style={{ fontSize: '0.95rem', color: 'var(--accent-gold)', margin: 0, fontWeight: 700 }}>
                    Elliptic Benchmark Evaluation
                  </h3>
                </div>

                {benchmarks.elliptic_benchmarks && benchmarks.elliptic_benchmarks.length > 0 && (
                  <div style={{ background: 'var(--bg-panel-nested)', border: '1px solid var(--border-color)', borderRadius: '10px', padding: '12px 14px', marginBottom: '16px' }}>
                    <div style={{ width: '100%', height: 260 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={benchmarks.elliptic_benchmarks.slice(0, benchmarksLimit).map(row => ({
                            name: `${row.Model} (${row.Split})`,
                            Precision: row.Precision !== null && row.Precision !== undefined ? Number(row.Precision.toFixed(4)) : 0,
                            Recall: row.Recall !== null && row.Recall !== undefined ? Number(row.Recall.toFixed(4)) : 0,
                            'F1-Score': row['F1-Score'] !== null && row['F1-Score'] !== undefined ? Number(row['F1-Score'].toFixed(4)) : 0,
                            'ROC-AUC': row['ROC-AUC'] !== null && row['ROC-AUC'] !== undefined ? Number(row['ROC-AUC'].toFixed(4)) : 0,
                          }))}
                          margin={{ top: 10, right: 25, left: 0, bottom: 25 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
                          <XAxis
                            dataKey="name"
                            tick={{ fill: chartColors.textSecondary, fontSize: 10 }}
                            axisLine={{ stroke: chartColors.borderColor }}
                            interval={0}
                            angle={-6}
                            textAnchor="end"
                          />
                          <YAxis
                            domain={[0, 1.0]}
                            tick={{ fill: chartColors.textSecondary, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                            axisLine={{ stroke: chartColors.borderColor }}
                            tickFormatter={(v) => v.toFixed(1)}
                          />
                          <Tooltip content={<DarkChartTooltip />} />
                          <Legend
                            verticalAlign="top"
                            align="right"
                            wrapperStyle={{ paddingBottom: '8px' }}
                            formatter={(value) => <span style={{ color: chartColors.textSecondary, fontSize: '0.75rem' }}>{value}</span>}
                          />
                          <Bar dataKey="Precision" fill={chartColors.cyan} radius={[3, 3, 0, 0]} />
                          <Bar dataKey="Recall" fill={chartColors.purple} radius={[3, 3, 0, 0]} />
                          <Bar dataKey="F1-Score" fill={chartColors.gold} radius={[3, 3, 0, 0]} />
                          <Bar dataKey="ROC-AUC" fill={chartColors.indigo} radius={[3, 3, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                <div className="table-scroll-container" style={{ maxHeight: '200px', marginBottom: '24px' }}>
                  <table className="stream-table">
                    <thead>
                      <tr><th>Model</th><th>Split</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>ROC-AUC</th><th>PR-AUC</th></tr>
                    </thead>
                    <tbody>
                      {benchmarks.elliptic_benchmarks.slice(0, benchmarksLimit).map((row, i) => (
                        <tr key={i}>
                          <td>{row.Model}</td><td>{row.Split}</td><td>{row.Precision?.toFixed(4)}</td>
                          <td>{row.Recall?.toFixed(4)}</td><td><strong>{row['F1-Score']?.toFixed(4)}</strong></td>
                          <td>{row['ROC-AUC']?.toFixed(4)}</td><td>{row['PR-AUC']?.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Grouped BarChart for BABD-13 Benchmarks */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                  <Sliders size={16} color="var(--accent-cyan)" />
                  <h3 style={{ fontSize: '0.95rem', color: 'var(--accent-cyan)', margin: 0, fontWeight: 700 }}>
                    BABD-13 Multi-Class Benchmark Evaluation
                  </h3>
                </div>

                {benchmarks.babd13_benchmarks && benchmarks.babd13_benchmarks.length > 0 && (
                  <div style={{ background: 'var(--bg-panel-nested)', border: '1px solid var(--border-color)', borderRadius: '10px', padding: '12px 14px', marginBottom: '16px' }}>
                    <div style={{ width: '100%', height: 260 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={benchmarks.babd13_benchmarks.slice(0, benchmarksLimit).map(row => ({
                            name: `${row.Model} (${row.Split})`,
                            Accuracy: row.Accuracy !== null && row.Accuracy !== undefined ? Number(row.Accuracy.toFixed(4)) : 0,
                            'Macro F1': row['Macro F1'] !== null && row['Macro F1'] !== undefined ? Number(row['Macro F1'].toFixed(4)) : 0,
                            'Weighted F1': row['Weighted F1'] !== null && row['Weighted F1'] !== undefined ? Number(row['Weighted F1'].toFixed(4)) : 0,
                          }))}
                          margin={{ top: 10, right: 25, left: 0, bottom: 25 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" vertical={false} />
                          <XAxis
                            dataKey="name"
                            tick={{ fill: chartColors.textSecondary, fontSize: 10 }}
                            axisLine={{ stroke: chartColors.borderColor }}
                            interval={0}
                            angle={-6}
                            textAnchor="end"
                          />
                          <YAxis
                            domain={[0, 1.0]}
                            tick={{ fill: chartColors.textSecondary, fontSize: 10, fontFamily: 'var(--font-mono)' }}
                            axisLine={{ stroke: chartColors.borderColor }}
                            tickFormatter={(v) => v.toFixed(1)}
                          />
                          <Tooltip content={<DarkChartTooltip />} />
                          <Legend
                            verticalAlign="top"
                            align="right"
                            wrapperStyle={{ paddingBottom: '8px' }}
                            formatter={(value) => <span style={{ color: chartColors.textSecondary, fontSize: '0.75rem' }}>{value}</span>}
                          />
                          <Bar dataKey="Accuracy" fill={chartColors.cyan} radius={[3, 3, 0, 0]} />
                          <Bar dataKey="Macro F1" fill={chartColors.gold} radius={[3, 3, 0, 0]} />
                          <Bar dataKey="Weighted F1" fill={chartColors.indigo} radius={[3, 3, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                <div className="table-scroll-container" style={{ maxHeight: '200px' }}>
                  <table className="stream-table">
                    <thead>
                      <tr><th>Model</th><th>Split</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th><th>Macro Precision</th><th>Macro Recall</th></tr>
                    </thead>
                    <tbody>
                      {benchmarks.babd13_benchmarks.slice(0, benchmarksLimit).map((row, i) => (
                        <tr key={i}>
                          <td>{row.Model}</td><td>{row.Split}</td><td>{row.Accuracy?.toFixed(4)}</td>
                          <td><strong>{row['Macro F1']?.toFixed(4)}</strong></td><td>{row['Weighted F1']?.toFixed(4)}</td>
                          <td>{row['Macro Precision']?.toFixed(4)}</td><td>{row['Macro Recall']?.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
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
