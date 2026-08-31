import React, { useEffect, useRef, useState, useCallback } from 'react';
import cytoscape from 'cytoscape';
import {
  Share2,
  Info,
  AlertTriangle,
  Fingerprint,
  FileSearch,
  Sparkles,
  Layers,
  CheckCircle2,
  AlertOctagon,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  FileCode,
  RotateCcw,
  Network,
  Activity,
  Globe,
  Radio
} from 'lucide-react';

export default function GraphView({
  entityId,
  apiBase,
  chartColors,
  FactorBarChartComponent,
  onSelectEntity
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  const [currentEntity, setCurrentEntity] = useState(entityId || '37sczWKSEWeqJPXj9MijJa2zQhYVqoQ2AG');
  const [hops, setHops] = useState(2);
  const [includeIp, setIncludeIp] = useState(true);
  const [activeLayout, setActiveLayout] = useState('cose');
  const [loading, setLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  // Sync with prop if changed from outside
  useEffect(() => {
    if (entityId && entityId.trim() && entityId !== currentEntity) {
      setCurrentEntity(entityId.trim());
    }
  }, [entityId]);

  // Helper to determine node color based on risk score
  const getNodeColor = (riskScore, type) => {
    if (type === 'ip') return '#a855f7'; // Purple for IP Nodes
    if (type === 'tx') return '#00d2ff'; // Cyan for Transaction Nodes
    if (riskScore === null || riskScore === undefined) return '#64748b'; // Slate gray for unscored
    if (riskScore >= 0.80) return '#ff4d4d'; // Critical (>=80)
    if (riskScore >= 0.60) return '#fb923c'; // High Risk (60-79)
    return '#00e676'; // Low Risk (<60)
  };

  // Fetch graph data from backend
  const fetchGraph = useCallback((targetId, currentHops, showIp) => {
    if (!targetId || !targetId.trim()) return;
    setLoading(true);

    const url = `${apiBase}/api/graph/${encodeURIComponent(targetId.trim())}?hops=${currentHops}&include_ip=${showIp}`;
    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        setGraphData(data);
        if (data.found && data.nodes && data.nodes.length > 0) {
          const primaryNode = data.nodes.find((n) => n.is_queried || n.id === targetId) || data.nodes[0];
          setSelectedNode(primaryNode);
        }
      })
      .catch((err) => {
        console.error('Graph API load error:', err);
        setGraphData({ found: false, message: 'Failed to load graph intelligence.' });
      })
      .finally(() => {
        setLoading(false);
      });
  }, [apiBase]);

  useEffect(() => {
    fetchGraph(currentEntity, hops, includeIp);
  }, [currentEntity, hops, includeIp, fetchGraph]);

  // Apply Cytoscape Layout
  const runLayout = useCallback((layoutName) => {
    if (!cyRef.current) return;
    const cy = cyRef.current;

    let layoutOptions = { name: layoutName, animate: true, animationDuration: 450, padding: 40 };
    if (layoutName === 'cose') {
      layoutOptions = {
        ...layoutOptions,
        nodeRepulsion: 7000,
        idealEdgeLength: 120,
        gravity: 0.25,
        numIter: 1000
      };
    } else if (layoutName === 'concentric') {
      layoutOptions = {
        ...layoutOptions,
        concentric: (node) => (node.data('is_queried') ? 3 : node.data('type') === 'tx' ? 2 : 1),
        levelWidth: () => 1
      };
    } else if (layoutName === 'breadthfirst') {
      layoutOptions = {
        ...layoutOptions,
        directed: true,
        spacingFactor: 1.2
      };
    }

    const l = cy.layout(layoutOptions);
    l.run();
  }, []);

  // Cytoscape initialization and updates
  useEffect(() => {
    if (!containerRef.current || !graphData || !graphData.found || !graphData.nodes || loading) {
      return;
    }

    const elements = [
      ...graphData.nodes.map((n) => ({
        group: 'nodes',
        data: {
          id: n.id,
          label: n.label || n.id,
          type: n.type,
          risk_score: n.risk_score,
          category: n.category,
          is_queried: n.is_queried,
          color: getNodeColor(n.risk_score, n.type),
          rawNode: n
        }
      })),
      ...(graphData.edges || []).map((e, idx) => ({
        group: 'edges',
        data: {
          id: `e-${idx}-${e.source}-${e.target}`,
          source: e.source,
          target: e.target,
          label: e.label || 'LINK'
        }
      }))
    ];

    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      style: [
        // 1. Wallets / Addresses (Circles)
        {
          selector: 'node[type="wallet"], node[type="address"]',
          style: {
            'shape': 'ellipse',
            'background-color': 'data(color)',
            'background-opacity': 0.85,
            'label': 'data(label)',
            'color': '#f0f4f8',
            'font-size': '11px',
            'font-family': 'var(--font-mono, monospace)',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'text-background-color': 'rgba(10, 13, 20, 0.9)',
            'text-background-opacity': 0.85,
            'text-background-padding': '3px 6px',
            'text-background-shape': 'roundrectangle',
            'width': (ele) => (ele.data('is_queried') ? 38 : 30),
            'height': (ele) => (ele.data('is_queried') ? 38 : 30),
            'border-width': (ele) => (ele.data('is_queried') ? 3.5 : 1.5),
            'border-color': (ele) => (ele.data('is_queried') ? '#ffffff' : 'rgba(255, 255, 255, 0.3)'),
            'transition-property': 'background-color, width, height, border-width',
            'transition-duration': '0.2s'
          }
        },
        // 2. Transactions (Cyan Rounded Squares)
        {
          selector: 'node[type="tx"]',
          style: {
            'shape': 'round-rectangle',
            'background-color': '#00d2ff',
            'background-opacity': 0.85,
            'label': 'data(label)',
            'color': '#ffffff',
            'font-size': '10px',
            'font-family': 'var(--font-mono, monospace)',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'text-background-color': 'rgba(0, 30, 50, 0.9)',
            'text-background-opacity': 0.85,
            'text-background-padding': '3px 6px',
            'text-background-shape': 'roundrectangle',
            'width': (ele) => (ele.data('is_queried') ? 34 : 26),
            'height': (ele) => (ele.data('is_queried') ? 34 : 26),
            'border-width': 2,
            'border-color': '#ffffff'
          }
        },
        // 3. Network IP Nodes (Purple Diamonds)
        {
          selector: 'node[type="ip"]',
          style: {
            'shape': 'diamond',
            'background-color': '#a855f7',
            'background-opacity': 0.9,
            'label': 'data(label)',
            'color': '#f3e8ff',
            'font-size': '10px',
            'font-family': 'var(--font-mono, monospace)',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'text-background-color': 'rgba(30, 10, 50, 0.9)',
            'text-background-opacity': 0.85,
            'text-background-padding': '3px 6px',
            'text-background-shape': 'roundrectangle',
            'width': 30,
            'height': 30,
            'border-width': 2,
            'border-color': '#e9d5ff'
          }
        },
        // 4. Selected Node Glow
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': '#f7931a',
            'shadow-blur': 16,
            'shadow-color': '#f7931a',
            'shadow-opacity': 0.9
          }
        },
        // 5. Standard On-Chain Flow Edges (FROM / TO)
        {
          selector: 'edge[label="FROM"], edge[label="TO"], edge[label="co-spend"]',
          style: {
            'width': 2,
            'line-color': 'rgba(255, 255, 255, 0.25)',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'rgba(255, 255, 255, 0.4)',
            'arrow-scale': 0.9,
            'label': 'data(label)',
            'font-size': '8px',
            'color': '#8b98a5',
            'text-background-color': 'rgba(10, 13, 20, 0.8)',
            'text-background-opacity': 0.7,
            'text-background-padding': '2px 4px',
            'text-background-shape': 'roundrectangle'
          }
        },
        // 6. Network Observed Broadcast Edges (Dashed Purple)
        {
          selector: 'edge[label="OBSERVED"]',
          style: {
            'width': 2,
            'line-color': '#a855f7',
            'line-style': 'dashed',
            'line-dash-pattern': [6, 3],
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#c084fc',
            'arrow-scale': 0.9,
            'label': 'OBSERVED',
            'font-size': '8px',
            'color': '#e9d5ff',
            'text-background-color': 'rgba(30, 10, 50, 0.85)',
            'text-background-opacity': 0.8,
            'text-background-padding': '2px 4px',
            'text-background-shape': 'roundrectangle'
          }
        }
      ]
    });

    // Single Click: Select and Inspect Node
    cy.on('tap', 'node', (evt) => {
      const nodeData = evt.target.data('rawNode');
      setSelectedNode(nodeData);
    });

    // Double Click: Set Focal Entity and Expand Multi-Hop
    cy.on('dbltap', 'node', (evt) => {
      const nodeData = evt.target.data('rawNode');
      if (nodeData && nodeData.id) {
        setCurrentEntity(nodeData.id);
        if (onSelectEntity) onSelectEntity(nodeData.id);
      }
    });

    cyRef.current = cy;
    runLayout(activeLayout);

    const timer = setTimeout(() => {
      if (cyRef.current) {
        cyRef.current.resize();
        cyRef.current.fit(undefined, 40);
      }
    }, 120);

    return () => {
      clearTimeout(timer);
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [graphData, loading, activeLayout, runLayout, onSelectEntity]);

  const handleLayoutChange = (layoutName) => {
    setActiveLayout(layoutName);
    runLayout(layoutName);
  };

  const handleZoomIn = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current && cyRef.current.fit(undefined, 35);

  const handleExportPng = () => {
    if (!cyRef.current) return;
    const png64 = cyRef.current.png({ full: true, scale: 2, bg: '#0a0d14' });
    const link = document.createElement('a');
    link.download = `bitsentinel-graph-${currentEntity.slice(0, 10)}.png`;
    link.href = png64;
    link.click();
  };

  const handleExportJson = () => {
    if (!graphData) return;
    const jsonStr = JSON.stringify(graphData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const link = document.createElement('a');
    link.download = `bitsentinel-graph-${currentEntity.slice(0, 10)}.json`;
    link.href = URL.createObjectURL(blob);
    link.click();
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Interactive Graph Top Control Bar */}
      <div style={{
        background: 'var(--bg-panel-nested)',
        border: '1px solid var(--border-color)',
        borderRadius: '10px',
        padding: '12px 16px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        {/* Left: Title & Subtitle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            background: 'rgba(0, 210, 255, 0.1)',
            border: '1px solid rgba(0, 210, 255, 0.25)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Share2 size={18} color="var(--accent-cyan)" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ fontSize: '0.98rem', fontWeight: 700, margin: 0, color: 'var(--text-heading)' }}>
                Interactive Cross-Layer Investigation Graph
              </h3>
            </div>
            <p style={{ margin: '2px 0 0 0', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              Double-click any node to expand multi-hop connections. Click to inspect metadata.
            </p>
          </div>
        </div>

        {/* Right: Interactive Toolbar Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          {/* Focal Entity Tag */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            background: 'var(--bg-card)',
            border: '1px solid var(--border-color)',
            padding: '4px 10px',
            borderRadius: '6px',
            fontSize: '0.78rem'
          }}>
            <span style={{ color: 'var(--text-muted)' }}>Focal:</span>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)', fontWeight: 700 }}>
              {currentEntity.length > 18 ? `${currentEntity.slice(0, 10)}...${currentEntity.slice(-6)}` : currentEntity}
            </span>
            <button
              onClick={() => fetchGraph(currentEntity, hops, includeIp)}
              style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: '2px', display: 'flex' }}
              title="Refresh Graph"
            >
              <RotateCcw size={12} />
            </button>
          </div>

          {/* Hops Selector (1 | 2 | 3) */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.78rem' }}>
            <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>Hops:</span>
            <div className="filter-pills">
              {[1, 2, 3].map((h) => (
                <button
                  key={h}
                  className={`pill-btn ${hops === h ? 'active' : ''}`}
                  onClick={() => setHops(h)}
                  type="button"
                >
                  {h}
                </button>
              ))}
            </div>
          </div>

          {/* IP Layer Toggle */}
          <button
            onClick={() => setIncludeIp(!includeIp)}
            className="btn-action"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              background: includeIp ? 'rgba(168, 85, 247, 0.15)' : 'transparent',
              color: includeIp ? '#c084fc' : 'var(--text-muted)',
              borderColor: includeIp ? '#a855f7' : 'var(--border-color)',
              fontWeight: 700,
              fontSize: '0.76rem',
              padding: '4px 10px'
            }}
          >
            <Globe size={13} />
            <span>IP Layer: {includeIp ? 'ON' : 'OFF'}</span>
          </button>

          {/* Layout Selector */}
          <div className="filter-pills">
            {['cose', 'concentric', 'breadthfirst', 'circle'].map((layoutName) => (
              <button
                key={layoutName}
                className={`pill-btn ${activeLayout === layoutName ? 'active' : ''}`}
                onClick={() => handleLayoutChange(layoutName)}
                type="button"
                style={{ textTransform: 'capitalize' }}
              >
                {layoutName}
              </button>
            ))}
          </div>

          {/* Export Buttons */}
          <div style={{ display: 'flex', gap: '4px' }}>
            <button onClick={handleExportPng} className="btn-action" style={{ padding: '4px 8px', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }} title="Export Graph as PNG Image">
              <Download size={13} />
              <span>Export PNG</span>
            </button>
            <button onClick={handleExportJson} className="btn-action" style={{ padding: '4px 8px', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }} title="Export Graph JSON Data">
              <FileCode size={13} />
              <span>Export JSON</span>
            </button>
          </div>
        </div>
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
          Computing multi-hop cross-layer topology intelligence ...
        </div>
      )}

      {!loading && graphData && !graphData.found && (
        <div style={{ background: 'var(--bg-panel-nested)', padding: '20px', borderRadius: '10px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
          <AlertTriangle size={24} color="var(--accent-gold)" style={{ margin: '0 auto 8px auto' }} />
          <h4 style={{ color: 'var(--text-heading)', margin: '0 0 4px 0' }}>Graph Investigation Entity Not Found</h4>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>
            {graphData.message || `No graph data available for '${currentEntity}'.`}
          </p>
        </div>
      )}

      {!loading && graphData && graphData.found && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '16px', minHeight: '520px' }}>
          {/* Main Cytoscape Canvas Container */}
          <div style={{ background: 'var(--bg-panel-nested)', border: '1px solid var(--border-color)', borderRadius: '10px', position: 'relative', overflow: 'hidden', minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
            
            {/* Top Canvas Action Overlay Toolbar */}
            <div style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 10, display: 'flex', gap: '4px', background: 'rgba(10, 13, 20, 0.85)', padding: '4px', borderRadius: '6px', border: '1px solid var(--border-color)', backdropFilter: 'blur(6px)' }}>
              <button onClick={handleZoomIn} className="btn-action" style={{ padding: '4px 8px' }} title="Zoom In">
                <ZoomIn size={14} />
              </button>
              <button onClick={handleZoomOut} className="btn-action" style={{ padding: '4px 8px' }} title="Zoom Out">
                <ZoomOut size={14} />
              </button>
              <button onClick={handleFit} className="btn-action" style={{ padding: '4px 8px' }} title="Fit View">
                <Maximize2 size={14} />
              </button>
            </div>

            {/* On-Canvas Visual Legend (Top Left Overlay) */}
            <div style={{
              position: 'absolute',
              top: '12px',
              left: '12px',
              zIndex: 10,
              background: 'rgba(10, 13, 20, 0.88)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              padding: '10px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
              fontSize: '0.74rem',
              backdropFilter: 'blur(8px)',
              pointerEvents: 'none'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ff4d4d' }}></span>
                <span style={{ color: 'var(--text-secondary)' }}>Critical Wallet (≥80)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#fb923c' }}></span>
                <span style={{ color: 'var(--text-secondary)' }}>High Risk Wallet (60-79)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#00e676' }}></span>
                <span style={{ color: 'var(--text-secondary)' }}>Low Risk Wallet</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ width: '10px', height: '10px', borderRadius: '2px', background: '#00d2ff' }}></span>
                <span style={{ color: 'var(--text-secondary)' }}>Transaction (TXID)</span>
              </div>
              {includeIp && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ width: '10px', height: '10px', transform: 'rotate(45deg)', background: '#a855f7' }}></span>
                  <span style={{ color: '#e9d5ff', fontWeight: 600 }}>Network IP Node</span>
                </div>
              )}
            </div>

            {/* Cytoscape Canvas DOM element */}
            <div ref={containerRef} style={{ flex: 1, width: '100%', minHeight: '460px', background: 'transparent' }} />

            {/* Bottom Telemetry & Engine Bar */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '6px 16px',
              borderTop: '1px solid var(--border-color)',
              background: 'rgba(10, 13, 20, 0.85)',
              fontSize: '0.72rem',
              color: 'var(--text-muted)'
            }}>
              <div>
                <span>Engine: </span>
                <strong style={{ color: 'var(--text-primary)' }}>FastAPI + NetworkX & GraphSAGE GNN</strong>
                <span style={{ margin: '0 8px' }}>|</span>
                <span>ML Model: </span>
                <strong style={{ color: 'var(--accent-gold)' }}>BABD-13 Random Forest + Self-Supervised Embeddings</strong>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--alert-low)' }}></span>
                <span style={{ color: 'var(--alert-low)', fontWeight: 600 }}>Cross-Layer Active</span>
              </div>
            </div>
          </div>

          {/* Right Inspection & Forensic Attribution Sidebar */}
          <div style={{
            background: 'var(--bg-panel-nested)',
            border: '1px solid var(--border-color)',
            borderRadius: '10px',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
            overflowY: 'auto',
            maxHeight: '560px'
          }}>
            <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 700 }}>
                Selected Entity Node
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--accent-indigo)', wordBreak: 'break-all', fontWeight: 700, marginTop: '4px' }}>
                {selectedNode ? selectedNode.id : 'Click a node to inspect'}
              </div>
            </div>

            {selectedNode ? (
              <div>
                {/* Node Metadata Badges */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
                  {selectedNode.type && (
                    <span style={{
                      background: selectedNode.type === 'ip' ? 'rgba(168, 85, 247, 0.15)' : selectedNode.type === 'tx' ? 'rgba(0, 210, 255, 0.15)' : 'var(--bg-card)',
                      color: selectedNode.type === 'ip' ? '#c084fc' : selectedNode.type === 'tx' ? '#00d2ff' : 'var(--text-secondary)',
                      border: `1px solid ${selectedNode.type === 'ip' ? '#a855f7' : selectedNode.type === 'tx' ? '#00d2ff' : 'var(--border-color)'}`,
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontSize: '0.74rem',
                      textTransform: 'uppercase',
                      fontWeight: 700
                    }}>
                      Type: {selectedNode.type}
                    </span>
                  )}
                  {selectedNode.category && (
                    <span style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.74rem', color: 'var(--accent-gold)', fontWeight: 700 }}>
                      {selectedNode.category}
                    </span>
                  )}
                  {selectedNode.risk_score !== null && selectedNode.risk_score !== undefined && (
                    <span style={{
                      background: selectedNode.risk_score >= 0.70 ? 'var(--alert-critical-bg)' : 'var(--alert-low-bg)',
                      color: selectedNode.risk_score >= 0.70 ? 'var(--alert-critical)' : 'var(--alert-low)',
                      border: `1px solid ${selectedNode.risk_score >= 0.70 ? 'var(--alert-critical-border)' : 'var(--alert-low-border)'}`,
                      padding: '2px 8px',
                      borderRadius: '4px',
                      fontSize: '0.74rem',
                      fontWeight: 700
                    }}>
                      Risk: {(selectedNode.risk_score * (selectedNode.risk_score <= 1.0 ? 100 : 1)).toFixed(1)}%
                    </span>
                  )}
                </div>

                {/* Network IP Specific Metadata */}
                {selectedNode.type === 'ip' && (
                  <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '10px 12px', marginBottom: '12px', fontSize: '0.8rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Subnet /24:</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{selectedNode.subnet}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                      <span style={{ color: 'var(--text-muted)' }}>BGP ASN:</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 600 }}>{selectedNode.asn}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Organization:</span>
                      <span style={{ color: 'var(--accent-indigo)', fontWeight: 600 }}>{selectedNode.asn_name}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '3px 0' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Country:</span>
                      <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{selectedNode.country}</span>
                    </div>
                  </div>
                )}

                {/* Node Forensic Explanation Box */}
                {selectedNode.explanation && (
                  <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderLeftWidth: '3px', borderLeftColor: selectedNode.type === 'ip' ? '#a855f7' : 'var(--accent-indigo)', padding: '10px 12px', borderRadius: '6px', marginBottom: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                      <Sparkles size={14} color={selectedNode.type === 'ip' ? '#c084fc' : 'var(--accent-indigo)'} />
                      <span style={{ color: selectedNode.type === 'ip' ? '#c084fc' : 'var(--accent-indigo)', fontSize: '0.72rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        Forensic Explainability:
                      </span>
                    </div>
                    <p style={{ color: 'var(--text-primary)', fontSize: '0.82rem', margin: 0, lineHeight: '1.4' }}>
                      {selectedNode.explanation}
                    </p>
                  </div>
                )}

                {/* Feature Attribution Bar Chart */}
                {selectedNode.top_factors && selectedNode.top_factors.length > 0 && FactorBarChartComponent && (
                  <FactorBarChartComponent factors={selectedNode.top_factors} chartColors={chartColors} />
                )}

                {/* Action: Set as Focal */}
                <button
                  onClick={() => {
                    setCurrentEntity(selectedNode.id);
                    if (onSelectEntity) onSelectEntity(selectedNode.id);
                  }}
                  className="btn-action"
                  style={{ width: '100%', marginTop: '8px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '6px', color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
                >
                  <Share2 size={13} />
                  <span>Set as Focal & Expand Graph</span>
                </button>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)' }}>
                <Activity size={24} style={{ margin: '0 auto 8px auto', opacity: 0.5 }} />
                <p style={{ fontSize: '0.82rem', margin: 0 }}>
                  Select any node on canvas to inspect metadata, or double-click to expand connections.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
