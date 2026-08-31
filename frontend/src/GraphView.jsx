import React, { useEffect, useRef, useState } from 'react';
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
  Maximize2
} from 'lucide-react';

export default function GraphView({
  entityId,
  apiBase,
  chartColors,
  FactorBarChartComponent
}) {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  const [loading, setLoading] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  // Helper to determine node color based on risk score
  const getNodeColor = (riskScore) => {
    if (riskScore === null || riskScore === undefined) {
      return '#64748b'; // Slate gray for unscored nodes
    }
    if (riskScore >= 0.85) return chartColors.critical;
    if (riskScore >= 0.70) return chartColors.high;
    if (riskScore >= 0.50) return chartColors.medium;
    return chartColors.low;
  };

  useEffect(() => {
    if (!entityId || !entityId.trim()) {
      setGraphData(null);
      setSelectedNode(null);
      return;
    }

    setLoading(true);
    setSelectedNode(null);

    fetch(`${apiBase}/api/graph/${encodeURIComponent(entityId.trim())}`)
      .then((res) => res.json())
      .then((data) => {
        setGraphData(data);
        if (data.found && data.nodes && data.nodes.length > 0) {
          const primaryNode = data.nodes.find((n) => n.is_queried) || data.nodes[0];
          setSelectedNode(primaryNode);
        }
      })
      .catch((err) => {
        console.error('Graph API load error:', err);
        setGraphData({ found: false, message: 'Failed to load graph data from backend.' });
      })
      .finally(() => {
        setLoading(false);
      });
  }, [entityId, apiBase]);

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
          color: getNodeColor(n.risk_score),
          rawNode: n
        }
      })),
      ...(graphData.edges || []).map((e, idx) => ({
        group: 'edges',
        data: {
          id: `e-${idx}`,
          source: e.source,
          target: e.target,
          label: e.label || 'co-spend'
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
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'color': '#f0f4f8',
            'font-size': '11px',
            'font-family': 'var(--font-mono, monospace)',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'text-background-color': 'rgba(10, 13, 20, 0.85)',
            'text-background-opacity': 0.8,
            'text-background-padding': '3px 5px',
            'text-background-shape': 'roundrectangle',
            'width': (ele) => (ele.data('is_queried') ? 34 : 26),
            'height': (ele) => (ele.data('is_queried') ? 34 : 26),
            'border-width': (ele) => (ele.data('is_queried') ? 3 : 1),
            'border-color': (ele) => (ele.data('is_queried') ? '#ffffff' : 'rgba(255, 255, 255, 0.2)'),
            'transition-property': 'background-color, width, height, border-width',
            'transition-duration': '0.2s'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 4,
            'border-color': chartColors.gold,
            'shadow-blur': 12,
            'shadow-color': chartColors.gold,
            'shadow-opacity': 0.8
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': 'rgba(255, 255, 255, 0.15)',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': 'rgba(255, 255, 255, 0.25)',
            'arrow-scale': 0.8,
            'line-style': 'dashed'
          }
        }
      ],
      layout: {
        name: 'cose',
        animate: true,
        animationDuration: 400,
        nodeRepulsion: 5000,
        idealEdgeLength: 110,
        gravity: 0.3,
        padding: 35
      }
    });

    // Node click handler
    cy.on('tap', 'node', (evt) => {
      const nodeData = evt.target.data('rawNode');
      setSelectedNode(nodeData);
    });

    cyRef.current = cy;

    // Trigger resize & fit after container is ready
    const timer = setTimeout(() => {
      if (cyRef.current) {
        cyRef.current.resize();
        cyRef.current.fit(undefined, 35);
      }
    }, 100);

    return () => {
      clearTimeout(timer);
      if (cyRef.current) {
        cyRef.current.destroy();
        cyRef.current = null;
      }
    };
  }, [graphData, chartColors, loading]);

  const handleZoomIn = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 1.25);
  const handleZoomOut = () => cyRef.current && cyRef.current.zoom(cyRef.current.zoom() * 0.8);
  const handleFit = () => cyRef.current && cyRef.current.fit(undefined, 30);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
      {/* Informative Edge Linkage Caption */}
      {graphData && graphData.edges_note && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-panel-nested)', borderLeft: '3px solid var(--accent-gold)', padding: '10px 14px', borderRadius: '6px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
          <Info size={16} color="var(--accent-gold)" style={{ flexShrink: 0 }} />
          <span>{graphData.edges_note}</span>
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', padding: '36px', color: 'var(--text-muted)' }}>
          Loading network graph intelligence ...
        </div>
      )}

      {!loading && graphData && !graphData.found && (
        <div style={{ background: 'var(--bg-panel-nested)', padding: '20px', borderRadius: '10px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
          <AlertTriangle size={24} color="var(--accent-gold)" style={{ margin: '0 auto 8px auto' }} />
          <h4 style={{ color: 'var(--text-heading)', margin: '0 0 4px 0' }}>Graph Investigation Entity Not Found</h4>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0 }}>
            {graphData.message || `No graph data available for '${entityId}'.`}
          </p>
        </div>
      )}

      {!loading && graphData && graphData.found && (
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) 340px', gap: '16px', minHeight: '440px' }}>
          {/* Main Cytoscape Canvas Container */}
          <div style={{ background: 'var(--bg-panel-nested)', border: '1px solid var(--border-color)', borderRadius: '10px', position: 'relative', overflow: 'hidden', minHeight: '420px', display: 'flex', flexDirection: 'column' }}>
            {/* Canvas Header Toolbar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 14px', borderBottom: '1px solid var(--border-color)', background: 'rgba(10, 13, 20, 0.6)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Share2 size={16} color="var(--accent-cyan)" />
                <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-heading)' }}>
                  {graphData.entity_type === 'address' ? 'Co-Spend Wallet Cluster Topology' : 'Transaction Risk Profile'}
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  ({graphData.nodes?.length || 0} node{graphData.nodes?.length === 1 ? '' : 's'})
                </span>
              </div>

              {/* Zoom & Fit Controls */}
              <div style={{ display: 'flex', gap: '4px' }}>
                <button onClick={handleZoomIn} className="btn-action" style={{ padding: '3px 7px' }} title="Zoom In">
                  <ZoomIn size={14} />
                </button>
                <button onClick={handleZoomOut} className="btn-action" style={{ padding: '3px 7px' }} title="Zoom Out">
                  <ZoomOut size={14} />
                </button>
                <button onClick={handleFit} className="btn-action" style={{ padding: '3px 7px' }} title="Fit View">
                  <Maximize2 size={14} />
                </button>
              </div>
            </div>

            {/* Canvas viewport */}
            <div ref={containerRef} style={{ flex: 1, width: '100%', minHeight: '380px', background: 'transparent' }} />

            {/* Risk Legend Footer */}
            <div style={{ display: 'flex', gap: '14px', padding: '6px 14px', borderTop: '1px solid var(--border-color)', background: 'rgba(10, 13, 20, 0.7)', fontSize: '0.72rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: chartColors.critical }}></span>
                <span style={{ color: 'var(--text-muted)' }}>Critical (≥85%)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: chartColors.high }}></span>
                <span style={{ color: 'var(--text-muted)' }}>High (≥70%)</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: chartColors.low }}></span>
                <span style={{ color: 'var(--text-muted)' }}>Normal / Safe</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#64748b' }}></span>
                <span style={{ color: 'var(--text-muted)' }}>Unscored Sibling</span>
              </div>
            </div>
          </div>

          {/* Selected Node Details & Factor Attribution Sidebar */}
          <div style={{ background: 'var(--bg-panel-nested)', border: '1px solid var(--border-color)', borderRadius: '10px', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', overflowY: 'auto', maxHeight: '520px' }}>
            <div style={{ borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
              <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', fontWeight: 700 }}>
                Selected Entity Node
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: 'var(--accent-indigo)', wordBreak: 'break-all', fontWeight: 700, marginTop: '4px' }}>
                {selectedNode ? selectedNode.id : 'Click a node to inspect'}
              </div>
            </div>

            {selectedNode ? (
              <div>
                {/* Node Metadata Badges */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '12px' }}>
                  {selectedNode.type && (
                    <span style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.74rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 600 }}>
                      Type: {selectedNode.type}
                    </span>
                  )}
                  {selectedNode.category && (
                    <span style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '2px 8px', borderRadius: '4px', fontSize: '0.74rem', color: 'var(--accent-gold)', fontWeight: 700 }}>
                      {selectedNode.category}
                    </span>
                  )}
                  {selectedNode.risk_score !== null && selectedNode.risk_score !== undefined && (
                    <span style={{ background: selectedNode.risk_score >= 0.70 ? 'var(--alert-critical-bg)' : 'var(--alert-low-bg)', color: selectedNode.risk_score >= 0.70 ? 'var(--alert-critical)' : 'var(--alert-low)', border: `1px solid ${selectedNode.risk_score >= 0.70 ? 'var(--alert-critical-border)' : 'var(--alert-low-border)'}`, padding: '2px 8px', borderRadius: '4px', fontSize: '0.74rem', fontWeight: 700 }}>
                      Score: {(selectedNode.risk_score * (selectedNode.risk_score <= 1.0 ? 100 : 1)).toFixed(1)}%
                    </span>
                  )}
                </div>

                {/* Node Explanation Box */}
                {selectedNode.explanation && (
                  <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)', borderLeftWidth: '3px', borderLeftColor: 'var(--accent-indigo)', padding: '10px 12px', borderRadius: '6px', marginBottom: '12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                      <Sparkles size={14} color="var(--accent-indigo)" />
                      <span style={{ color: 'var(--accent-indigo)', fontSize: '0.72rem', fontWeight: 'bold', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        Forensic Explainability:
                      </span>
                    </div>
                    <p style={{ color: 'var(--text-primary)', fontSize: '0.82rem', margin: 0, lineHeight: '1.4' }}>
                      {selectedNode.explanation}
                    </p>
                  </div>
                )}

                {/* Node Feature Attribution Factor Bar Chart */}
                {selectedNode.top_factors && selectedNode.top_factors.length > 0 && FactorBarChartComponent && (
                  <FactorBarChartComponent factors={selectedNode.top_factors} chartColors={chartColors} />
                )}
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
                Select any node in the topology canvas to inspect forensic factors and co-spend attributes.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
