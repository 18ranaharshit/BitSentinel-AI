import React, { useEffect, useRef } from 'react';

/**
 * NetworkBackground
 * 
 * Lightweight 2D Canvas animated background representing a Bitcoin transaction/node graph.
 * Performance guardrails:
 *   1. Pauses animation when document is hidden (Page Visibility API).
 *   2. Respects prefers-reduced-motion (renders a single static frame).
 *   3. Capped at 40 nodes with strict distance thresholds to maintain 60fps budget.
 *   4. Zero layout impact: pointer-events: none, z-index: 0.
 *   5. Theme-adaptive coloring (Light Mode vs Dark Mode).
 */
export default function NetworkBackground({ theme = 'light' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d', { alpha: true });
    if (!ctx) return;

    let animationFrameId = null;
    let isRunning = true;
    let width = 0;
    let height = 0;

    // Check prefers-reduced-motion
    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    let prefersReducedMotion = reducedMotionQuery.matches;

    const handleMotionPreferenceChange = (e) => {
      prefersReducedMotion = e.matches;
      if (prefersReducedMotion && animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
        drawFrame(0); // Render single static frame
      } else if (!prefersReducedMotion && !animationFrameId && !document.hidden) {
        lastTime = performance.now();
        loop(lastTime);
      }
    };

    if (reducedMotionQuery.addEventListener) {
      reducedMotionQuery.addEventListener('change', handleMotionPreferenceChange);
    }

    // Set canvas dimensions
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.scale(dpr, dpr);

      // Re-bound node positions
      nodes.forEach(node => {
        if (node.x > width) node.x = width * Math.random();
        if (node.y > height) node.y = height * Math.random();
      });
    };

    // Node initialization (capped at 40 nodes)
    const NODE_COUNT = 40;
    const MAX_DISTANCE = 140;
    const nodes = [];

    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push({
        x: Math.random() * (window.innerWidth || 1200),
        y: Math.random() * (window.innerHeight || 800),
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        radius: Math.random() * 2.2 + 1.8,
        type: Math.random() > 0.3 ? 'primary' : 'accent', // primary (indigo/cyan) vs accent (gold)
      });
    }

    // Active transaction pulses traveling along edges
    const pulses = [];
    let lastPulseTime = 0;

    const spawnPulse = (time) => {
      if (time - lastPulseTime < 1800) return; // spawn every ~1.8s
      lastPulseTime = time;

      // Pick a random node with neighbors
      const sourceIdx = Math.floor(Math.random() * nodes.length);
      const source = nodes[sourceIdx];
      const neighbors = [];

      for (let j = 0; j < nodes.length; j++) {
        if (sourceIdx === j) continue;
        const target = nodes[j];
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MAX_DISTANCE) {
          neighbors.push(target);
        }
      }

      if (neighbors.length > 0) {
        const target = neighbors[Math.floor(Math.random() * neighbors.length)];
        pulses.push({
          x1: source.x,
          y1: source.y,
          x2: target.x,
          y2: target.y,
          progress: 0,
          speed: 0.015 + Math.random() * 0.01,
        });
      }
    };

    // Palette configuration based on theme
    const isDark = theme === 'dark';
    const colors = {
      nodePrimary: isDark ? 'rgba(0, 210, 255, 0.45)' : 'rgba(79, 70, 229, 0.35)',
      nodeAccent: isDark ? 'rgba(247, 147, 26, 0.65)' : 'rgba(247, 147, 26, 0.55)',
      edgeBase: isDark ? 'rgba(0, 210, 255,' : 'rgba(79, 70, 229,',
      pulse: isDark ? 'rgba(247, 147, 26, 0.85)' : 'rgba(247, 147, 26, 0.85)',
      pulseGlow: isDark ? 'rgba(0, 210, 255, 0.9)' : 'rgba(79, 70, 229, 0.7)',
    };

    // Render single frame
    const drawFrame = (time) => {
      ctx.clearRect(0, 0, width, height);

      // Draw Edges
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i];
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < MAX_DISTANCE) {
            const alpha = (1 - dist / MAX_DISTANCE) * (isDark ? 0.12 : 0.08);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `${colors.edgeBase} ${alpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Draw and Update Pulses
      for (let i = pulses.length - 1; i >= 0; i--) {
        const p = pulses[i];
        p.progress += p.speed;

        if (p.progress >= 1) {
          pulses.splice(i, 1);
          continue;
        }

        const px = p.x1 + (p.x2 - p.x1) * p.progress;
        const py = p.y1 + (p.y2 - p.y1) * p.progress;

        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = colors.pulse;
        ctx.shadowColor = colors.pulseGlow;
        ctx.shadowBlur = isDark ? 8 : 4;
        ctx.fill();
        ctx.shadowBlur = 0; // reset
      }

      // Draw Nodes
      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = n.type === 'accent' ? colors.nodeAccent : colors.nodePrimary;
        ctx.fill();

        // Update positions if animating
        if (!prefersReducedMotion) {
          n.x += n.vx;
          n.y += n.vy;

          if (n.x < 0 || n.x > width) n.vx *= -1;
          if (n.y < 0 || n.y > height) n.vy *= -1;
        }
      }
    };

    // Animation Loop
    let lastTime = performance.now();
    const loop = (time) => {
      if (!isRunning) return;

      spawnPulse(time);
      drawFrame(time);

      if (!prefersReducedMotion) {
        animationFrameId = requestAnimationFrame(loop);
      }
    };

    // Page Visibility API handler
    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
          animationFrameId = null;
        }
      } else {
        if (!prefersReducedMotion && !animationFrameId) {
          lastTime = performance.now();
          animationFrameId = requestAnimationFrame(loop);
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('resize', resize);

    resize();

    if (prefersReducedMotion) {
      drawFrame(0);
    } else {
      animationFrameId = requestAnimationFrame(loop);
    }

    return () => {
      isRunning = false;
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('resize', resize);
      if (reducedMotionQuery.removeEventListener) {
        reducedMotionQuery.removeEventListener('change', handleMotionPreferenceChange);
      }
    };
  }, [theme]);

  return (
    <canvas
      ref={canvasRef}
      className="network-bg-canvas"
      aria-hidden="true"
    />
  );
}
