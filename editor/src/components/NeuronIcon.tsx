import React from 'react';

const FNS: Record<string, (x: number) => number> = {
  sigmoid: x => 1 / (1 + Math.exp(-x)),
  relu: x => Math.max(0, x),
  tanh_neuron: x => Math.tanh(x),
  threshold: x => x >= 0 ? 1 : 0,
  identity: x => x,
  negate: x => -x,
  gaussian: x => Math.exp(-x * x),
  log: x => Math.log(Math.max(x, 1e-7)),
  leaky_relu: x => x >= 0 ? x : 0.1 * x, // 0.1 slope for better visual representation
  prelu: x => x >= 0 ? x : 0.25 * x,
  relu6: x => Math.min(Math.max(0, x), 6),
  elu: x => x >= 0 ? x : Math.exp(x) - 1,
  selu: x => 1.0507 * (x >= 0 ? x : 1.6732 * (Math.exp(x) - 1)),
  gelu: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))),
  silu: x => x / (1 + Math.exp(-x)),
  mish: x => x * Math.tanh(Math.log(1 + Math.exp(x))),
  softplus: x => Math.log(1 + Math.exp(x)),
  softsign: x => x / (1 + Math.abs(x)),
  hard_sigmoid: x => Math.max(0, Math.min(1, x / 6 + 0.5)),
  hard_tanh: x => Math.max(-1, Math.min(1, x)),
  hard_swish: x => x * Math.max(0, Math.min(1, x / 6 + 0.5)),
};

export default function NeuronIcon({ name, expanded }: { name: string, expanded?: boolean }) {
  const fn = FNS[name];
  if (!fn) return null;

  const points = [];
  const range = [-4, 4];
  const steps = 40;
  
  let minY = Infinity;
  let maxY = -Infinity;
  const rawPoints = [];

  for (let i = 0; i <= steps; i++) {
    const x = range[0] + (range[1] - range[0]) * (i / steps);
    const y = fn(x);
    if (isFinite(y)) {
       minY = Math.min(minY, y);
       maxY = Math.max(maxY, y);
       rawPoints.push([x, y]);
    }
  }

  const padY = (maxY - minY) * 0.15 || 1;
  minY -= padY;
  maxY += padY;

  for (const [x, y] of rawPoints) {
    const px = ((x - range[0]) / (range[1] - range[0])) * 100;
    const py = 100 - ((y - minY) / (maxY - minY)) * 100;
    points.push(`${px},${py}`);
  }

  const path = points.join(" ");
  const zeroX = ((0 - range[0]) / (range[1] - range[0])) * 100;
  const zeroY = 100 - ((0 - Math.min(Math.max(0, minY), maxY)) / (maxY - minY)) * 100;

  const size = expanded ? 160 : 14;

  return (
    <svg 
      width={size} 
      height={size} 
      viewBox="0 0 100 100" 
      className={`overflow-visible inline-block ${expanded ? '' : 'ml-1.5 opacity-80 group-hover:opacity-100 transition-opacity'}`}
    >
      <line x1="0" y1={zeroY} x2="100" y2={zeroY} stroke="#4b5563" strokeWidth={expanded ? 1.5 : 8} />
      <line x1={zeroX} y1="0" x2={zeroX} y2="100" stroke="#4b5563" strokeWidth={expanded ? 1.5 : 8} />
      <polyline points={path} fill="none" stroke={expanded ? "#60a5fa" : "#93c5fd"} strokeWidth={expanded ? 3.5 : 12} strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
