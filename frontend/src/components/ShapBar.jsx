import React from 'react'

export default function ShapBar({ feature, shapValue, maxAbs, direction }) {
  const pct = maxAbs > 0 ? Math.min(100, (Math.abs(shapValue) / maxAbs) * 100) : 0
  return (
    <div className="shap-row">
      <div className="feature-name">{feature}</div>
      <div className="shap-bar-track">
        <div
          className={`shap-bar-fill ${direction}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="shap-value">{shapValue > 0 ? '+' : ''}{shapValue.toFixed(3)}</div>
    </div>
  )
}
