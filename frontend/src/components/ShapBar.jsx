import React from 'react'

export default function ShapBar({ feature, label, shapValue = 0, maxAbs = 1, direction, description }) {
  const numVal = typeof shapValue === 'number' && !isNaN(shapValue) ? shapValue : 0
  const numMax = typeof maxAbs === 'number' && maxAbs > 0 ? maxAbs : 1
  const pct = Math.min(100, (Math.abs(numVal) / numMax) * 100)
  const isFraud = direction === 'fraud' || numVal > 0

  return (
    <div style={{ marginBottom: 10, background: 'var(--color-surface-2)', padding: '10px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--color-text)' }}>
          {label || feature}
          {label && label !== feature && (
            <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--color-text-muted)', fontWeight: 400 }}>({feature})</span>
          )}
        </div>
        <div style={{ fontWeight: 600, fontSize: 12, color: isFraud ? 'var(--color-danger)' : 'var(--color-success)', fontVariantNumeric: 'tabular-nums' }}>
          {numVal > 0 ? '+' : ''}{numVal.toFixed(4)}
        </div>
      </div>

      <div className="shap-bar-track" style={{ height: 6, background: 'var(--color-surface-3)' }}>
        <div
          className={`shap-bar-fill ${isFraud ? 'fraud' : 'legit'}`}
          style={{ width: `${Math.max(pct, 3)}%` }}
        />
      </div>

      {description && (
        <div style={{ marginTop: 5, fontSize: 11, color: 'var(--color-text-muted)', lineHeight: 1.3 }}>
          {description}
        </div>
      )}
    </div>
  )
}
