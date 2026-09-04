import React from 'react'

export function FraudBadge({ isFraud }) {
  return (
    <span className={`badge ${isFraud ? 'badge-fraud' : 'badge-legit'}`}>
      {isFraud ? '⚠ FRAUD' : '✓ NOT FRAUD'}
    </span>
  )
}

export function RiskLevelBadge({ level }) {
  const cls = level === 'HIGH' ? 'badge-high' : level === 'MEDIUM' ? 'badge-medium' : 'badge-low'
  return <span className={`badge ${cls}`}>{level} RISK</span>
}
