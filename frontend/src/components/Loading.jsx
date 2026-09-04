import React from 'react'

export default function Loading({ label = 'Loading...' }) {
  return (
    <div className="loading-state">
      <div className="spinner" />
      {label}
    </div>
  )
}
