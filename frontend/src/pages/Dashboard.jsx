import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts'
import { dashboardApi, predictionApi } from '../services/api'
import Loading from '../components/Loading'
import ErrorBanner from '../components/ErrorBanner'
import { FraudBadge, RiskLevelBadge } from '../components/RiskBadge'

// Restrained semantic chart colors
const PIE_COLORS_PRED = ['#22c55e', '#ef4444'] // legitimate=green, fraud=red
const RISK_COLORS = ['#22c55e', '#f59e0b', '#ef4444'] // low, medium, high

// Tooltip formatter helper
const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#151b26',
      border: '1px solid #263248',
      borderRadius: 4,
      padding: '8px 12px',
      fontSize: 12,
    }}>
      {label && <div style={{ color: '#8b97b0', marginBottom: 4 }}>{label}</div>}
      {payload.map((p) => (
        <div key={p.dataKey || p.name} style={{ color: p.color || '#e2e8f0', marginBottom: 2 }}>
          {p.name || p.dataKey}: <strong>{p.value}</strong>
        </div>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [liveStats, setLiveStats] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    ;(async () => {
      try {
        const [liveRes, historyRes] = await Promise.all([
          dashboardApi.liveStats(),
          predictionApi.history(10).catch(() => ({ data: [] })),
        ])
        setLiveStats(liveRes.data)
        setHistory(historyRes.data)
      } catch (err) {
        setError('Failed to load application activity records.')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  if (loading) return <Loading label="Loading activity dashboard..." />
  if (error) return <ErrorBanner message={error} />
  if (!liveStats) return null

  const hasPredictions = liveStats.total_analyzed > 0

  // Live prediction distribution (actual records)
  const predDistData = [
    { name: 'Legitimate', value: liveStats.prediction_distribution.legitimate },
    { name: 'Fraud', value: liveStats.prediction_distribution.fraud },
  ]

  // Risk level distribution (actual records)
  const riskDistData = [
    { name: 'Low Risk', value: liveStats.risk_level_distribution.low },
    { name: 'Medium Risk', value: liveStats.risk_level_distribution.medium },
    { name: 'High Risk', value: liveStats.risk_level_distribution.high },
  ]

  return (
    <div>
      <div className="page-header">
        <h1>Dashboard</h1>
        <p>Review real-time transaction activity and risk analysis across submitted transactions.</p>
      </div>

      {/* ── Live Stats Summary Cards ── */}
      <div className="grid grid-4">
        <div className="card">
          <h3>Transactions Analyzed</h3>
          <div className="stat">{liveStats.total_analyzed.toLocaleString()}</div>
          <div className="stat-sub">submitted to TrustCheck</div>
        </div>
        <div className="card">
          <h3>Fraud Detected</h3>
          <div className="stat" style={{ color: liveStats.fraud_detected > 0 ? 'var(--color-danger)' : 'inherit' }}>
            {liveStats.fraud_detected.toLocaleString()}
          </div>
          <div className="stat-sub">flagged as fraudulent</div>
        </div>
        <div className="card">
          <h3>Detection Rate</h3>
          <div className="stat">{hasPredictions ? `${liveStats.detection_rate}%` : '0%'}</div>
          <div className="stat-sub">fraud / total analyzed</div>
        </div>
        <div className="card">
          <h3>Average Risk Score</h3>
          <div className="stat">
            {hasPredictions ? `${liveStats.avg_risk_score}` : '0'}
            <span style={{ fontSize: 13, fontWeight: 400, color: 'var(--color-text-muted)' }}> / 100</span>
          </div>
          <div className="stat-sub">across all transactions</div>
        </div>
      </div>

      {/* ── Activity Distribution Charts ── */}
      <div className="grid grid-2" style={{ marginTop: 16 }}>
        {/* Prediction Distribution */}
        <div className="card">
          <h3 style={{ marginBottom: 4 }}>Prediction Distribution</h3>
          <p style={{ margin: '0 0 14px', fontSize: 12, color: 'var(--color-text-muted)' }}>
            Breakdown of legitimate vs. fraudulent transactions
          </p>
          {!hasPredictions ? (
            <div className="empty-state" style={{ padding: '36px 0' }}>
              No transactions analyzed yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={predDistData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={75}
                  label={({ name, value }) => `${name}: ${value}`}
                  labelLine={false}
                >
                  {predDistData.map((entry, index) => (
                    <Cell key={entry.name} fill={PIE_COLORS_PRED[index % PIE_COLORS_PRED.length]} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#8b97b0' }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Risk Level Distribution */}
        <div className="card">
          <h3 style={{ marginBottom: 4 }}>Risk Level Distribution</h3>
          <p style={{ margin: '0 0 14px', fontSize: 12, color: 'var(--color-text-muted)' }}>
            Low (0–29) · Medium (30–69) · High (70–100)
          </p>
          {!hasPredictions ? (
            <div className="empty-state" style={{ padding: '36px 0' }}>
              No transactions analyzed yet.
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={riskDistData} barSize={36}>
                <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
                <XAxis dataKey="name" stroke="#5b6884" fontSize={11} />
                <YAxis stroke="#5b6884" fontSize={11} allowDecimals={false} />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="value" name="Transactions" radius={[2, 2, 0, 0]}>
                  {riskDistData.map((entry, index) => (
                    <Cell key={entry.name} fill={RISK_COLORS[index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* ── Recent Predictions Table ── */}
      <div className="section-title">Recent Predictions</div>
      <div className="card">
        {history.length === 0 ? (
          <div className="empty-state">
            No transactions analyzed yet. Go to "Predict Transaction" to analyze one.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Date &amp; Time</th>
                  <th>Amount</th>
                  <th>Type</th>
                  <th>Prediction</th>
                  <th>Risk Score</th>
                  <th>Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {history.map((p) => (
                  <tr key={p.id}>
                    <td style={{ color: 'var(--color-text-muted)', fontSize: 12 }}>
                      {new Date(p.created_at).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}
                    </td>
                    <td style={{ fontWeight: 600 }}>
                      {p.amount != null ? `$${Number(p.amount).toFixed(2)}` : 'N/A'}
                    </td>
                    <td style={{ color: 'var(--color-text-muted)', textTransform: 'capitalize', fontSize: 12 }}>
                      {(p.transaction_type || '—').replace(/_/g, ' ')}
                    </td>
                    <td><FraudBadge isFraud={p.is_fraud} /></td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{p.risk_score.toFixed(0)}&thinsp;/&thinsp;100</td>
                    <td><RiskLevelBadge level={p.risk_level} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
