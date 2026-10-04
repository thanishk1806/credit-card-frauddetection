import React, { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { dashboardApi } from '../services/api'
import Loading from '../components/Loading'
import ErrorBanner from '../components/ErrorBanner'

// Professional chart tooltip
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
      {label !== undefined && <div style={{ color: '#8b97b0', marginBottom: 4 }}>{label}</div>}
      {payload.map((p) => (
        <div key={p.dataKey || p.name} style={{ color: p.color, marginBottom: 2 }}>
          {p.name || p.dataKey}: <strong>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

export default function ModelComparison() {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    ;(async () => {
      try {
        const { data } = await dashboardApi.modelPerformance()
        setSummary(data)
      } catch (err) {
        setError(
          err.response?.status === 503
            ? 'Models have not been trained yet. Run `python scripts/train_models.py` first.'
            : 'Failed to load model performance data.'
        )
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  if (loading) return <Loading label="Loading model evaluation metrics..." />
  if (error) return <ErrorBanner message={error} />
  if (!summary) return null

  const { all_models: models, best_model: best } = summary

  const rocData = []
  models.forEach((m) => {
    if (m.roc_curve?.fpr && m.roc_curve?.tpr) {
      m.roc_curve.fpr.forEach((fpr, i) => {
        if (!rocData[i]) rocData[i] = { fpr: Number(fpr.toFixed(3)) }
        rocData[i][m.model_name] = Number(m.roc_curve.tpr[i]?.toFixed(3))
      })
    }
  })

  const prData = []
  models.forEach((m) => {
    if (m.pr_curve?.recall && m.pr_curve?.precision) {
      m.pr_curve.recall.forEach((rec, i) => {
        if (!prData[i]) prData[i] = { recall: Number(rec.toFixed(3)) }
        prData[i][m.model_name] = Number(m.pr_curve.precision[i]?.toFixed(3))
      })
    }
  })

  const lineColors = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']

  return (
    <div>
      <div className="page-header">
        <h1>Model Performance</h1>
        <p>Evaluation metrics and validation curves computed on the held-out test dataset.</p>
      </div>

      {/* ── Metric Comparison Table ── */}
      <div className="card" style={{ marginBottom: 20 }}>
        <h3 style={{ marginBottom: 6, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
          Model Metric Comparison
        </h3>
        <p style={{ margin: '0 0 14px', fontSize: 12, color: 'var(--color-text-muted)' }}>
          Comparative evaluation results across all five trained ensemble architectures:
        </p>

        <div style={{ overflowX: 'auto' }}>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
                <th>PR-AUC</th>
                <th>MCC</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const isBest = best && m.model_name === best.model_name
                return (
                  <tr key={m.model_name} className={isBest ? 'best-row' : ''}>
                    <td style={{ fontWeight: isBest ? 600 : 500 }}>
                      {m.model_name}
                      {isBest && (
                        <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--color-primary)', fontWeight: 600 }}>
                          (Selected Best)
                        </span>
                      )}
                    </td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{m.precision.toFixed(3)}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{m.recall.toFixed(3)}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 600 }}>{m.f1_score.toFixed(3)}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{m.roc_auc.toFixed(3)}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{m.pr_auc.toFixed(3)}</td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{m.mcc.toFixed(3)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Best Model Selection Summary ── */}
      {best && (
        <div className="card" style={{ marginBottom: 20 }}>
          <h3 style={{ marginBottom: 4, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
            Selected Production Model: {best.model_name}
          </h3>
          <p style={{ color: 'var(--color-text-muted)', fontSize: 13, margin: 0, lineHeight: 1.5 }}>
            {best.reason}
          </p>
        </div>
      )}

      {/* ── Evaluation Curves ── */}
      <div className="grid grid-2">
        <div className="card">
          <h3 style={{ marginBottom: 4, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
            ROC Curves
          </h3>
          <p style={{ margin: '0 0 12px', fontSize: 12, color: 'var(--color-text-muted)' }}>
            True Positive Rate vs. False Positive Rate
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
              <XAxis dataKey="fpr" stroke="#5b6884" fontSize={11} />
              <YAxis stroke="#5b6884" fontSize={11} domain={[0, 1]} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#8b97b0' }} />
              {models.map((m, idx) => (
                <Line
                  key={m.model_name}
                  type="monotone"
                  dataKey={m.model_name}
                  stroke={lineColors[idx % lineColors.length]}
                  dot={false}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 style={{ marginBottom: 4, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
            Precision-Recall Curves
          </h3>
          <p style={{ margin: '0 0 12px', fontSize: 12, color: 'var(--color-text-muted)' }}>
            Precision vs. Recall trade-off across classification thresholds
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={prData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263248" />
              <XAxis dataKey="recall" stroke="#5b6884" fontSize={11} />
              <YAxis stroke="#5b6884" fontSize={11} domain={[0, 1]} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 12, color: '#8b97b0' }} />
              {models.map((m, idx) => (
                <Line
                  key={m.model_name}
                  type="monotone"
                  dataKey={m.model_name}
                  stroke={lineColors[idx % lineColors.length]}
                  dot={false}
                  strokeWidth={2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
