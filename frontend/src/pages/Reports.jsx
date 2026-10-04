import React, { useEffect, useState } from 'react'
import { predictionApi, reportApi } from '../services/api'
import Loading from '../components/Loading'
import ErrorBanner from '../components/ErrorBanner'
import { FraudBadge, RiskLevelBadge } from '../components/RiskBadge'

export default function Reports() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [downloadingId, setDownloadingId] = useState(null)

  useEffect(() => {
    ;(async () => {
      try {
        const { data } = await predictionApi.history(50)
        setHistory(data)
      } catch {
        setError('Failed to load transaction audit history.')
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  const handleDownload = async (id) => {
    setDownloadingId(id)
    try {
      const response = await reportApi.download(id)
      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }))
      const link = document.createElement('a')
      link.href = url
      link.download = `fraud_audit_report_${id}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch {
      setError('Failed to generate PDF report for this transaction.')
    } finally {
      setDownloadingId(null)
    }
  }

  if (loading) return <Loading label="Loading audit history..." />

  return (
    <div>
      <div className="page-header">
        <h1>Audit Reports &amp; History</h1>
        <p>Review forensic audit logs and download comprehensive PDF verification reports for analyzed transactions.</p>
      </div>

      <ErrorBanner message={error} />

      <div className="card">
        {history.length === 0 ? (
          <div className="empty-state">
            No transactions analyzed yet. Run a prediction on the "Predict Transaction" page first.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Date &amp; Time</th>
                  <th>Amount</th>
                  <th>Payment Channel</th>
                  <th>Merchant Category</th>
                  <th>Inference Model</th>
                  <th>Verdict</th>
                  <th>Risk Score</th>
                  <th>Audit PDF</th>
                </tr>
              </thead>
              <tbody>
                {history.map((p) => (
                  <tr key={p.id}>
                    <td style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
                      {new Date(p.created_at).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}
                    </td>
                    <td style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                      ${p.amount !== undefined && p.amount !== null ? Number(p.amount).toFixed(2) : '—'}
                    </td>
                    <td style={{ textTransform: 'capitalize', fontSize: 12 }}>
                      {(p.transaction_type || 'Online').replace(/_/g, ' ')}
                    </td>
                    <td style={{ textTransform: 'capitalize', fontSize: 12 }}>
                      {(p.merchant_category || 'Retail').replace(/_/g, ' ')}
                    </td>
                    <td style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>{p.model_name}</td>
                    <td><FraudBadge isFraud={p.is_fraud} /></td>
                    <td>
                      <RiskLevelBadge level={p.risk_level} />
                      <span style={{ fontSize: 12, marginLeft: 6, fontVariantNumeric: 'tabular-nums' }}>
                        ({p.risk_score.toFixed(0)})
                      </span>
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '4px 10px', fontSize: 12 }}
                        onClick={() => handleDownload(p.id)}
                        disabled={downloadingId === p.id}
                      >
                        {downloadingId === p.id ? 'Generating...' : 'PDF Report'}
                      </button>
                    </td>
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
