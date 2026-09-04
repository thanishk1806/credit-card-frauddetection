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
    (async () => {
      try {
        const { data } = await predictionApi.history(50)
        setHistory(data)
      } catch {
        setError('Failed to load prediction history.')
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
      link.download = `fraud_report_${id}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch {
      setError('Failed to generate PDF report for this prediction.')
    } finally {
      setDownloadingId(null)
    }
  }

  if (loading) return <Loading label="Loading reports..." />

  return (
    <div>
      <div className="page-header">
        <h1>Reports</h1>
        <p>Download a detailed PDF fraud analysis report for any past prediction.</p>
      </div>

      <ErrorBanner message={error} />

      <div className="card">
        {history.length === 0 ? (
          <div className="empty-state">
            No predictions yet. Analyze a transaction on the "Predict Transaction" page first.
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Model</th>
                <th>Prediction</th>
                <th>Risk Level</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {history.map((p) => (
                <tr key={p.id}>
                  <td>{new Date(p.created_at).toLocaleString()}</td>
                  <td>{p.model_name}</td>
                  <td><FraudBadge isFraud={p.is_fraud} /></td>
                  <td><RiskLevelBadge level={p.risk_level} /></td>
                  <td>
                    <button
                      className="btn btn-secondary"
                      onClick={() => handleDownload(p.id)}
                      disabled={downloadingId === p.id}
                    >
                      {downloadingId === p.id ? 'Preparing...' : '⬇ PDF'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
