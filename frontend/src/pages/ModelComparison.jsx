import React, { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { dashboardApi } from '../services/api'
import Loading from '../components/Loading'
import ErrorBanner from '../components/ErrorBanner'

export default function ModelComparison() {
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
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

  if (loading) return <Loading label="Loading model comparison..." />
  if (error) return <ErrorBanner message={error} />
  if (!summary) return null

  const { all_models: models, best_model: best } = summary

  const rocData = []
  models.forEach((m) => {
    m.roc_curve.fpr.forEach((fpr, i) => {
      if (!rocData[i]) rocData[i] = { fpr: Number(fpr.toFixed(3)) }
      rocData[i][m.model_name] = Number(m.roc_curve.tpr[i].toFixed(3))
    })
  })

  const prData = []
  models.forEach((m) => {
    m.pr_curve.recall.forEach((rec, i) => {
      if (!prData[i]) prData[i] = { recall: Number(rec.toFixed(3)) }
      prData[i][m.model_name] = Number(m.pr_curve.precision[i].toFixed(3))
    })
  })

  const lineColors = ['#5b7cfa', '#29c48c', '#f5a623', '#ef4b5f', '#a06cd5']

  return (
    <div>
      <div className="page-header">
        <h1>Model Performance Comparison</h1>
        <p>Random Forest · AdaBoost · XGBoost · LightGBM · CatBoost — evaluated on the held-out test set.</p>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ marginBottom: 10 }}>Metric Comparison Table</h3>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>ROC-AUC</th>
              <th>PR-AUC</th>
              <th>MCC</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.model_name} className={m.model_name === best.model_name ? 'best-row' : ''}>
                <td>{m.model_name === best.model_name ? '⭐ ' : ''}{m.model_name}</td>
                <td>{m.precision.toFixed(4)}</td>
                <td>{m.recall.toFixed(4)}</td>
                <td>{m.f1_score.toFixed(4)}</td>
                <td>{m.roc_auc.toFixed(4)}</td>
                <td>{m.pr_auc.toFixed(4)}</td>
                <td>{m.mcc.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <h3>Best Model: {best.model_name}</h3>
        <p style={{ color: 'var(--color-text-muted)', fontSize: 14 }}>{best.reason}</p>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3 style={{ marginBottom: 12 }}>ROC Curves</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={rocData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263154" />
              <XAxis dataKey="fpr" stroke="#93a0c2" fontSize={11} />
              <YAxis stroke="#93a0c2" fontSize={11} />
              <Tooltip contentStyle={{ background: '#12182b', border: '1px solid #263154' }} />
              <Legend />
              {models.map((m, idx) => (
                <Line key={m.model_name} type="monotone" dataKey={m.model_name} stroke={lineColors[idx % lineColors.length]} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 style={{ marginBottom: 12 }}>Precision-Recall Curves</h3>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={prData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263154" />
              <XAxis dataKey="recall" stroke="#93a0c2" fontSize={11} />
              <YAxis stroke="#93a0c2" fontSize={11} />
              <Tooltip contentStyle={{ background: '#12182b', border: '1px solid #263154' }} />
              <Legend />
              {models.map((m, idx) => (
                <Line key={m.model_name} type="monotone" dataKey={m.model_name} stroke={lineColors[idx % lineColors.length]} dot={false} strokeWidth={2} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="section-title">Confusion Matrices</div>
      <div className="grid grid-4">
        {models.map((m) => (
          <div className="card" key={m.model_name}>
            <h3 style={{ marginBottom: 10 }}>{m.model_name}</h3>
            <table>
              <tbody>
                <tr>
                  <td style={{ color: 'var(--color-text-muted)' }}>TN: {m.confusion_matrix[0][0]}</td>
                  <td style={{ color: 'var(--color-danger)' }}>FP: {m.confusion_matrix[0][1]}</td>
                </tr>
                <tr>
                  <td style={{ color: 'var(--color-warning)' }}>FN: {m.confusion_matrix[1][0]}</td>
                  <td style={{ color: 'var(--color-success)' }}>TP: {m.confusion_matrix[1][1]}</td>
                </tr>
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  )
}
