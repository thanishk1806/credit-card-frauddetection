import React, { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, Legend, PieChart, Pie, Cell,
} from 'recharts'
import { dashboardApi, predictionApi } from '../services/api'
import Loading from '../components/Loading'
import ErrorBanner from '../components/ErrorBanner'
import { FraudBadge, RiskLevelBadge } from '../components/RiskBadge'

const COLORS = ['#29c48c', '#ef4b5f']

export default function Dashboard() {
  const [summary, setSummary] = useState(null)
  const [history, setHistory] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    (async () => {
      try {
        const [summaryRes, historyRes] = await Promise.all([
          dashboardApi.summary(),
          predictionApi.history(10).catch(() => ({ data: [] })),
        ])
        setSummary(summaryRes.data)
        setHistory(historyRes.data)
      } catch (err) {
        setError(
          err.response?.status === 503
            ? 'Models have not been trained yet. Run `python scripts/train_models.py` in the backend, then reload this page.'
            : 'Failed to load dashboard data.'
        )
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  if (loading) return <Loading label="Loading dashboard..." />
  if (error) return <ErrorBanner message={error} />
  if (!summary) return null

  const { dataset, best_model: bestModel, all_models: allModels } = summary

  const pieData = [
    { name: 'Legitimate', value: dataset.legitimate_transactions },
    { name: 'Fraud', value: dataset.fraud_transactions },
  ]

  const metricChartData = allModels.map((m) => ({
    name: m.model_name,
    F1: Number(m.f1_score.toFixed(3)),
    'PR-AUC': Number(m.pr_auc.toFixed(3)),
    'ROC-AUC': Number(m.roc_auc.toFixed(3)),
  }))

  return (
    <div>
      <div className="page-header">
        <h1>Credit Card Fraud Detection</h1>
        <p>Ensemble ML model comparison and live fraud analytics overview.</p>
      </div>

      <div className="grid grid-4">
        <div className="card">
          <h3>Total Transactions</h3>
          <div className="stat">{dataset.total_transactions.toLocaleString()}</div>
          <div className="stat-sub">from training dataset</div>
        </div>
        <div className="card">
          <h3>Fraud Cases</h3>
          <div className="stat">{dataset.fraud_transactions.toLocaleString()}</div>
          <div className="stat-sub">{dataset.fraud_percentage}% of total</div>
        </div>
        <div className="card">
          <h3>Fraud Rate</h3>
          <div className="stat">{dataset.fraud_percentage}%</div>
          <div className="stat-sub">highly imbalanced</div>
        </div>
        <div className="card">
          <h3>Best Model</h3>
          <div className="stat" style={{ fontSize: 20 }}>{bestModel.model_name}</div>
          <div className="stat-sub">by {bestModel.selection_metric.replace('_', ' ')}</div>
        </div>
      </div>

      <div className="section-title">Model Performance Comparison</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={metricChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#263154" />
            <XAxis dataKey="name" stroke="#93a0c2" fontSize={12} />
            <YAxis stroke="#93a0c2" fontSize={12} domain={[0, 1]} />
            <Tooltip contentStyle={{ background: '#12182b', border: '1px solid #263154' }} />
            <Legend />
            <Bar dataKey="F1" fill="#5b7cfa" radius={[4, 4, 0, 0]} />
            <Bar dataKey="PR-AUC" fill="#29c48c" radius={[4, 4, 0, 0]} />
            <Bar dataKey="ROC-AUC" fill="#f5a623" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-2" style={{ marginTop: 20 }}>
        <div className="card">
          <h3 style={{ marginBottom: 12 }}>Class Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} dataKey="value" nameKey="name" outerRadius={80} label>
                {pieData.map((entry, index) => (
                  <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ background: '#12182b', border: '1px solid #263154' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h3 style={{ marginBottom: 12 }}>ROC Curve — Best Model ({bestModel.model_name})</h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={bestModel.metrics.roc_curve.fpr.map((fpr, i) => ({
              fpr: Number(fpr.toFixed(3)), tpr: Number(bestModel.metrics.roc_curve.tpr[i].toFixed(3)),
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#263154" />
              <XAxis dataKey="fpr" stroke="#93a0c2" fontSize={11} label={{ value: 'FPR', position: 'insideBottom', offset: -2, fill: '#93a0c2' }} />
              <YAxis stroke="#93a0c2" fontSize={11} />
              <Tooltip contentStyle={{ background: '#12182b', border: '1px solid #263154' }} />
              <Line type="monotone" dataKey="tpr" stroke="#5b7cfa" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="section-title">Recent Predictions</div>
      <div className="card">
        {history.length === 0 ? (
          <div className="empty-state">No predictions yet. Go to "Predict Transaction" to analyze one.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Date</th>
                <th>Model</th>
                <th>Prediction</th>
                <th>Probability</th>
                <th>Risk Score</th>
                <th>Risk Level</th>
              </tr>
            </thead>
            <tbody>
              {history.map((p) => (
                <tr key={p.id}>
                  <td>{new Date(p.created_at).toLocaleString()}</td>
                  <td>{p.model_name}</td>
                  <td><FraudBadge isFraud={p.is_fraud} /></td>
                  <td>{(p.fraud_probability * 100).toFixed(2)}%</td>
                  <td>{p.risk_score.toFixed(0)}/100</td>
                  <td><RiskLevelBadge level={p.risk_level} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
