import React, { useState } from 'react'
import { predictionApi, reportApi } from '../services/api'
import ErrorBanner from '../components/ErrorBanner'
import { FraudBadge, RiskLevelBadge } from '../components/RiskBadge'
import ShapBar from '../components/ShapBar'

const ALL_V_FIELDS = Array.from({ length: 28 }, (_, i) => `V${i + 1}`)

// The 5 primary user-facing inputs
const PRIMARY_FIELDS = [
  {
    name: 'Amount',
    label: 'Transaction Amount ($)',
    placeholder: 'e.g. 149.99',
    description: 'Transaction value in USD',
    defaultValue: '149.99',
    step: '0.01',
    min: '0',
    required: true,
  },
  {
    name: 'Time',
    label: 'Time Offset (seconds)',
    placeholder: 'e.g. 3600',
    description: 'Seconds elapsed in window (0 - 172,800)',
    defaultValue: '3600',
    step: '1',
    min: '0',
    required: false,
  },
  {
    name: 'V14',
    label: 'V14 - Security & Pattern Score',
    placeholder: '0.0000',
    description: 'Key fraud signal (Negative values < -2 indicate elevated risk)',
    defaultValue: '0.0',
    step: 'any',
    required: false,
  },
  {
    name: 'V17',
    label: 'V17 - Location & Device Anomaly',
    placeholder: '0.0000',
    description: 'Identity metric (Negative values < -2 indicate anomaly)',
    defaultValue: '0.0',
    step: 'any',
    required: false,
  },
  {
    name: 'V4',
    label: 'V4 - Velocity / Activity Index',
    placeholder: '0.0000',
    description: 'Frequency indicator (Positive values > +2 indicate velocity spike)',
    defaultValue: '0.0',
    step: 'any',
    required: false,
  },
]

const PRESETS = [
  {
    name: '🟢 Normal Purchase',
    description: 'Standard retail store purchase (Low Risk)',
    values: {
      Amount: '32.50',
      Time: '4500',
      V14: '0.15',
      V17: '0.08',
      V4: '-0.25',
    },
  },
  {
    name: '🔴 Stolen Card Attack',
    description: 'High-velocity unusual transaction (High Risk Fraud)',
    values: {
      Amount: '940.00',
      Time: '7200',
      V14: '-5.85',
      V17: '-4.60',
      V4: '4.95',
    },
  },
  {
    name: '🟡 Suspicious Surge',
    description: 'Unusual amount and abnormal pattern (Medium Risk)',
    values: {
      Amount: '450.00',
      Time: '1800',
      V14: '-2.40',
      V17: '-1.85',
      V4: '2.50',
    },
  },
]

function defaultForm() {
  const f = {
    Amount: '149.99',
    Time: '3600',
    V14: '0.0',
    V17: '0.0',
    V4: '0.0',
  }
  ALL_V_FIELDS.forEach((v) => {
    if (!f[v]) f[v] = '0.0'
  })
  return f
}

export default function Predict() {
  const [form, setForm] = useState(defaultForm())
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  const applyPreset = (preset) => {
    const next = defaultForm()
    Object.entries(preset.values).forEach(([k, v]) => {
      next[k] = v
    })
    setForm(next)
  }

  const handleRandomize = () => {
    const next = defaultForm()
    next.Amount = (Math.random() * 400 + 5).toFixed(2)
    next.Time = Math.floor(Math.random() * 172792).toString()
    next.V14 = (Math.random() * 6 - 3).toFixed(4)
    next.V17 = (Math.random() * 6 - 3).toFixed(4)
    next.V4 = (Math.random() * 6 - 3).toFixed(4)
    setForm(next)
  }

  const handleReset = () => {
    setForm(defaultForm())
    setResult(null)
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const payload = {}
      ALL_V_FIELDS.forEach((v) => {
        payload[v] = 0.0
      })
      Object.entries(form).forEach(([k, v]) => {
        payload[k] = v === '' ? 0.0 : Number(v)
      })
      // Ensure required Amount is a number
      payload.Amount = Number(form.Amount) || 0.0

      const { data } = await predictionApi.predict(payload)
      setResult(data)
    } catch (err) {
      setError(
        err.response?.status === 503
          ? 'Models have not been trained yet. Run `python scripts/train_models.py` in the backend first.'
          : err.response?.data?.detail || 'Prediction failed. Please check your input values.'
      )
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async () => {
    if (!result) return
    setDownloading(true)
    try {
      const response = await reportApi.download(result.prediction_id)
      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }))
      const link = document.createElement('a')
      link.href = url
      link.download = `fraud_report_${result.prediction_id}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch {
      setError('Failed to generate PDF report.')
    } finally {
      setDownloading(false)
    }
  }

  const maxAbsShap = result
    ? Math.max(
        ...result.top_fraud_contributors.map((c) => Math.abs(c.shap_value)),
        ...result.top_legitimate_contributors.map((c) => Math.abs(c.shap_value)),
        0.0001
      )
    : 1

  return (
    <div>
      <div className="page-header">
        <h1>Predict Transaction</h1>
        <p>Evaluate credit card transaction risk with 5 core inputs powered by ensemble Machine Learning and Explainable AI.</p>
      </div>

      <ErrorBanner message={error} />

      {/* Preset simulation buttons */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 10 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 15, color: 'var(--color-text)' }}>⚡ Quick Test Scenarios</h3>
            <p style={{ margin: '3px 0 0', fontSize: 13, color: 'var(--color-text-muted)' }}>
              Click any scenario to populate the 5 transaction inputs instantly:
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {PRESETS.map((preset) => (
              <button
                key={preset.name}
                type="button"
                className="btn btn-secondary"
                style={{ fontSize: 13, padding: '6px 12px' }}
                onClick={() => applyPreset(preset)}
                title={preset.description}
              >
                {preset.name}
              </button>
            ))}
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: 13, padding: '6px 12px' }}
              onClick={handleRandomize}
            >
              🎲 Random
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ fontSize: 13, padding: '6px 12px', opacity: 0.8 }}
              onClick={handleReset}
            >
              ↺ Reset
            </button>
          </div>
        </div>
      </div>

      {/* 5 Core Inputs Form */}
      <form onSubmit={handleSubmit} className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 16, color: 'var(--color-text)', fontWeight: 600 }}>
              Primary Transaction Inputs (5 Core Features)
            </h3>
            <span style={{ fontSize: 13, color: 'var(--color-text-muted)' }}>
              Fill the 5 intuitive fields below to analyze risk. Remaining background PCA features default to 0.0.
            </span>
          </div>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 16,
          }}
        >
          {PRIMARY_FIELDS.map((field) => (
            <div
              key={field.name}
              style={{
                background: 'var(--color-surface-2)',
                padding: '14px',
                borderRadius: '8px',
                border: '1px solid var(--color-border)',
              }}
            >
              <label
                style={{
                  display: 'block',
                  fontSize: 13,
                  fontWeight: 600,
                  color: 'var(--color-text)',
                  marginBottom: 4,
                }}
              >
                {field.label} {field.required && <span style={{ color: 'var(--color-danger)' }}>*</span>}
              </label>
              <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginBottom: 8, minHeight: 28 }}>
                {field.description}
              </div>
              <input
                type="number"
                step={field.step}
                min={field.min}
                required={field.required}
                placeholder={field.placeholder}
                value={form[field.name] ?? ''}
                onChange={(e) => handleChange(field.name, e.target.value)}
                style={{
                  width: '100%',
                  padding: '9px 12px',
                  borderRadius: '6px',
                  border: '1px solid var(--color-border)',
                  background: 'var(--color-surface)',
                  color: 'var(--color-text)',
                  fontSize: 14,
                  fontWeight: 600,
                }}
              />
            </div>
          ))}
        </div>

        {/* Optional Collapsible Advanced Features for academic completeness */}
        <div style={{ marginTop: 20, borderTop: '1px solid var(--color-border)', paddingTop: 16 }}>
          <button
            type="button"
            className="btn btn-secondary"
            style={{ fontSize: 13, padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 6 }}
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <span>{showAdvanced ? '▼ Hide' : '► Show'} Advanced PCA Features (V1–V28)</span>
            <span style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
              ({showAdvanced ? 'Custom values loaded' : 'Currently defaulting to 0.0'})
            </span>
          </button>

          {showAdvanced && (
            <div style={{ marginTop: 16, background: 'rgba(0,0,0,0.2)', padding: 16, borderRadius: 8 }}>
              <p style={{ margin: '0 0 12px', fontSize: 12, color: 'var(--color-text-muted)' }}>
                Secondary anonymized PCA variables (V1–V28 excluding V4, V14, V17). Defaults to 0.0 (dataset mean).
              </p>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))',
                  gap: 10,
                }}
              >
                {ALL_V_FIELDS.filter((v) => !['V14', 'V17', 'V4'].includes(v)).map((v) => (
                  <div key={v} className="field" style={{ margin: 0 }}>
                    <label style={{ fontSize: 11 }}>{v}</label>
                    <input
                      type="number"
                      step="any"
                      value={form[v] ?? '0.0'}
                      onChange={(e) => handleChange(v, e.target.value)}
                      style={{ padding: '6px 8px', fontSize: 12 }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 12, marginTop: 24 }}>
          <button
            className="btn btn-primary"
            type="submit"
            disabled={loading}
            style={{ width: 'auto', minWidth: 220, padding: '12px 24px', fontSize: 15 }}
          >
            {loading ? 'Analyzing with Ensemble Models...' : '🔍 Analyze Transaction'}
          </button>
        </div>
      </form>

      {/* Results Section */}
      {result && (
        <div style={{ marginTop: 28 }}>
          <div className={`result-hero ${result.is_fraud ? 'fraud' : 'legit'}`}>
            <div className={`verdict ${result.is_fraud ? 'fraud-text' : 'legit-text'}`}>
              {result.is_fraud ? '⚠ HIGH RISK / FRAUD DETECTED' : '✓ TRANSACTION VERIFIED LEGITIMATE'}
            </div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginTop: 8 }}>
              <FraudBadge isFraud={result.is_fraud} />
              <RiskLevelBadge level={result.risk_level} />
            </div>

            <div className="result-stats">
              <div className="item">
                <div className="label">Fraud Probability</div>
                <div className="value" style={{ color: result.is_fraud ? 'var(--color-danger)' : 'inherit' }}>
                  {(result.fraud_probability * 100).toFixed(2)}%
                </div>
              </div>
              <div className="item">
                <div className="label">Legitimate Probability</div>
                <div className="value" style={{ color: !result.is_fraud ? 'var(--color-success)' : 'inherit' }}>
                  {(result.legitimate_probability * 100).toFixed(2)}%
                </div>
              </div>
              <div className="item">
                <div className="label">Risk Score</div>
                <div className="value">{result.risk_score.toFixed(0)} / 100</div>
              </div>
              <div className="item">
                <div className="label">Best ML Model</div>
                <div className="value">{result.model_used}</div>
              </div>
            </div>

            <button
              className="btn btn-primary"
              style={{ marginTop: 20, width: 'auto', minWidth: 220 }}
              onClick={handleDownload}
              disabled={downloading}
            >
              {downloading ? 'Generating Report...' : '⬇ Download Explanatory PDF Report'}
            </button>
          </div>

          <div className="card">
            <h3 style={{ marginBottom: 16, fontSize: 16, color: 'var(--color-text)', fontWeight: 600 }}>
              🧠 Explainable AI: Why Was This Transaction Flagged?
            </h3>
            <p style={{ margin: '0 0 16px', fontSize: 13, color: 'var(--color-text-muted)' }}>
              SHAP (SHapley Additive exPlanations) values computed for this prediction quantify each feature's contribution to the outcome:
            </p>

            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 13, color: 'var(--color-danger)', fontWeight: 700, marginBottom: 8 }}>
                🔴 Top Fraud Risk Drivers (Pushed toward Fraud)
              </div>
              {result.top_fraud_contributors.length === 0 && (
                <div style={{ color: 'var(--color-text-muted)', fontSize: 13 }}>No significant fraud drivers detected.</div>
              )}
              {result.top_fraud_contributors.map((c) => (
                <ShapBar key={c.feature} feature={c.feature} shapValue={c.shap_value} maxAbs={maxAbsShap} direction="fraud" />
              ))}
            </div>

            <div>
              <div style={{ fontSize: 13, color: 'var(--color-success)', fontWeight: 700, marginBottom: 8 }}>
                🟢 Top Legitimate Trust Drivers (Pushed toward Legitimate)
              </div>
              {result.top_legitimate_contributors.length === 0 && (
                <div style={{ color: 'var(--color-text-muted)', fontSize: 13 }}>No significant legitimate drivers detected.</div>
              )}
              {result.top_legitimate_contributors.map((c) => (
                <ShapBar key={c.feature} feature={c.feature} shapValue={c.shap_value} maxAbs={maxAbsShap} direction="legit" />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
