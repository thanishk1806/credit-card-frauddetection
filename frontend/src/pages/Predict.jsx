import React, { useState } from 'react'
import { predictionApi, reportApi } from '../services/api'
import ErrorBanner from '../components/ErrorBanner'
import { FraudBadge, RiskLevelBadge } from '../components/RiskBadge'
import ShapBar from '../components/ShapBar'

const TRANSACTION_TYPES = [
  { value: 'online', label: 'Online / E-Commerce' },
  { value: 'pos', label: 'In-Store POS Terminal' },
  { value: 'atm', label: 'ATM Cash Withdrawal' },
  { value: 'mobile_app', label: 'Mobile Wallet / UPI' },
  { value: 'contactless', label: 'Contactless Tap & Pay' },
  { value: 'wire_transfer', label: 'Bank Wire / Transfer' },
]

const MERCHANT_CATEGORIES = [
  { value: 'grocery', label: 'Grocery & Supermarket' },
  { value: 'electronics', label: 'Electronics & Gadgets' },
  { value: 'food_dining', label: 'Food & Restaurants' },
  { value: 'fuel', label: 'Fuel & Gas Stations' },
  { value: 'travel_airlines', label: 'Travel & Airlines' },
  { value: 'entertainment', label: 'Entertainment & Gaming' },
  { value: 'healthcare', label: 'Healthcare & Pharmacy' },
  { value: 'luxury_jewelry', label: 'Luxury & Jewelry' },
  { value: 'retail_shopping', label: 'Retail & Shopping' },
  { value: 'utilities', label: 'Utilities & Bills' },
  { value: 'other', label: 'Other / General' },
]

function defaultForm() {
  const now = new Date()
  const localIso = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 16)
  return {
    amount: '',
    timestamp: localIso,
    transaction_type: 'online',
    merchant_category: 'retail_shopping',
    location: '',
    tx_velocity_5m: '0',
    card_present: false,
    international_transaction: false,
  }
}

export default function Predict() {
  const [form, setForm] = useState(defaultForm())
  const [currency, setCurrency] = useState('$')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [showShapBars, setShowShapBars] = useState(false)

  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setResult(null)

    const numAmount = parseFloat(form.amount)
    if (isNaN(numAmount) || numAmount <= 0) {
      setError('Please enter a valid transaction amount greater than 0.')
      return
    }

    const numVelocity = parseInt(form.tx_velocity_5m, 10)
    if (isNaN(numVelocity) || numVelocity < 0) {
      setError('Please enter a valid recent transaction activity (0 or greater).')
      return
    }

    setLoading(true)
    try {
      const payload = {
        amount: numAmount,
        timestamp: form.timestamp ? new Date(form.timestamp).toISOString() : new Date().toISOString(),
        transaction_type: form.transaction_type,
        merchant_category: form.merchant_category,
        location: form.location.trim() || 'Unspecified Location',
        tx_velocity_5m: numVelocity,
        card_present: Boolean(form.card_present),
        international_transaction: Boolean(form.international_transaction),
      }

      const { data } = await predictionApi.predict(payload)
      setResult(data)
    } catch (err) {
      setError(
        err.response?.status === 503
          ? 'ML models are being initialized. Please ensure training is completed.'
          : err.response?.data?.detail || 'Prediction failed. Please verify the transaction fields.'
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
      link.download = `fraud_audit_report_${result.prediction_id}.pdf`
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch {
      setError('Failed to generate PDF audit report.')
    } finally {
      setDownloading(false)
    }
  }

  const maxAbsShap = result
    ? Math.max(
        ...(result.top_fraud_contributors || []).map((c) => Math.abs(c.shap_value)),
        ...(result.top_legitimate_contributors || []).map((c) => Math.abs(c.shap_value)),
        0.0001
      )
    : 1

  return (
    <div>
      <div className="page-header">
        <h1>Predict Transaction Risk</h1>
        <p>
          Enter transaction parameters to evaluate fraud probability, composite risk score, and feature-level explanations.
        </p>
      </div>

      <ErrorBanner message={error} />

      {/* Transaction Input Form */}
      <form onSubmit={handleSubmit} className="card" style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18, borderBottom: '1px solid var(--color-border)', paddingBottom: 12 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 14, color: 'var(--color-text)', textTransform: 'none', letterSpacing: 0, fontWeight: 600 }}>
              Transaction Parameters
            </h3>
            <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>
              Fill in the transaction details below to run inference across all trained models.
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}>Currency:</span>
            <button
              type="button"
              onClick={() => setCurrency('$')}
              className="btn btn-secondary"
              style={{
                padding: '3px 8px',
                fontSize: 12,
                background: currency === '$' ? 'var(--color-primary)' : 'var(--color-surface-2)',
                color: currency === '$' ? '#ffffff' : 'var(--color-text-muted)',
                borderColor: currency === '$' ? 'var(--color-primary)' : 'var(--color-border)',
              }}
            >
              USD ($)
            </button>
            <button
              type="button"
              onClick={() => setCurrency('₹')}
              className="btn btn-secondary"
              style={{
                padding: '3px 8px',
                fontSize: 12,
                background: currency === '₹' ? 'var(--color-primary)' : 'var(--color-surface-2)',
                color: currency === '₹' ? '#ffffff' : 'var(--color-text-muted)',
                borderColor: currency === '₹' ? 'var(--color-primary)' : 'var(--color-border)',
              }}
            >
              INR (₹)
            </button>
          </div>
        </div>

        {/* Form Inputs Grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
            gap: 16,
          }}
        >
          {/* 1. Transaction Amount */}
          <div className="field" style={{ margin: 0 }}>
            <label>
              Transaction Amount ({currency}) <span style={{ color: 'var(--color-danger)' }}>*</span>
            </label>
            <div style={{ position: 'relative' }}>
              <span style={{ position: 'absolute', left: 10, top: 9, color: 'var(--color-text-muted)', fontWeight: 600 }}>
                {currency}
              </span>
              <input
                type="number"
                step="0.01"
                min="0.01"
                required
                placeholder="e.g. 149.50"
                value={form.amount}
                onChange={(e) => handleChange('amount', e.target.value)}
                style={{ paddingLeft: 26, fontWeight: 600 }}
              />
            </div>
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Required monetary value
            </div>
          </div>

          {/* 2. Date & Time */}
          <div className="field" style={{ margin: 0 }}>
            <label>
              Transaction Timestamp <span style={{ color: 'var(--color-danger)' }}>*</span>
            </label>
            <input
              type="datetime-local"
              required
              value={form.timestamp}
              onChange={(e) => handleChange('timestamp', e.target.value)}
            />
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Time of authorization
            </div>
          </div>

          {/* 3. Transaction Type */}
          <div className="field" style={{ margin: 0 }}>
            <label>Payment Channel</label>
            <select
              value={form.transaction_type}
              onChange={(e) => handleChange('transaction_type', e.target.value)}
            >
              {TRANSACTION_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Payment initiation method
            </div>
          </div>

          {/* 4. Merchant Category */}
          <div className="field" style={{ margin: 0 }}>
            <label>Merchant Category</label>
            <select
              value={form.merchant_category}
              onChange={(e) => handleChange('merchant_category', e.target.value)}
            >
              {MERCHANT_CATEGORIES.map((c) => (
                <option key={c.value} value={c.value}>
                  {c.label}
                </option>
              ))}
            </select>
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Merchant business segment
            </div>
          </div>

          {/* 5. Location */}
          <div className="field" style={{ margin: 0 }}>
            <label>Transaction Location</label>
            <input
              type="text"
              placeholder="e.g. Mumbai, India or Online"
              value={form.location}
              onChange={(e) => handleChange('location', e.target.value)}
            />
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Merchant or terminal city/country
            </div>
          </div>

          {/* 6. Recent Transaction Activity */}
          <div className="field" style={{ margin: 0 }}>
            <label>RECENT TRANSACTION ACTIVITY</label>
            <input
              type="number"
              min="0"
              step="1"
              placeholder="e.g. 3"
              value={form.tx_velocity_5m}
              onChange={(e) => {
                const val = e.target.value
                if (val === '' || /^\d+$/.test(val)) {
                  handleChange('tx_velocity_5m', val)
                }
              }}
              required
            />
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Number of transactions made in the previous 5 minutes
            </div>
          </div>

          {/* 7. Card Present */}
          <div className="field" style={{ margin: 0 }}>
            <label>Physical Card Present</label>
            <select
              value={form.card_present ? 'true' : 'false'}
              onChange={(e) => handleChange('card_present', e.target.value === 'true')}
            >
              <option value="false">No (Card-Not-Present / Online)</option>
              <option value="true">Yes (Card Chip / Swipe / Tap)</option>
            </select>
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Physical authentication status
            </div>
          </div>

          {/* 8. International Transaction */}
          <div className="field" style={{ margin: 0 }}>
            <label>Cross-Border Transaction</label>
            <select
              value={form.international_transaction ? 'true' : 'false'}
              onChange={(e) => handleChange('international_transaction', e.target.value === 'true')}
            >
              <option value="false">Domestic Transaction</option>
              <option value="true">International / Foreign Merchant</option>
            </select>
            <div style={{ fontSize: 11, color: 'var(--color-text-muted)', marginTop: 4 }}>
              Clearing region
            </div>
          </div>
        </div>

        <div style={{ marginTop: 20 }}>
          <button
            className="btn btn-primary"
            type="submit"
            disabled={loading}
            style={{ padding: '10px 24px', fontSize: 14 }}
          >
            {loading ? 'Evaluating Transaction...' : 'Analyze Transaction Risk'}
          </button>
        </div>
      </form>

      {/* Loading state indicator */}
      {loading && (
        <div className="card" style={{ padding: 28, textAlign: 'center', marginBottom: 24 }}>
          <div className="spinner" style={{ marginBottom: 12 }} />
          <h3 style={{ fontSize: 14, color: 'var(--color-text)', margin: '0 0 4px', textTransform: 'none' }}>
            Evaluating Transaction Across Ensemble Models
          </h3>
          <p style={{ fontSize: 12, color: 'var(--color-text-muted)', margin: 0 }}>
            Running inference through Random Forest, AdaBoost, XGBoost, LightGBM, and CatBoost...
          </p>
        </div>
      )}

      {/* Prediction Results Display */}
      {result && !loading && (
        <div style={{ marginTop: 24 }}>
          {/* 1. Overall Verdict Hero */}
          <div className={`result-hero ${result.is_fraud ? 'fraud' : 'legit'}`}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
              <div>
                <div style={{ fontSize: 12, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-text-muted)', marginBottom: 4, fontWeight: 600 }}>
                  TrustCheck Final Assessment
                </div>
                <div className={`verdict ${result.is_fraud ? 'fraud-text' : 'legit-text'}`}>
                  {result.is_fraud ? 'HIGH RISK / FRAUD DETECTED' : 'TRANSACTION EVALUATED AS LEGITIMATE'}
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 8, flexWrap: 'wrap' }}>
                  <FraudBadge isFraud={result.is_fraud} />
                  <RiskLevelBadge level={result.risk_level} />
                  <span style={{ fontSize: 12, color: 'var(--color-text-muted)', marginLeft: 4 }}>
                    Base ML Model: <strong>{result.model_used}</strong>
                  </span>
                </div>
              </div>

              <button
                className="btn btn-secondary"
                style={{ padding: '8px 16px', fontSize: 13 }}
                onClick={handleDownload}
                disabled={downloading}
              >
                {downloading ? 'Generating Report...' : 'Download PDF Audit Report'}
              </button>
            </div>

            <div className="result-stats">
              <div className="item">
                <div className="label">TrustCheck Final Assessment</div>
                <div className="value" style={{ color: result.is_fraud ? 'var(--color-danger)' : 'var(--color-success)' }}>
                  {result.prediction || (result.is_fraud ? 'FRAUD' : 'LEGITIMATE')}
                </div>
              </div>
              <div className="item">
                <div className="label">TrustCheck Risk Score</div>
                <div className="value" style={{ color: (result.risk_score ?? 0) >= 70 ? 'var(--color-danger)' : (result.risk_score ?? 0) >= 30 ? 'var(--color-warning)' : 'var(--color-success)' }}>
                  {(result.risk_score ?? 0).toFixed(1)} <span style={{ fontSize: 12, fontWeight: 400, color: 'var(--color-text-muted)' }}>/ 100</span>
                </div>
              </div>
              <div className="item">
                <div className="label">Risk Classification</div>
                <div className="value">{result.risk_level || 'LOW'} RISK</div>
              </div>
              <div className="item">
                <div className="label">ML Model Probability</div>
                <div className="value" style={{ color: (result.ml_fraud_probability ?? result.fraud_probability ?? 0) >= 0.5 ? 'var(--color-danger)' : 'var(--color-text-muted)' }}>
                  {(((result.ml_fraud_probability ?? result.fraud_probability) ?? 0) * 100).toFixed(2)}%
                </div>
              </div>
            </div>

            {/* Assessment Differentiation Callout Banner */}
            {result.ml_prediction && result.prediction && (
              <div className={`diff-alert ${result.ml_prediction !== result.prediction ? 'divergent' : 'concordant'}`}>
                {result.ml_prediction !== result.prediction ? (
                  <div>
                    <strong>⚠️ Model &amp; Decision Layer Differentiation:</strong>
                    <div style={{ marginTop: 4, fontSize: 12.5 }}>
                      The offline machine learning classifier evaluated base statistical features at{' '}
                      <strong>{(((result.ml_fraud_probability ?? result.fraud_probability) ?? 0) * 100).toFixed(2)}%</strong> fraud probability (<strong>{result.ml_prediction}</strong>).
                      In contrast, the separate TrustCheck Transaction Risk Decision Layer identified elevated multi-signal risk across user-entered transaction parameters (amount, hour, velocity, channel, authentication, and cross-border indicators), elevating the final risk score to{' '}
                      <strong>{(result.risk_score ?? 0).toFixed(1)}/100</strong> and final assessment to <strong>{result.prediction}</strong>.
                    </div>
                  </div>
                ) : (
                  <div>
                    <strong>✓ Concordant Assessment:</strong>
                    <div style={{ marginTop: 4, fontSize: 12.5 }}>
                      Both the machine learning model ({(((result.ml_fraud_probability ?? result.fraud_probability) ?? 0) * 100).toFixed(2)}% fraud probability) and the TrustCheck Transaction Risk Decision Layer concordantly assessed this transaction as <strong>{result.prediction}</strong> with <strong>{result.risk_level} RISK</strong>.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* 2. Side-by-Side Dual Assessment Panels */}
          <div className="assessment-grid">
            {/* Panel A: ML Model Output */}
            <div className="assessment-panel highlight-neutral">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <h4 style={{ margin: 0, fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-text)' }}>
                    A. Existing ML Model Output
                  </h4>
                  <span style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
                    Genuine classifier inference (unaltered)
                  </span>
                </div>
                <span className={`badge ${(result.ml_is_fraud ?? ((result.ml_fraud_probability ?? result.fraud_probability) >= 0.5)) ? 'badge-fraud' : 'badge-legit'}`}>
                  {result.ml_prediction || 'LEGITIMATE'}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 12.5, marginBottom: 14 }}>
                <div style={{ background: 'var(--color-surface-2)', padding: '8px 10px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11 }}>Fraud Probability</div>
                  <div style={{ fontWeight: 700, fontSize: 15, marginTop: 2, color: 'var(--color-text)' }}>
                    {(((result.ml_fraud_probability ?? result.fraud_probability) ?? 0) * 100).toFixed(2)}%
                  </div>
                </div>
                <div style={{ background: 'var(--color-surface-2)', padding: '8px 10px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11 }}>Legitimate Probability</div>
                  <div style={{ fontWeight: 700, fontSize: 15, marginTop: 2, color: 'var(--color-success)' }}>
                    {(((result.ml_legitimate_probability ?? result.legitimate_probability) ?? 1) * 100).toFixed(2)}%
                  </div>
                </div>
              </div>

              <div style={{ fontSize: 11.5, color: 'var(--color-text-muted)', lineHeight: 1.5, borderTop: '1px solid var(--color-border)', paddingTop: 10 }}>
                <div>• <strong>Model:</strong> {result.model_used}</div>
                <div>• <strong>Feature Space:</strong> PCA components V1–V28, Scaled Time &amp; Amount</div>
                <div>• <strong>Contribution to Final:</strong> 20% ensemble weighting</div>
              </div>
            </div>

            {/* Panel B: TrustCheck Final Assessment */}
            <div className={`assessment-panel ${result.is_fraud ? 'highlight-fraud' : 'highlight-legit'}`}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <h4 style={{ margin: 0, fontSize: 13, textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--color-text)' }}>
                    B. TrustCheck Final Assessment
                  </h4>
                  <span style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
                    Deterministic transaction risk decision layer
                  </span>
                </div>
                <span className={`badge ${result.is_fraud ? 'badge-fraud' : 'badge-legit'}`}>
                  {result.prediction || (result.is_fraud ? 'FRAUD' : 'LEGITIMATE')}
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, fontSize: 12.5, marginBottom: 14 }}>
                <div style={{ background: 'var(--color-surface-2)', padding: '8px 10px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11 }}>TrustCheck Risk Score</div>
                  <div style={{ fontWeight: 700, fontSize: 15, marginTop: 2, color: (result.risk_score ?? 0) >= 70 ? 'var(--color-danger)' : (result.risk_score ?? 0) >= 30 ? 'var(--color-warning)' : 'var(--color-success)' }}>
                    {(result.risk_score ?? 0).toFixed(1)} <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--color-text-muted)' }}>/ 100</span>
                  </div>
                </div>
                <div style={{ background: 'var(--color-surface-2)', padding: '8px 10px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11 }}>Risk Level</div>
                  <div style={{ fontWeight: 700, fontSize: 15, marginTop: 2 }}>
                    {result.risk_level || 'LOW'} RISK
                  </div>
                </div>
              </div>

              <div style={{ fontSize: 11.5, color: 'var(--color-text-muted)', lineHeight: 1.5, borderTop: '1px solid var(--color-border)', paddingTop: 10 }}>
                <div>• <strong>Decision Architecture:</strong> 20% Base ML + 80% Multi-Factor Transaction Risk Layer</div>
                <div>• <strong>Fraud Cutoff:</strong> Risk Score &ge; 50.0 (or ML Fraud Probability &ge; 50%)</div>
                <div>• <strong>Explainability:</strong> Deterministic signal evaluation across user inputs</div>
              </div>
            </div>
          </div>

          {/* 3. Transaction Risk Decision Layer Factors Breakdown */}
          {result.risk_factors && result.risk_factors.length > 0 && (
            <div className="card" style={{ marginBottom: 20 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <div>
                  <h3 style={{ margin: 0, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
                    Transaction Risk Decision Layer — Input Signals Analysis
                  </h3>
                  <p style={{ margin: '4px 0 0', fontSize: 12, color: 'var(--color-text-muted)' }}>
                    Explainable risk factors evaluated directly from the user-entered transaction parameters:
                  </p>
                </div>
                <span style={{ fontSize: 11, color: 'var(--color-text-muted)' }}>
                  Layer Score: <strong>{(result.transaction_risk_score ?? result.risk_score)?.toFixed(1)}/100</strong>
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
                {result.risk_factors.map((factor) => {
                  const isSuspicious = factor.direction === 'suspicious'
                  return (
                    <div
                      key={factor.signal}
                      className={`risk-factor-card ${isSuspicious ? 'suspicious' : 'normal'}`}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontWeight: 600, fontSize: 12.5, color: 'var(--color-text)' }}>
                          {factor.label}
                        </div>
                        <span className={`badge ${isSuspicious ? 'badge-fraud' : 'badge-legit'}`}>
                          {isSuspicious ? 'Elevated Signal' : 'Normal Range'}
                        </span>
                      </div>

                      <div style={{ fontSize: 11.5, color: 'var(--color-text)', lineHeight: 1.4 }}>
                        {factor.description}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                        <div style={{ flex: 1, height: 4, background: 'var(--color-surface-3)', borderRadius: 2, overflow: 'hidden' }}>
                          <div
                            style={{
                              width: `${Math.min(100, Math.max(2, factor.risk_percent))}%`,
                              height: '100%',
                              background: isSuspicious ? 'var(--color-danger)' : 'var(--color-success)',
                            }}
                          />
                        </div>
                        <span style={{ fontSize: 11, fontVariantNumeric: 'tabular-nums', fontWeight: 600, color: isSuspicious ? 'var(--color-danger)' : 'var(--color-text-muted)' }}>
                          {factor.risk_percent}%
                        </span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* 4. Individual Model Predictions for this Transaction */}
          {result.model_predictions && Object.keys(result.model_predictions).length > 0 && (
            <div className="card" style={{ marginBottom: 20 }}>
              <h3 style={{ marginBottom: 4, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
                Ensemble Model Predictions
              </h3>
              <p style={{ margin: '0 0 12px', fontSize: 12, color: 'var(--color-text-muted)' }}>
                Independent prediction output from each model for this specific transaction:
              </p>

              <div style={{ overflowX: 'auto' }}>
                <table>
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Prediction</th>
                      <th>Fraud Probability</th>
                      <th>Legitimate Probability</th>
                      <th>Risk Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.model_predictions || {}).map(([key, m]) => {
                      const isBest = (result.model_used || '').toLowerCase().includes(key.toLowerCase()) ||
                                     (m?.model_name || '').toLowerCase() === (result.model_used || '').toLowerCase()
                      return (
                        <tr key={key} className={isBest ? 'best-row' : ''}>
                          <td style={{ fontWeight: isBest ? 600 : 500 }}>
                            {m?.model_name || key}
                            {isBest && <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--color-primary)', fontWeight: 600 }}>(Selected)</span>}
                          </td>
                          <td>
                            <span className={`badge ${m?.is_fraud ? 'badge-fraud' : 'badge-legit'}`}>
                              {m?.prediction || 'LEGITIMATE'}
                            </span>
                          </td>
                          <td style={{ fontWeight: 600, color: m?.is_fraud ? 'var(--color-danger)' : 'var(--color-text)', fontVariantNumeric: 'tabular-nums' }}>
                            {((m?.fraud_probability ?? 0) * 100).toFixed(2)}%
                          </td>
                          <td style={{ fontWeight: 600, color: !m?.is_fraud ? 'var(--color-success)' : 'var(--color-text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                            {((m?.legitimate_probability ?? 0) * 100).toFixed(2)}%
                          </td>
                          <td>
                            <span style={{ fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{(m?.risk_score ?? 0).toFixed(0)}</span>
                            <span style={{ fontSize: 12, color: 'var(--color-text-muted)' }}> / 100</span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 5. SHAP Explanation ("Why did the system give this result?") */}
          <div className="card" style={{ marginBottom: 20 }}>
            <h3 style={{ marginBottom: 4, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
              Machine Learning Feature Attribution (SHAP)
            </h3>
            <p style={{ margin: '0 0 14px', fontSize: 12, color: 'var(--color-text-muted)' }}>
              Feature-level SHAP attributions explaining the underlying machine learning model prediction:
            </p>

            {/* Natural language summary box */}
            {result.explanation && result.explanation.length > 0 && (
              <div
                style={{
                  background: result.is_fraud ? 'var(--color-danger-bg)' : 'var(--color-success-bg)',
                  border: `1px solid ${result.is_fraud ? 'var(--color-danger-border)' : 'var(--color-success-border)'}`,
                  borderRadius: 'var(--radius-sm)',
                  padding: '14px 18px',
                  marginBottom: 14,
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 600, color: result.is_fraud ? 'var(--color-danger)' : 'var(--color-success)', marginBottom: 6 }}>
                  {result.is_fraud
                    ? 'Key risk factors identified:'
                    : 'Factors supporting normal transaction behavior:'}
                </div>
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.6, color: 'var(--color-text)' }}>
                  {result.explanation.map((sentence, idx) => (
                    <li key={idx}>{sentence}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Toggle detailed SHAP attribution bars */}
            <div style={{ marginTop: 8 }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowShapBars(!showShapBars)}
                style={{ fontSize: 12, padding: '5px 12px' }}
              >
                {showShapBars ? 'Hide Feature Attribution Details' : 'View Feature Attribution Details (SHAP)'}
              </button>
            </div>

            {showShapBars && (
              <div style={{ marginTop: 16 }}>
                {/* Factors Increasing Fraud Risk */}
                <div style={{ marginBottom: 18 }}>
                  <div style={{ fontSize: 12, color: 'var(--color-danger)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Factors Increasing Fraud Risk
                  </div>
                  {(!result.top_fraud_contributors || result.top_fraud_contributors.length === 0) ? (
                    <div style={{ color: 'var(--color-text-muted)', fontSize: 12, padding: '8px 12px', background: 'var(--color-surface-2)', borderRadius: 'var(--radius-sm)' }}>
                      No significant fraud-risk contributors identified for this transaction.
                    </div>
                  ) : (
                    result.top_fraud_contributors.map((c) => (
                      <ShapBar
                        key={c.feature}
                        feature={c.feature}
                        label={c.label}
                        shapValue={c.shap_value}
                        maxAbs={maxAbsShap}
                        direction="fraud"
                        description={c.description}
                      />
                    ))
                  )}
                </div>

                {/* Factors Supporting Legitimacy */}
                <div>
                  <div style={{ fontSize: 12, color: 'var(--color-success)', fontWeight: 600, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    Factors Supporting Transaction Legitimacy
                  </div>
                  {(!result.top_legitimate_contributors || result.top_legitimate_contributors.length === 0) ? (
                    <div style={{ color: 'var(--color-text-muted)', fontSize: 12, padding: '8px 12px', background: 'var(--color-surface-2)', borderRadius: 'var(--radius-sm)' }}>
                      No significant legitimacy contributors recorded.
                    </div>
                  ) : (
                    result.top_legitimate_contributors.map((c) => (
                      <ShapBar
                        key={c.feature}
                        feature={c.feature}
                        label={c.label}
                        shapValue={c.shap_value}
                        maxAbs={maxAbsShap}
                        direction="legit"
                        description={c.description}
                      />
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* 4. Important Transaction Details */}
          {result.transaction_summary && (
            <div className="card">
              <h3 style={{ marginBottom: 12, textTransform: 'none', fontSize: 14, color: 'var(--color-text)', fontWeight: 600 }}>
                Transaction Summary
              </h3>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: 12,
                  fontSize: 13,
                }}
              >
                <div style={{ background: 'var(--color-surface-2)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Amount</div>
                  <div style={{ fontWeight: 600, marginTop: 2 }}>{currency} {result.transaction_summary.amount?.toFixed(2)}</div>
                </div>
                <div style={{ background: 'var(--color-surface-2)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Payment Channel</div>
                  <div style={{ fontWeight: 600, marginTop: 2, textTransform: 'capitalize' }}>
                    {(result.transaction_summary.transaction_type || '—').replace(/_/g, ' ')}
                  </div>
                </div>
                <div style={{ background: 'var(--color-surface-2)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Merchant Category</div>
                  <div style={{ fontWeight: 600, marginTop: 2, textTransform: 'capitalize' }}>
                    {(result.transaction_summary.merchant_category || '—').replace(/_/g, ' ')}
                  </div>
                </div>
                <div style={{ background: 'var(--color-surface-2)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Location</div>
                  <div style={{ fontWeight: 600, marginTop: 2 }}>{result.transaction_summary.location || '—'}</div>
                </div>
                <div style={{ background: 'var(--color-surface-2)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Recent Activity</div>
                  <div style={{ fontWeight: 600, marginTop: 2 }}>
                    {result.transaction_summary.tx_velocity_5m ?? 0}
                  </div>
                </div>
                <div style={{ background: 'var(--color-surface-2)', padding: '10px 12px', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ color: 'var(--color-text-muted)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Authentication / Clearance</div>
                  <div style={{ fontWeight: 600, marginTop: 2 }}>
                    {result.transaction_summary.card_present ? 'Card Present' : 'Card Not Present'} · {result.transaction_summary.international_transaction ? 'International' : 'Domestic'}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
