import React from 'react'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Unhandled React Error:', error, errorInfo)
    this.setState({ errorInfo })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 32, maxWidth: 640, margin: '60px auto', background: 'var(--color-surface)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-danger-border)' }}>
          <h2 style={{ color: 'var(--color-danger)', fontSize: 18, margin: '0 0 12px' }}>An Error Occurred</h2>
          <p style={{ color: 'var(--color-text-muted)', fontSize: 13, marginBottom: 16 }}>
            {this.state.error?.message || 'Something went wrong while rendering this view.'}
          </p>
          <button
            className="btn btn-primary"
            onClick={() => {
              this.setState({ hasError: false, error: null, errorInfo: null })
              window.location.reload()
            }}
          >
            Reload Page
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
