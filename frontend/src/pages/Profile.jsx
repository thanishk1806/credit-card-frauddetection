import React from 'react'
import { useAuth } from '../context/AuthContext'

export default function Profile() {
  const { user, logout } = useAuth()

  if (!user) return null

  return (
    <div>
      <div className="page-header">
        <h1>User Profile</h1>
        <p>Account credentials and session details for the logged-in analyst.</p>
      </div>

      <div className="card" style={{ maxWidth: 480 }}>
        <div className="field">
          <label>Analyst Username</label>
          <input value={user.username} disabled />
        </div>
        <div className="field">
          <label>Registered Email</label>
          <input value={user.email} disabled />
        </div>
        <div className="field">
          <label>Account Created</label>
          <input value={new Date(user.created_at).toLocaleDateString('en-IN', { dateStyle: 'long' })} disabled />
        </div>
        <div style={{ marginTop: 20 }}>
          <button className="btn btn-secondary" onClick={logout}>
            Sign Out
          </button>
        </div>
      </div>
    </div>
  )
}
