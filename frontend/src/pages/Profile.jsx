import React from 'react'
import { useAuth } from '../context/AuthContext'

export default function Profile() {
  const { user, logout } = useAuth()

  if (!user) return null

  return (
    <div>
      <div className="page-header">
        <h1>Profile</h1>
        <p>Your account details.</p>
      </div>

      <div className="card" style={{ maxWidth: 480 }}>
        <div className="field">
          <label>Username</label>
          <input value={user.username} disabled />
        </div>
        <div className="field">
          <label>Email</label>
          <input value={user.email} disabled />
        </div>
        <div className="field">
          <label>Member Since</label>
          <input value={new Date(user.created_at).toLocaleDateString()} disabled />
        </div>
        <button className="btn btn-secondary" onClick={logout}>Logout</button>
      </div>
    </div>
  )
}
