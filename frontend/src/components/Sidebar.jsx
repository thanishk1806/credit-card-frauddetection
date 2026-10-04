import React from 'react'
import { NavLink } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: '📊' },
  { to: '/predict', label: 'Predict Transaction', icon: '🔍' },
  { to: '/models', label: 'Model Performance', icon: '📈' },
  { to: '/reports', label: 'Reports', icon: '📄' },
  { to: '/profile', label: 'Profile', icon: '👤' },
]

export default function Sidebar() {
  const { user, logout } = useAuth()

  return (
    <aside className="sidebar">
      <div>
        <div className="sidebar-logo">Trust<span>Check</span></div>
        <div className="sidebar-sub">Smart Transaction Risk Analysis</div>
      </div>

      <nav>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}
          >
            <span>{item.icon}</span> {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        {user && (
          <div className="user-chip">
            Signed in as <strong>{user.username}</strong>
          </div>
        )}
        <button className="btn btn-secondary" style={{ width: '100%' }} onClick={logout}>
          Logout
        </button>
      </div>
    </aside>
  )
}
