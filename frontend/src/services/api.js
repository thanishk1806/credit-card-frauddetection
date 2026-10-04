import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({ baseURL: API_BASE_URL })

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('access_token')
      if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  }
)

export const authApi = {
  register: (data) => api.post('/auth/register', data),
  login: (data) => api.post('/auth/login', data),
  me: () => api.get('/auth/me'),
}

export const dashboardApi = {
  summary: () => api.get('/dashboard/summary'),
  liveStats: () => api.get('/dashboard/live-stats'),
  modelPerformance: () => api.get('/models/performance'),
  bestModel: () => api.get('/models/best'),
}

export const predictionApi = {
  predict: (transaction) => api.post('/predict', transaction),
  history: (limit = 50) => api.get(`/predictions?limit=${limit}`),
  getById: (id) => api.get(`/predictions/${id}`),
}

export const reportApi = {
  downloadUrl: (predictionId) => `${API_BASE_URL}/reports/${predictionId}/download`,
  download: (predictionId) =>
    api.get(`/reports/${predictionId}/download`, { responseType: 'blob' }),
}

export default api
