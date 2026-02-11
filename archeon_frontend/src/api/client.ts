import axios from 'axios';

// API Configuration
// VITE_API_URL should be the base server URL (e.g. http://localhost:9005)
// But currently it might be /v1.
// Let's standardize: BASE_URL is the server root.
export const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:9005';
export const API_URL = `${BASE_URL}/v1`;

export const apiClient = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});
