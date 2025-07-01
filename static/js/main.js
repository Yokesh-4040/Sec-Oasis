// Main JavaScript file for SecOasis

// API Configuration
const API_BASE = window.location.origin;

// Utility Functions
class API {
    static async request(endpoint, options = {}) {
        const url = `${API_BASE}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Request failed');
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    }

    static async get(endpoint) {
        return this.request(endpoint);
    }

    static async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    static async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    static async delete(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }
}

// Notification System
class Notifications {
    static show(message, type = 'info') {
        const alertClass = type === 'error' ? 'danger' : type;
        const alertHtml = `
            <div class="alert alert-${alertClass} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Find or create alert container
        let container = document.querySelector('.alert-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'alert-container';
            document.querySelector('main').insertBefore(container, document.querySelector('main').firstChild);
        }
        
        container.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert');
            if (alert) {
                const bsAlert = new bootstrap.Alert(alert);
                bsAlert.close();
            }
        }, 5000);
    }

    static success(message) {
        this.show(message, 'success');
    }

    static error(message) {
        this.show(message, 'error');
    }

    static warning(message) {
        this.show(message, 'warning');
    }

    static info(message) {
        this.show(message, 'info');
    }
}

// Loading State Management
class LoadingManager {
    static show(element, text = 'Loading...') {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (element) {
            element.classList.add('loading');
            const originalContent = element.innerHTML;
            element.dataset.originalContent = originalContent;
            element.innerHTML = `
                <div class="text-center">
                    <i class="fas fa-spinner fa-spin me-2"></i>${text}
                </div>
            `;
        }
    }

    static hide(element) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        if (element) {
            element.classList.remove('loading');
            if (element.dataset.originalContent) {
                element.innerHTML = element.dataset.originalContent;
                delete element.dataset.originalContent;
            }
        }
    }
}

// Face Recognition Simulation (placeholder for actual FACEIO integration)
class FaceRecognition {
    static async simulateEnrollment(employeeName) {
        // Simulate camera access and face capture
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                // Generate a simulated facial ID
                const facialId = `face_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                resolve({
                    facialId: facialId,
                    message: `Face enrolled successfully for ${employeeName}`
                });
            }, 3000);
        });
    }

    static async simulateAuthentication() {
        // Simulate face recognition for authentication
        return new Promise((resolve, reject) => {
            setTimeout(() => {
                // For demo purposes, we'll randomly succeed or fail
                const success = Math.random() > 0.3; // 70% success rate
                
                if (success) {
                    // Return a simulated facial ID
                    const facialId = `face_demo_${Math.random().toString(36).substr(2, 9)}`;
                    resolve({
                        facialId: facialId,
                        message: 'Face recognized successfully'
                    });
                } else {
                    reject(new Error('Face not recognized'));
                }
            }, 2000);
        });
    }
}

// Form Utilities
class FormUtils {
    static validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    static validatePassword(password) {
        return password.length >= 6;
    }

    static serializeForm(form) {
        const formData = new FormData(form);
        const data = {};
        for (let [key, value] of formData.entries()) {
            data[key] = value;
        }
        return data;
    }

    static resetForm(form) {
        if (typeof form === 'string') {
            form = document.querySelector(form);
        }
        if (form) {
            form.reset();
        }
    }
}

// Date/Time Utilities
class DateUtils {
    static formatTime(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    static formatDate(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    static formatDateTime(date) {
        return `${this.formatDate(date)} ${this.formatTime(date)}`;
    }

    static isToday(date) {
        if (typeof date === 'string') {
            date = new Date(date);
        }
        const today = new Date();
        return date.toDateString() === today.toDateString();
    }
}

// Global event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Handle logout
    const logoutLinks = document.querySelectorAll('a[href="/logout"]');
    logoutLinks.forEach(link => {
        link.addEventListener('click', async function(e) {
            e.preventDefault();
            try {
                await API.get('/api/auth/logout');
                window.location.href = '/login';
            } catch (error) {
                console.error('Logout error:', error);
                window.location.href = '/login';
            }
        });
    });
});

// Export for use in other modules
window.API = API;
window.Notifications = Notifications;
window.LoadingManager = LoadingManager;
window.FaceRecognition = FaceRecognition;
window.FormUtils = FormUtils;
window.DateUtils = DateUtils; 