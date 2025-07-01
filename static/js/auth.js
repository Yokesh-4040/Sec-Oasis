// Authentication functionality for SecOasis

class AuthManager {
    constructor() {
        this.init();
    }

    init() {
        this.initEventListeners();
    }

    initEventListeners() {
        // Login form
        const loginForm = document.getElementById('loginForm');
        if (loginForm) {
            loginForm.addEventListener('submit', (e) => this.handleLogin(e));
        }

        // Face login button
        const faceLoginBtn = document.getElementById('faceLoginBtn');
        if (faceLoginBtn) {
            faceLoginBtn.addEventListener('click', () => this.showFaceLoginModal());
        }

        // Face authentication button
        const authenticateBtn = document.getElementById('authenticateBtn');
        if (authenticateBtn) {
            authenticateBtn.addEventListener('click', () => this.handleFaceAuthentication());
        }
    }

    async handleLogin(e) {
        e.preventDefault();

        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        // Basic validation
        if (!email || !password) {
            Notifications.error('Please enter both email and password');
            return;
        }

        if (!FormUtils.validateEmail(email)) {
            Notifications.error('Please enter a valid email address');
            return;
        }

        try {
            LoadingManager.show('#loginForm button[type="submit"]', 'Logging in...');

            const result = await API.post('/api/auth/login', {
                email: email,
                password: password
            });

            Notifications.success('Login successful! Redirecting...');
            
            // Redirect to dashboard after short delay
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1000);

        } catch (error) {
            Notifications.error(error.message || 'Login failed');
        } finally {
            LoadingManager.hide('#loginForm button[type="submit"]');
        }
    }

    showFaceLoginModal() {
        const modal = new bootstrap.Modal(document.getElementById('faceLoginModal'));
        modal.show();

        // Start face recognition simulation after modal is shown
        setTimeout(() => {
            this.simulateFaceRecognition();
        }, 500);
    }

    async simulateFaceRecognition() {
        const cameraContainer = document.querySelector('#faceLoginModal .camera-placeholder');
        
        try {
            // Show scanning state
            cameraContainer.innerHTML = `
                <i class="fas fa-camera fa-3x text-primary mb-3"></i>
                <p class="text-primary">Scanning face...</p>
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Processing...</span>
                </div>
            `;

            // Simulate face recognition process
            const result = await FaceRecognition.simulateAuthentication();
            
            // Show success state
            cameraContainer.innerHTML = `
                <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                <p class="text-success">Face recognized!</p>
                <small class="text-muted">Facial ID: ${result.facialId}</small>
            `;

            // Auto-proceed with authentication
            setTimeout(() => {
                this.handleFaceAuthentication(result.facialId);
            }, 1500);

        } catch (error) {
            // Show error state
            cameraContainer.innerHTML = `
                <i class="fas fa-times-circle fa-3x text-danger mb-3"></i>
                <p class="text-danger">Face not recognized</p>
                <small class="text-muted">Please try again or use email/password</small>
            `;
            
            Notifications.error('Face not recognized. Please try manual login.');
        }
    }

    async handleFaceAuthentication(facialId = null) {
        if (!facialId) {
            // If no facial ID provided, try to simulate recognition again
            this.simulateFaceRecognition();
            return;
        }

        try {
            LoadingManager.show('#authenticateBtn', 'Authenticating...');

            // For demo purposes, we'll simulate finding an employee with this facial ID
            // In a real implementation, this would query the backend
            const demoEmployee = {
                id: 1,
                name: 'Demo User',
                email: 'demo@secoasis.com',
                facial_id: facialId
            };

            Notifications.success(`Welcome back, ${demoEmployee.name}!`);
            
            // Close modal and redirect
            const modal = bootstrap.Modal.getInstance(document.getElementById('faceLoginModal'));
            modal.hide();
            
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1000);

        } catch (error) {
            Notifications.error(error.message || 'Face authentication failed');
        } finally {
            LoadingManager.hide('#authenticateBtn');
        }
    }
}

// Initialize authentication manager when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('loginForm') || document.getElementById('faceLoginBtn')) {
        new AuthManager();
    }
}); 