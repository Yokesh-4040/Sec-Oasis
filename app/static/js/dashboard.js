// Dashboard functionality for SecOasis

class Dashboard {
    constructor() {
        this.init();
    }

    init() {
        this.loadDashboardStats();
        this.loadRecentActivity();
        this.initEventListeners();
        this.loadEmployeesList();
    }

    async loadDashboardStats() {
        try {
            // Load employees count
            const employees = await API.get('/api/employees');
            document.getElementById('totalEmployees').textContent = employees.length;

            // Load attendance stats for today
            const today = new Date().toISOString().split('T')[0];
            const attendanceLogs = await API.get(`/api/attendance/logs?start_date=${today}&end_date=${today}`);
            
            // Calculate stats
            const presentToday = attendanceLogs.filter(log => log.check_in && DateUtils.isToday(log.check_in)).length;
            const stillWorking = attendanceLogs.filter(log => log.check_in && !log.check_out && DateUtils.isToday(log.check_in)).length;
            
            // Late arrivals (assuming 9 AM is start time)
            const lateArrivals = attendanceLogs.filter(log => {
                if (!log.check_in) return false;
                const checkInTime = new Date(log.check_in);
                return checkInTime.getHours() > 9 || (checkInTime.getHours() === 9 && checkInTime.getMinutes() > 0);
            }).length;

            document.getElementById('presentToday').textContent = presentToday;
            document.getElementById('stillWorking').textContent = stillWorking;
            document.getElementById('lateArrivals').textContent = lateArrivals;

        } catch (error) {
            console.error('Error loading dashboard stats:', error);
            Notifications.error('Failed to load dashboard statistics');
        }
    }

    async loadRecentActivity() {
        try {
            const recentLogs = await API.get('/api/attendance/logs?limit=10');
            const recentActivityEl = document.getElementById('recentActivity');
            
            if (recentLogs.length === 0) {
                recentActivityEl.innerHTML = '<p class="text-muted text-center">No recent activity</p>';
                return;
            }

            const activityHtml = recentLogs.map(log => {
                const checkInTime = log.check_in ? DateUtils.formatTime(log.check_in) : 'N/A';
                const checkOutTime = log.check_out ? DateUtils.formatTime(log.check_out) : 'Still working';
                const date = DateUtils.formatDate(log.date);
                
                return `
                    <div class="d-flex align-items-center justify-content-between py-2 border-bottom">
                        <div>
                            <strong>Employee #${log.employee_id}</strong><br>
                            <small class="text-muted">${date}</small>
                        </div>
                        <div class="text-end">
                            <small class="text-success">In: ${checkInTime}</small><br>
                            <small class="text-warning">Out: ${checkOutTime}</small>
                        </div>
                    </div>
                `;
            }).join('');

            recentActivityEl.innerHTML = activityHtml;

        } catch (error) {
            console.error('Error loading recent activity:', error);
            document.getElementById('recentActivity').innerHTML = 
                '<p class="text-danger text-center">Failed to load recent activity</p>';
        }
    }

    async loadEmployeesList() {
        try {
            const employees = await API.get('/api/employees');
            const employeeSelect = document.getElementById('employeeSelect');
            
            if (employeeSelect) {
                employeeSelect.innerHTML = '<option value="">Select an employee...</option>';
                employees.forEach(employee => {
                    const option = document.createElement('option');
                    option.value = employee.id;
                    option.textContent = `${employee.name} (${employee.position})`;
                    employeeSelect.appendChild(option);
                });
            }
        } catch (error) {
            console.error('Error loading employees list:', error);
        }
    }

    initEventListeners() {
        // Quick action buttons
        document.getElementById('checkInBtn')?.addEventListener('click', () => {
            this.showAttendanceModal('check-in');
        });

        document.getElementById('checkOutBtn')?.addEventListener('click', () => {
            this.showAttendanceModal('check-out');
        });

        document.getElementById('addEmployeeBtn')?.addEventListener('click', () => {
            this.showAddEmployeeModal();
        });

        document.getElementById('viewReportsBtn')?.addEventListener('click', () => {
            window.location.href = '/reports';
        });

        // Attendance modal
        document.getElementById('processAttendanceBtn')?.addEventListener('click', () => {
            this.processAttendance();
        });

        // Add employee form
        document.getElementById('addEmployeeForm')?.addEventListener('submit', (e) => {
            this.handleAddEmployee(e);
        });
    }

    showAttendanceModal(action) {
        const modal = new bootstrap.Modal(document.getElementById('attendanceModal'));
        document.getElementById('attendanceModalTitle').textContent = 
            action === 'check-in' ? 'Employee Check In' : 'Employee Check Out';
        
        // Store the action for later use
        document.getElementById('attendanceModal').dataset.action = action;
        
        modal.show();
        
        // Simulate face recognition after a short delay
        setTimeout(() => {
            this.simulateFaceRecognition();
        }, 1000);
    }

    async simulateFaceRecognition() {
        try {
            // Show scanning state
            const cameraContainer = document.querySelector('#attendanceModal .camera-placeholder');
            cameraContainer.innerHTML = `
                <i class="fas fa-camera fa-3x text-primary mb-3"></i>
                <p class="text-primary">Scanning face...</p>
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Processing...</span>
                </div>
            `;

            // Simulate face recognition
            const result = await FaceRecognition.simulateAuthentication();
            
            // Show success state
            cameraContainer.innerHTML = `
                <i class="fas fa-check-circle fa-3x text-success mb-3"></i>
                <p class="text-success">Face recognized!</p>
                <small class="text-muted">Facial ID: ${result.facialId}</small>
            `;
            
            // Auto-process attendance after successful recognition
            setTimeout(() => {
                this.processAttendanceWithFaceId(result.facialId);
            }, 1500);

        } catch (error) {
            // Show error state
            const cameraContainer = document.querySelector('#attendanceModal .camera-placeholder');
            cameraContainer.innerHTML = `
                <i class="fas fa-times-circle fa-3x text-danger mb-3"></i>
                <p class="text-danger">Face not recognized</p>
                <small class="text-muted">Please try again or select manually</small>
            `;
            
            Notifications.warning('Face not recognized. Please select employee manually.');
        }
    }

    async processAttendance() {
        const action = document.getElementById('attendanceModal').dataset.action;
        const selectedEmployeeId = document.getElementById('employeeSelect').value;
        
        if (!selectedEmployeeId) {
            Notifications.error('Please select an employee');
            return;
        }

        try {
            LoadingManager.show('#processAttendanceBtn', 'Processing...');
            
            // For demo purposes, we'll use the employee ID as a pseudo facial ID
            const pseudoFacialId = `employee_${selectedEmployeeId}_demo`;
            
            const endpoint = action === 'check-in' ? '/api/attendance/check-in' : '/api/attendance/check-out';
            const result = await API.post(endpoint, { facial_id: pseudoFacialId });
            
            Notifications.success(result.message);
            
            // Close modal and refresh data
            bootstrap.Modal.getInstance(document.getElementById('attendanceModal')).hide();
            this.loadDashboardStats();
            this.loadRecentActivity();
            
        } catch (error) {
            Notifications.error(error.message || 'Failed to process attendance');
        } finally {
            LoadingManager.hide('#processAttendanceBtn');
        }
    }

    async processAttendanceWithFaceId(facialId) {
        const action = document.getElementById('attendanceModal').dataset.action;
        
        try {
            const endpoint = action === 'check-in' ? '/api/attendance/check-in' : '/api/attendance/check-out';
            const result = await API.post(endpoint, { facial_id: facialId });
            
            Notifications.success(result.message);
            
            // Close modal and refresh data
            bootstrap.Modal.getInstance(document.getElementById('attendanceModal')).hide();
            this.loadDashboardStats();
            this.loadRecentActivity();
            
        } catch (error) {
            Notifications.error(error.message || 'Failed to process attendance');
        }
    }

    showAddEmployeeModal() {
        const modal = new bootstrap.Modal(document.getElementById('addEmployeeModal'));
        modal.show();
    }

    async handleAddEmployee(e) {
        e.preventDefault();
        
        const formData = {
            name: document.getElementById('employeeName').value,
            email: document.getElementById('employeeEmail').value,
            position: document.getElementById('employeePosition').value,
            department: document.getElementById('employeeDepartment').value,
            password: document.getElementById('employeePassword').value,
            is_admin: document.getElementById('employeeIsAdmin').checked
        };

        try {
            LoadingManager.show('#addEmployeeForm button[type="submit"]', 'Adding...');
            
            const result = await API.post('/api/auth/register', formData);
            
            Notifications.success('Employee added successfully');
            
            // Close modal and reset form
            bootstrap.Modal.getInstance(document.getElementById('addEmployeeModal')).hide();
            FormUtils.resetForm('#addEmployeeForm');
            
            // Refresh dashboard stats
            this.loadDashboardStats();
            this.loadEmployeesList();
            
        } catch (error) {
            Notifications.error(error.message || 'Failed to add employee');
        } finally {
            LoadingManager.hide('#addEmployeeForm button[type="submit"]');
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('totalEmployees')) {
        new Dashboard();
    }
}); 