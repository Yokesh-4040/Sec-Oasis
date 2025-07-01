from .auth import auth_bp
from .employee import employee_bp
from .attendance import attendance_bp
from .views import views_bp

__all__ = ['auth_bp', 'employee_bp', 'attendance_bp', 'views_bp'] 