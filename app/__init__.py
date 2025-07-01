from flask import Flask
from flask_login import LoginManager
from flask_cors import CORS
from .models import db
import os

login_manager = LoginManager()

@login_manager.user_loader
def load_user(user_id):
    from .models.employee import Employee
    return Employee.query.get(int(user_id))

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///attendance.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize extensions
    CORS(app)
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    # Import and register blueprints
    from .routes import auth_bp, employee_bp, attendance_bp, views_bp
    app.register_blueprint(views_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(employee_bp)
    app.register_blueprint(attendance_bp)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app 