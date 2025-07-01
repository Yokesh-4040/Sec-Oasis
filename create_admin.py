#!/usr/bin/env python3
"""
Script to create an initial admin user for SecOasis
"""

from app import create_app
from app.models import db
from app.models.employee import Employee
import sys

def create_admin_user():
    """Create an initial admin user"""
    app = create_app()
    
    with app.app_context():
        # Check if our specific admin already exists
        existing_admin = Employee.query.filter_by(email='admin@secoasis.com').first()
        if existing_admin:
            print(f"Admin user already exists: {existing_admin.email}")
            return
        
        # Default admin data
        admin_data = {
            'name': 'System Administrator',
            'email': 'admin@secoasis.com',
            'position': 'System Administrator',
            'department': 'IT',
            'password': 'admin123',
            'is_admin': True,
            'is_active': True
        }
        
        print("Creating initial admin user for SecOasis")
        print("=" * 50)
        
        # Create the admin user
        admin_user = Employee(
            name=admin_data['name'],
            email=admin_data['email'],
            position=admin_data['position'],
            department=admin_data['department'],
            is_admin=True,
            is_active=True
        )
        admin_user.set_password(admin_data['password'])
        
        try:
            db.session.add(admin_user)
            db.session.commit()
            
            print("✅ Admin user created successfully!")
            print("=" * 50)
            print(f"Name: {admin_user.name}")
            print(f"Email: {admin_user.email}")
            print(f"Position: {admin_user.position}")
            print(f"Department: {admin_user.department}")
            print(f"Password: {admin_data['password']}")
            print("=" * 50)
            print("⚠️  Please change the password after first login!")
            
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error creating admin user: {str(e)}")
            return False
            
        return True

if __name__ == '__main__':
    create_admin_user() 