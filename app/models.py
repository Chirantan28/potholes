from . import db
from sqlalchemy import LargeBinary

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

class Pothole(db.Model):
    __tablename__ = 'potholes'
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(200))
    status = db.Column(db.String(50), default='Pending')
    image = db.Column(LargeBinary, nullable=False)

class Reports(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    uid = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pid = db.Column(db.Integer, db.ForeignKey('potholes.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('reports', lazy=True))
    pothole = db.relationship('Pothole', backref=db.backref('reports', lazy=True))

    def __repr__(self):
        return f"<Report id={self.id} user_id={self.uid} pothole_id={self.pid}>"