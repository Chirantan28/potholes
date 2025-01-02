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
    
class Maintenance(db.Model):
    __tablename__ = 'maintenance'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    pothole_id = db.Column(db.Integer, db.ForeignKey('potholes.id'), nullable=False)
    development_image = db.Column(LargeBinary, nullable=True)
    maintenance_date = db.Column(db.Date, nullable=False)
    maintenance_type = db.Column(db.String(100), nullable=False)
    cost = db.Column(db.Float, nullable=False)
    notes = db.Column(db.String(500), nullable=True)

    pothole = db.relationship('Pothole', backref=db.backref('maintenance', lazy=True))

    def __repr__(self):
        return f"<Maintenance id={self.id} pothole_id={self.pothole_id} date={self.maintenance_date} type={self.maintenance_type}>"
    
class Risks(db.Model):
    __tablename__ = 'risks'
    pid = db.Column(db.Integer, primary_key=True)
    risk = db.Column(db.String(100), nullable=True)
    priority = db.Column(db.String(100), nullable=True)
    rpic = db.Column(db.LargeBinary, nullable=True)