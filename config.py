from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:mySQL@localhost:3306/pothole_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
engine = create_engine('mysql+pymysql://root:mySQLlocalhost:3306/pothole_system',pool_pre_ping=True)
connection = engine.connect()
print("Connected successfully!")
# Initialize SQLAlchemy
db = SQLAlchemy(app)
