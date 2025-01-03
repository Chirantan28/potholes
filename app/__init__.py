import base64
import os
from flask import Flask
from .extensions import db
from .routes import main  # Only import 'main' blueprint here

def create_app():
    app = Flask(__name__)
    app.secret_key = 'b28d1c7f84b7419d3fa428ffcbf0f5fa' 
    # Set the database URI (update with your actual database credentials)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Chirantana2814%40@localhost:3306/pothole_system'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize the database
    db.init_app(app)

    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp','jfif'}
    app.register_blueprint(main)
    @app.template_filter('b64encode')
    def b64encode_filter(data):
        return base64.b64encode(data).decode('utf-8')

    return app
