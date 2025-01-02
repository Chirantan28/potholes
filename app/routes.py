import os
from flask import current_app,Blueprint, render_template, request, redirect, url_for,flash,session,Response
from .models import Pothole,User,Reports,Maintenance,Risks
from .extensions import db
from werkzeug.security import generate_password_hash,check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps,  ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from sqlalchemy import update
import tensorflow as tf
from keras.models import load_model
from keras.layers import DepthwiseConv2D
import torch
import torchvision.transforms as transforms
from datetime import datetime
import io
import cv2
from sqlalchemy.exc import SQLAlchemyError


main = Blueprint('main', __name__)



@main.route('/')
def home():
    return render_template('index.html')

# Admin Dashboard Route
@main.route('/admin')
def admin():
    potholes = Pothole.query.all()
    return render_template('admin.html', potholes=potholes)

# Update Pothole Status Route
@main.route('/update_pothole/<int:pothole_id>', methods=['POST'])
def update_pothole(pothole_id):
    pothole = Pothole.query.get_or_404(pothole_id)
    new_status = request.form.get('status', 'Repaired')  # Get status from form (defaults to 'Repaired')
    
    try:
        pothole.status = new_status
        db.session.commit()
        flash(f"Pothole ID {pothole_id} updated to '{new_status}' successfully.", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"Error updating pothole: {str(e)}", "danger")
    
    return redirect(url_for('main.admin'))

@main.route('/login', methods=['GET', 'POST'])
def login():
    
    if 'user_id' in session:
        return redirect(url_for('main.dashboard')) 

    if request.method == 'POST':
        # Retrieve the username and password from the form
        username = request.form['username']
        password = request.form['password']

        # Query the database for the user by username
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):  # Use hash comparison
            session['user_id'] = user.id  # Store user ID in session
            session['is_admin'] = user.is_admin
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))  # Redirect to dashboard after successful login
        else:
            # Login failed
            flash('Invalid username or password', 'error')
            return render_template('login.html')  # Re-render the login page

    # Render login page for GET requests
    return render_template('login.html')



@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        repassword= request.form['repassword']
        # Hash the password
        if password==repassword:
            hashed_password = generate_password_hash(password)

            # Create and save the user
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()

            flash('Registration successful!', 'success')
            return redirect(url_for('main.login'))

    return render_template('register.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@main.route('/upload_pothole', methods=['GET', 'POST'])
def upload_pothole():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        image_binary = file.read()
        location = request.form['location']
        
        if file and allowed_file(file.filename):
            try:
                MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'potholes', 'models', 'pothole_detection.h5'))
                LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'potholes', 'models', 'labels.txt'))
                
                detector = PotholeDetector(MODEL_PATH, LABELS_PATH)
                class_name, confidence = detector.process_image(file)
                
                if class_name.strip().lower() == "pothole":
                    # Reset file pointer and read the original image
                    file.seek(0)
                    image = Image.open(file).convert("RGB")
                    
                    # Create a drawing object
                    draw = ImageDraw.Draw(image)
                    width, height = image.size
                    
                    # Draw red box around the image
                    box_color = (255, 0, 0)  # Red color
                    box_thickness = 3
                    draw.rectangle([(0, 0), (width-1, height-1)], outline=box_color, width=box_thickness)
                    
                    # Add confidence score text
                    font_size = int(height/20)  # Adjust font size based on image height
                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    text = f"Pothole: {confidence:.2f}"
                    text_color = (255, 0, 0)  # Red color
                    text_position = (10, 10)  # Padding from top-left corner
                    
                    # Draw text with outline for better visibility
                    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                        draw.text((text_position[0]+dx, text_position[1]+dy), text, fill=(0,0,0), font=font)
                    draw.text(text_position, text, fill=text_color, font=font)
                    
                    # Convert modified image back to bytes
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG', quality=95)
                    file_data = img_byte_arr.getvalue()
                    
                    # Handle user session
                    if 'user_id' not in session:
                        flash('You need to be logged in to report a pothole.', 'error')
                        return redirect(url_for('main.login'))

                    user_id = session['user_id']
                    
                    new_pothole = Pothole(
                        image=image_binary,  # Using the annotated image
                        location=location,
                        status='Pending'
                    )
                    db.session.add(new_pothole)
                    db.session.commit()

                    new_report = Reports(uid=user_id, pid=new_pothole.id)
                    db.session.add(new_report)
                    db.session.commit()
                    
                    return render_template("dashboard.html", alert_message="Pothole detected!")
                else:
                    return render_template("upload.html", alert_message="No pothole detected!")
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
                
         
        flash('Invalid file type. Only PNG, JPG, WEBP, JFIF and JPEG are allowed.', 'error')

    return render_template('upload.html')


@main.route('/repairs', methods=['GET', 'POST'])
def repairs():
    # Fetch all potholes with their maintenance records (if any)
    potholes_with_repairs = Pothole.query.join(Maintenance).all()

    # Fetch potholes from reports table for the "Add Maintenance" modal dropdown
    reports = Reports.query.all()

    if request.method == 'POST':
        pothole_id = request.form['pothole_id']
        maintenance_date = request.form['maintenance_date']
        maintenance_type = request.form['maintenance_type']
        cost = request.form['cost']
        notes = request.form['notes']
        file = request.files['file']

        # Handle file upload
        if file and allowed_file(file.filename):
            file_data = file.read()
        else:
            file_data = None

        # Add new maintenance record
        new_maintenance = Maintenance(
            pothole_id=pothole_id,
            development_image=file_data,
            maintenance_date=maintenance_date,
            maintenance_type=maintenance_type,
            cost=float(cost),
            notes=notes
        )
        
        db.session.add(new_maintenance)
        db.session.commit()

        flash('Maintenance added successfully!', 'success')
        return redirect(url_for('main.repairs'))

    return render_template('repairs.html', potholes_with_repairs=potholes_with_repairs, reports=reports)



@main.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('You need to be logged in to view the dashboard.', 'error')
        return redirect(url_for('main.login'))

    user_id = session['user_id']  # Get the logged-in user ID from session

    # Retrieve all potholes reported by the logged-in user using a join
    potholes = db.session.query(Pothole).join(Reports).filter(Reports.uid == user_id).all()

    # Render the dashboard template and pass the potholes data
    return render_template('dashboard.html', potholes=potholes)


@main.route('/logout')
def logout():
    # Clear the session data to log the user out
    session.pop('user_id', None)  # Remove the user_id from the session
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.home'))


@main.route('/pothole_image/<int:pothole_id>')
def pothole_image(pothole_id):
    pothole = Pothole.query.get_or_404(pothole_id)
    return Response(pothole.image, mimetype='image/jpeg')  # Adjust MIME type as needed
@main.route('/details')
def details():
    return render_template('details.html')


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)
        super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

class PotholeDetector:
    def __init__(self, model_path, labels_path):
        np.set_printoptions(suppress=True)
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
        self.class_names = open(labels_path, "r").readlines()

    def process_image(self, image_file):
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(image_file).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        
        prediction = self.model.predict(data)
        index = np.argmax(prediction)
        class_name = self.class_names[index]
        confidence_score = prediction[0][index]
        
        return class_name[2:], confidence_score


@main.route('/risk_analysis_dash', methods=['GET'])
def risk_analysis_dash():
    # Fetch all potholes from the database
    all_potholes = Pothole.query.all()

    # Fetch all records from the risks table
    all_risks = Risks.query.all()

    # Create a set of pothole IDs that already have risks analyzed
    analyzed_pothole_ids = {risk.pid for risk in all_risks}

    # Separate potholes into those with and without risk analysis
    potholes_with_risks = [pothole for pothole in all_risks if pothole.pid in analyzed_pothole_ids]
    potholes_without_risks = [pothole for pothole in all_potholes if pothole.id not in analyzed_pothole_ids]

    # Render the template with both lists
    return render_template(
        "risk_analysis_list.html",
        potholes_with_risks=potholes_with_risks,
        potholes_without_risks=potholes_without_risks,
    )


@main.route('/risk_analysis/<int:pothole_id>', methods=['GET'])
def risk_analysis(pothole_id):
    pothole = Pothole.query.get_or_404(pothole_id)

    if pothole.image:
        try:
            # Load YOLO model for pothole detection
            YOLO_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt'))
            model = torch.hub.load('yolov5', 'custom', path=YOLO_MODEL_PATH, source='local')

            # Load depth estimation model
            DEPTH_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'modelforDepth.h5'))
            depth_model = load_model(DEPTH_MODEL_PATH)

            # Read the pothole image
            image = cv2.imdecode(np.frombuffer(pothole.image, np.uint8), cv2.IMREAD_COLOR)

            # Run YOLO model on the image
            result = model(image)
            processed_image = result.render()[0]
            success, encoded_image = cv2.imencode('.jpg', processed_image)
            if not success:
                raise ValueError("Failed to encode the processed image.")

            # Convert encoded image to binary
            image_binary = encoded_image.tobytes()
            coordinate_lines = result.xyxy[0]

            depth = None
            for i in coordinate_lines:
                try:
                    # Crop the region of interest (ROI)
                    x1, y1, x2, y2 = map(int, i[:4])
                    roi = image[y1:y2, x1:x2]

                    # Convert ROI to grayscale and preprocess for depth model
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    resized_roi = cv2.resize(gray_roi, (128, 128))
                    roi_array = np.expand_dims(resized_roi, axis=0) / 255.0  # Normalize pixel values

                    # Predict depth
                    depth_prediction = depth_model.predict(np.expand_dims(roi_array, axis=-1))
                    depth = float(depth_prediction[0][0][0][0]) 
                except Exception as e:
                    print(f"Error estimating depth: {e}")
                    continue

            # Determine risk level based on depth
            if depth <= 5:
                risk_level = "Low Risk"
                priority = "Low"
            elif 5 < depth <= 10:
                risk_level = "Medium Risk"
                priority = "Medium"
            else:
                risk_level = "High Risk"
                priority = "High"

            # Update the database
            try:
                risk_record = Risks.query.filter_by(pid=pothole_id).first()
                if risk_record:
                    risk_record.risk = risk_level
                    risk_record.priority = priority
                    risk_record.depth = depth 
                    risk_record.rpic = image_binary
                else:
                    new_risk = Risks(pid=pothole_id, risk=risk_level, priority=priority, depth=depth, rpic=image_binary)
                    db.session.add(new_risk)

                db.session.commit()
                flash("Risk analysis and depth estimation completed successfully.", 'success')
            except SQLAlchemyError as e:
                db.session.rollback()
                flash(f"Database error: {str(e)}", 'error')

            return redirect(url_for('main.risk_analysis_dash'))

        except Exception as e:
            flash(f'Error processing image: {str(e)}', 'error')
            print(e)
            return redirect(url_for('main.risk_analysis_dash'))

    flash("Pothole image not available.", 'error')
    return redirect(url_for('main.risk_analysis_dash'))
