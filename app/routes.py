import os
from flask import current_app,Blueprint, render_template, request, redirect, url_for,flash,session,Response
from .models import Pothole,User,Reports,Maintenance
from .extensions import db
from werkzeug.security import generate_password_hash,check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from datetime import datetime



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

# Upload pothole route
@main.route('/upload_pothole', methods=['GET', 'POST'])
def upload_pothole():
    if request.method == 'POST':
        file = request.files['file']
        location = request.form['location']
        
        if file and allowed_file(file.filename):
            # Read the file's binary content
            file_data = file.read()

            # Handle user session
            if 'user_id' not in session:
                flash('You need to be logged in to report a pothole.', 'error')
                return redirect(url_for('main.login'))

            user_id = session['user_id']

            # Create pothole and report
            new_pothole = Pothole(image=file_data, location=location, status='Pending')
            db.session.add(new_pothole)
            db.session.commit()

            new_report = Reports(uid=user_id, pid=new_pothole.id)
            db.session.add(new_report)
            db.session.commit()

            flash('Pothole reported successfully!', 'success')
            return redirect(url_for('main.dashboard'))

        else:
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



# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, *args, **kwargs):
#         # Remove the 'groups' parameter if it exists
#         kwargs.pop('groups', None)
#         super(CustomDepthwiseConv2D, self).__init__(*args, **kwargs)

#     def get_config(self):
#         config = super(CustomDepthwiseConv2D, self).get_config()
#         if 'groups' in config:
#             del config['groups']
#         return config

# class VideoProcessor:
#     def __init__(self, model_path, labels_path):
#         # Register the custom layer
#         custom_objects = {
#             'DepthwiseConv2D': CustomDepthwiseConv2D
#         }
        
#         # Load the model with custom objects
#         try:
#             self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
#             print("Model loaded successfully!")
#             # Load the labels
#             self.class_names = open(labels_path, "r").readlines()
#             print("Labels loaded successfully!")
#         except Exception as e:
#             print(f"Error loading model or labels: {str(e)}")
#             raise
            
#         self.frame_queue = Queue(maxsize=30)
#         self.result_queue = Queue(maxsize=30)
#         self.is_running = False
        
#     def preprocess_frame(self, frame):
#         # Convert BGR to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Resize the image directly to 224x224 without maintaining aspect ratio
#         resized_frame = cv2.resize(rgb_frame, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
#         # Normalize the image to [-1, 1]
#         normalized_image_array = (resized_frame.astype(np.float32) / 127.5) - 1
        
#         # Save the non-normalized version for visualization
#         cv2.imwrite("preprocessed_frame_for_visualization.jpg", resized_frame)

#         # Create the array of the right shape
#         data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#         data[0] = normalized_image_array
        
#         return data


    
#     def process_frame(self, frame):
#         preprocessed = self.preprocess_frame(frame)
#         prediction = self.model.predict(preprocessed)
#         index = np.argmax(prediction)
#         class_name = self.class_names[index][2:]  # Remove the first two characters
#         confidence_score = prediction[0][index]
        
#         print(f"Predicted class: {class_name.strip()} with confidence: {confidence_score:.2f}")
#         # Return True if pothole (class 0) is detected with confidence
#         is_pothole = index == 0 and confidence_score > 0.6
#         return is_pothole, confidence_score
    
#     def draw_detection(self, frame, is_pothole, confidence):
#         if is_pothole:
#             height, width = frame.shape[:2]
#             # Draw a red box around the frame if pothole is detected
#             cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 2)
#             cv2.putText(frame, f'Pothole Detected! ({confidence:.2f})', (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         return frame

#     def process_video_thread(self):
#         while self.is_running:
#             if not self.frame_queue.empty():
#                 frame = self.frame_queue.get()
#                 is_pothole, confidence = self.process_frame(frame)
#                 processed_frame = self.draw_detection(frame.copy(), is_pothole, confidence)
                
#                 if is_pothole:
#                     # Save detected potholes to database
#                     _, buffer = cv2.imencode('.jpg', frame)
#                     frame_bytes = buffer.tobytes()
                    
#                     # Create new pothole entry
#                     new_pothole = Pothole(
#                         image=frame_bytes,
#                         location="Detected from video",
#                         status='Pending',
#                         timestamp=datetime.now()
#                     )
#                     db.session.add(new_pothole)
#                     db.session.commit()
                
#                 self.result_queue.put(processed_frame)
#             time.sleep(0.01)

#     def start_processing(self):
#         self.is_running = True
#         self.processing_thread = threading.Thread(target=self.process_video_thread)
#         self.processing_thread.start()

#     def stop_processing(self):
#         self.is_running = False
#         if hasattr(self, 'processing_thread'):
#             self.processing_thread.join()

# # Initialize paths
# MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','potholes', 'models', 'pothole_detection.h5'))
# LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'potholes','models', 'labels.txt'))
# video_processor = None

# # Routes remain the same
# @main.route('/video_feed')
# def video_feed():
#     def generate_frames():
#         video_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'test2.mp4')
#         cap = cv2.VideoCapture(video_path)
        
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break
                
#             if not video_processor.frame_queue.full():
#                 video_processor.frame_queue.put(frame)
            
#             if not video_processor.result_queue.empty():
#                 processed_frame = video_processor.result_queue.get()
#                 _, buffer = cv2.imencode('.jpg', processed_frame)
#                 frame_bytes = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
#             time.sleep(0.01)
        
#         cap.release()
    
#     return Response(generate_frames(),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @main.route('/video_detection')
# def video_detection():
#     global video_processor
#     if video_processor is None:
#         try:
#             video_processor = VideoProcessor(MODEL_PATH, LABELS_PATH)
#             video_processor.start_processing()
#             flash('Video processing started successfully!', 'success')
#         except Exception as e:
#             flash(f'Error starting video processing: {str(e)}', 'error')
#             return redirect(url_for('main.home'))
#     return render_template('video_detection.html')



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

@main.route('/upload_detection', methods=['GET', 'POST'])
def upload_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)
            
        file = request.files['file']
        location = request.form['location']
        
        if file and allowed_file(file.filename):
            try:
                MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'potholes','models', 'pothole_detection.h5'))
                LABELS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'potholes','models', 'labels.txt'))
                
                detector = PotholeDetector(MODEL_PATH, LABELS_PATH)
                class_name, confidence = detector.process_image(file)
                
                if class_name.strip().lower() == "pothole":
                    file.seek(0)
                    file_data = file.read()
                    
                    new_pothole = Pothole(
                        image=file_data,
                        location=location,
                        status='Pending',
                    )

                    new_report = Reports(uid=1, pid=new_pothole.id)
                    db.session.add(new_report)
                    db.session.commit()

                    
                    db.session.add(new_pothole)
                    db.session.commit()
                    
                    flash(f'Pothole detected with {confidence:.2f} confidence', 'success')
                else:
                    flash(f'Detected {class_name.strip()} with {confidence:.2f} confidence', 'info')
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(request.url)
                
            return redirect(url_for('main.dashboard'))
            
        flash('Invalid file type', 'error')
        
    return render_template('upload_cam.html')