import os
from flask import current_app,Blueprint, render_template, request, redirect, url_for,flash,session,Response
from .models import Pothole,User,Reports,Maintenance
from .extensions import db
from werkzeug.security import generate_password_hash,check_password_hash
from werkzeug.utils import secure_filename


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