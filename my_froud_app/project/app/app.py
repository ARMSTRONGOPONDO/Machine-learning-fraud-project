from flask import Flask, render_template, request, redirect, send_from_directory, url_for, flash
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import tensorflow as tf
import joblib
import shutil
import time
from werkzeug.utils import secure_filename
from dashboard import create_dashboard
import csv

# Get the base directory of the app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# Configuration with absolute paths
app.config.update({
    "UPLOAD_FOLDER": os.path.join(BASE_DIR, "uploads"),
    "PROCESSED_FOLDER": os.path.join(BASE_DIR, "processed_data"),
    "ALLOWED_EXTENSIONS": {'csv'},
    "MAX_CONTENT_LENGTH": 100 * 1024 * 1024  # 100MB limit
})

# Ensure directories exist with absolute paths
for folder in [app.config["UPLOAD_FOLDER"], app.config["PROCESSED_FOLDER"], os.path.join(BASE_DIR, 'static')]:
    os.makedirs(folder, exist_ok=True)

# Initialize dashboard
create_dashboard(app)

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_pie_chart(fraud_count, non_fraud_count):
    """Generate base64 encoded pie chart with proper resource cleanup"""
    try:
        plt.figure(figsize=(5, 5))
        plt.pie(
            [fraud_count, non_fraud_count],
            labels=["Fraudulent", "Non-Fraudulent"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["red", "green"]
        )
        plt.title("Fraud vs Non-Fraud Transactions")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    finally:
        plt.close('all')
        buf.close()

# --- Model Loading ---
MODELS = {
    'tf_model': ("kenyan_fraud_nn.keras", True),
    'xgb_model': ("kenyan_fraud_xgb.pkl", True),
    'meta_model': ("kenyan_fraud_rf.pkl", True),
    'preprocessor': ("preprocessor.pkl", False)  # No model validation needed
}

def load_models():
    """Load ML models and preprocessor with appropriate validation"""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    models = {}
    
    try:
        for name, (filename, is_model) in MODELS.items():
            path = os.path.join(model_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {filename} not found")
                
            # Load file
            if filename.endswith('.keras'):
                models[name] = tf.keras.models.load_model(path)
            else:
                models[name] = joblib.load(path)
                
            # Validate only actual models
            if is_model and not hasattr(models[name], 'predict'):
                raise ValueError(f"Invalid model format for {filename}")
                
        print("All components loaded successfully")
        return models
        
    except Exception as e:
        raise RuntimeError(f"Loading failed: {str(e)}")

# Load models during app startup
try:
    models = load_models()
    tf_model = models['tf_model']
    xgb_model = models['xgb_model']
    meta_model = models['meta_model']
    column_transformer = models['preprocessor']
except RuntimeError as e:
    print(f"Critical error: {str(e)}")
    exit(1)

# --- Routes ---
@app.route("/")
def index():
    return render_template(
        "index.html",
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route("/predict", methods=["POST"])
def predict():
    if "csvfile" in request.files:
        csv_file = request.files["csvfile"]
        if csv_file.filename.endswith(".csv"):
            try:
                # Create timestamp-based directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], timestamp)
                os.makedirs(upload_dir, exist_ok=True)

                # Save uploaded file
                filename = secure_filename(csv_file.filename)
                filepath = os.path.join(upload_dir, filename)
                csv_file.save(filepath)

                                # ====== CORRECTED PARSING CONFIGURATION ======
                csv_parse_params = {
                    "sep": ",",              
                    "quotechar": '"',
                    "quoting": csv.QUOTE_MINIMAL,
                    "skipinitialspace": True,
                    "engine": "python",
                    "header": 0
                }

                # Validate column structure
                expected_columns = [
                    "transaction_id", "user_name", "credit_card_type",
                    "transaction_amount", "merchant_category", "datetime",
                    "bank", "location", "is_foreign", "transaction_type",
                    "transaction_frequency", "time_since_last_txn_hrs"
                ]  # 12 columns matching your header

                # Read header
                df_header = pd.read_csv(filepath, nrows=0, **csv_parse_params)
                
                # Convert to lowercase and strip whitespace
                df_header.columns = [col.strip().lower() for col in df_header.columns]
                
                # Column count validation
                if len(df_header.columns) != len(expected_columns):
                    raise ValueError(
                        f"Expected {len(expected_columns)} columns, got {len(df_header.columns)}"
                    )

                # Column name validation
                missing_columns = set(expected_columns) - set(df_header.columns)
                if missing_columns:
                    raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

                # Process data in chunks
                chunk_size = 1000
                processed_chunks = []

                for chunk in pd.read_csv(filepath, chunksize=chunk_size, **csv_parse_params):
                    # Enforce column names from header
                    chunk.columns = df_header.columns.tolist()
                    
                    # Validate column names
                    if not all(col in chunk.columns for col in expected_columns):
                        missing = set(expected_columns) - set(chunk.columns)
                        raise ValueError(f"Missing columns: {', '.join(missing)}")

                    # Transform and predict
                    transformed_data = column_transformer.transform(chunk)
                    tf_pred = (tf_model.predict(transformed_data) > 0.5).astype(int).flatten()
                    xgb_pred = xgb_model.predict(transformed_data)
                    meta_pred = meta_model.predict(transformed_data)

                    # Add predictions
                    chunk["TF_Prediction"] = ["Fraudulent" if p else "Non-Fraudulent" for p in tf_pred]
                    chunk["XGB_Prediction"] = ["Fraudulent" if p else "Non-Fraudulent" for p in xgb_pred]
                    chunk["Meta_Prediction"] = ["Fraudulent" if p else "Non-Fraudulent" for p in meta_pred]

                    processed_chunks.append(chunk)

                # Combine and save results
                processed_data = pd.concat(processed_chunks, ignore_index=True)
                output_filename = f"{timestamp}_processed_data.csv"
                processed_path = os.path.join(app.config["PROCESSED_FOLDER"], output_filename)
                processed_data.to_csv(processed_path, index=False)

                # Move to static folder
                static_path = os.path.join(BASE_DIR, 'static', output_filename)   # Use app.static_folder

                try:
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(static_path), exist_ok=True)
                    
                    # First try copy, then verify
                    shutil.copy2(processed_path, static_path)
                    
                    # Verify copy succeeded
                    if not os.path.exists(static_path):
                        raise RuntimeError(f"File copy failed - destination missing: {static_path}")
                    if os.path.getsize(static_path) == 0:
                        raise RuntimeError("Copied file is empty")
                    
                    print(f"[SUCCESS] File copied to: {static_path}")
                    print(f"[VERIFICATION] File size: {os.path.getsize(static_path)} bytes")
                    
                except Exception as copy_error:
                    print(f"[CRITICAL] File copy failed: {str(copy_error)}")
                    # Try alternative method if copy fails
                    try:
                        with open(processed_path, 'rb') as src, open(static_path, 'wb') as dst:
                            dst.write(src.read())
                        print(f"[FALLBACK] Used alternative copy method")
                    except Exception as fallback_error:
                        print(f"[FATAL] All copy methods failed: {str(fallback_error)}")
                        raise RuntimeError("Could not save output file") from fallback_error
                # Generate visualization
                fraud_count = (processed_data["Meta_Prediction"] == "Fraudulent").sum()
                non_fraud_count = (processed_data["Meta_Prediction"] == "Non-Fraudulent").sum()
                pie_chart = generate_pie_chart(fraud_count, non_fraud_count)

                return render_template(
                    "results.html",
                    column_names=processed_data.columns.tolist(),
                    data=processed_data.head(50).values.tolist(),
                    pie_chart=pie_chart,
                    processed_data_filename=output_filename,
                    processed_data_filepath=static_path,
                    current_year=datetime.now().year  # Add this line
                )
                            # With this verified version:

            except ValueError as e:
                flash(f"Error: {str(e)}", "error")
            except Exception as e:
                flash(f"Processing error: {str(e)}", "error")
            return redirect(url_for("index"))
        else:
            flash("Invalid file format. Upload a CSV.", "error")
            return redirect(url_for("index"))
    else:
        flash("No file uploaded.", "error")
        return redirect(url_for("index"))

@app.route("/download_results/<filename>")
def download_results(filename):
    """Serve processed CSV file from static directory"""
    try:
        # Sanitize and verify filename
        safe_filename = secure_filename(filename)
        if not safe_filename.endswith('.csv'):
            raise ValueError("Invalid file type")
            
        print(f"[DOWNLOAD] Request for: {safe_filename}")
        
        # Get absolute static path
        static_dir = os.path.abspath(app.static_folder)
        full_path = os.path.join(static_dir, safe_filename)
        
        # Enhanced file verification
        if not os.path.exists(full_path):
            available_files = [f for f in os.listdir(static_dir) if f.endswith('.csv')]
            print(f"[ERROR] File not found. Available files: {available_files}")
            raise FileNotFoundError("Requested file not available")
            
        # Force CSV MIME type and download
        response = send_from_directory(
            static_dir,
            safe_filename,
            as_attachment=True,
            mimetype='text/csv',
            download_name=f"Fraud_Report_{safe_filename}"
        )
        
        # Additional headers to prevent caching issues
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        
        print(f"[SUCCESS] Served file: {full_path}")
        return response
        
    except Exception as e:
        print(f"[DOWNLOAD ERROR] {str(e)}")
        flash(f"Download error: {str(e)}", "error")
        return redirect(url_for("index"))

@app.route("/launch_dashboard")
def launch_dashboard():
    try:
        # Get absolute path to processed directory
        processed_dir = os.path.abspath(app.config["PROCESSED_FOLDER"])
        print(f"\n[DEBUG] Checking for processed files in directory: {processed_dir}")
        
        # Verify directory exists
        if not os.path.exists(processed_dir):
            print(f"[ERROR] Directory does not exist: {processed_dir}")
            raise FileNotFoundError(f"Directory not found: {processed_dir}")
        
        # List all files in directory
        all_files = os.listdir(processed_dir)
        print(f"[DEBUG] All files in directory: {all_files}")
        
        # Filter and sort processed files
        processed_files = sorted(
            [f for f in all_files if f.endswith('_processed_data.csv')],
            key=lambda x: datetime.strptime(x.split('_')[0], "%Y%m%d"),
            reverse=True
        )
        
        print(f"[DEBUG] Found processed files: {processed_files}")
        
        if not processed_files:
            print("[ERROR] No processed files found matching pattern '*_processed_data.csv'")
            raise FileNotFoundError("No processed files available")
            
        latest_file = processed_files[0]
        latest_file_path = os.path.join(processed_dir, latest_file)
        
        # Verify the file exists
        if not os.path.exists(latest_file_path):
            print(f"[ERROR] File not found at path: {latest_file_path}")
            raise FileNotFoundError(f"File not found: {latest_file_path}")
        
        print(f"[DEBUG] Latest processed file: {latest_file_path}")
        print(f"[DEBUG] File exists: {os.path.exists(latest_file_path)}")
        print(f"[DEBUG] File size: {os.path.getsize(latest_file_path)} bytes")
        
        # Write the path to the file
        with open("latest_file_path.txt", "w") as f:
            f.write(latest_file_path)
            print(f"[DEBUG] Wrote path to latest_file_path.txt: {latest_file_path}")
            
        return redirect("/dashboard/")
        
    except Exception as e:
        print(f"[ERROR] Exception in launch_dashboard: {str(e)}")
        flash(str(e), "error")
        return redirect(url_for("index"))

# Streamlit integration
streamlit_process = None

@app.route('/ai-fraud-detection')
def launch_streamlit():
    global streamlit_process

    # Clean up any existing process
    if streamlit_process and streamlit_process.poll() is None:
        streamlit_process.terminate()
        time.sleep(2)

    try:
        streamlit_script = os.path.join(
            os.path.dirname(__file__), 
            "assistance_API/main.py"
        )
        
        streamlit_process = subprocess.Popen(
            ['streamlit', 'run', streamlit_script, '--server.port', '8502'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, 'PYTHONPATH': os.getcwd()}
        )
        time.sleep(5)  # Allow time for initialization
        return redirect("http://localhost:8502")
        
    except Exception as e:
        flash(f"Failed to launch analytics: {str(e)}", "error")
        return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')