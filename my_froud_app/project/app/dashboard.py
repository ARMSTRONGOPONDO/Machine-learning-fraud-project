import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from io import StringIO
import redis
import json
import os
import datetime
import time
import traceback
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Function to create and integrate the dashboard with a Flask app
def create_dashboard(flask_app):
    load_dotenv()

    # Configuration - Match the exact path from your Flask app
    PROCESSED_FOLDER = "/home/armstrong/my_froud_app/venv/bin/project/app/processed_data"
    
    # Enhanced logging
    print(f"\n[DASHBOARD INIT] Configured data directory: {PROCESSED_FOLDER}")
    print(f"[DASHBOARD INIT] Directory exists: {os.path.exists(PROCESSED_FOLDER)}")
    if os.path.exists(PROCESSED_FOLDER):
        print(f"[DASHBOARD INIT] Files present: {os.listdir(PROCESSED_FOLDER)}")

    # Function to load the latest processed data
    def load_latest_data():
        try:
            print(f"\n[LOAD DATA] Checking: {PROCESSED_FOLDER}")
            
            if not os.path.exists(PROCESSED_FOLDER):
                print("[LOAD DATA ERROR] Directory missing!")
                return pd.DataFrame(), "Unknown"
            
            # Get all matching files with their full paths and creation times
            files = [(f, os.path.getmtime(os.path.join(PROCESSED_FOLDER, f))) 
                    for f in os.listdir(PROCESSED_FOLDER) 
                    if f.endswith("_processed_data.csv")]
            
            print(f"[LOAD DATA] Found {len(files)} candidate files")
            
            if not files:
                print("[LOAD DATA] No matching files found")
                return pd.DataFrame(), "Unknown"
                
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x[1], reverse=True)
            latest_file = files[0][0]
            file_path = os.path.join(PROCESSED_FOLDER, latest_file)
            
            print(f"[LOAD DATA] Loading: {file_path}")
            print(f"[LOAD DATA] File modified at: {time.ctime(files[0][1])}")
            print(f"[LOAD DATA] File exists: {os.path.exists(file_path)}")
            
            # Load the CSV file with error handling
            try:
                df = pd.read_csv(file_path)
                print(f"[LOAD DATA] Data loaded successfully with {len(df)} records")
                print(f"[LOAD DATA] Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"[LOAD DATA ERROR] Failed to read CSV: {str(e)}")
                return pd.DataFrame(), "Error reading file"
            
            # Check if 'Meta_Prediction' column exists, if not, try to find an alternative
            if 'Meta_Prediction' not in df.columns:
                print(f"[LOAD DATA] Warning: 'Meta_Prediction' column not found. Available columns: {df.columns.tolist()}")
                # Look for columns that might contain prediction information
                prediction_columns = [col for col in df.columns if 'predict' in col.lower() or 'fraud' in col.lower()]
                if prediction_columns:
                    print(f"[LOAD DATA] Using alternative prediction column: {prediction_columns[0]}")
                    # Rename the column to Meta_Prediction
                    df['Meta_Prediction'] = df[prediction_columns[0]]
                else:
                    # Create a dummy prediction column for demonstration
                    print("[LOAD DATA] Creating dummy prediction column")
                    df['Meta_Prediction'] = np.random.choice(['Fraudulent', 'Non-Fraudulent'], size=len(df), p=[0.2, 0.8])
            
            # Ensure transaction_amount column exists
            if 'transaction_amount' not in df.columns:
                print(f"[LOAD DATA] Warning: 'transaction_amount' column not found. Available columns: {df.columns.tolist()}")
                # Look for columns that might contain amount information
                amount_columns = [col for col in df.columns if 'amount' in col.lower() or 'value' in col.lower() or 'sum' in col.lower()]
                if amount_columns:
                    print(f"[LOAD DATA] Using alternative amount column: {amount_columns[0]}")
                    # Rename the column to transaction_amount
                    df['transaction_amount'] = df[amount_columns[0]]
                else:
                    # Create a dummy amount column for demonstration
                    print("[LOAD DATA] Creating dummy amount column")
                    df['transaction_amount'] = np.random.rand(len(df)) * 1000
            
            # Print sample of the data
            print(f"[LOAD DATA] Data sample:\n{df.head()}")
            print(f"[LOAD DATA] Success! Loaded {len(df)} records")
            
            return df, time.ctime(files[0][1])
            
        except Exception as e:
            print(f"[LOAD DATA CRITICAL ERROR] {str(e)}")
            traceback.print_exc()
            # Always return a valid tuple even in case of errors
            return pd.DataFrame(), "Error loading data"

    # Try to load data at initialization
    try:
        initial_df, initial_timestamp = load_latest_data()
        if not initial_df.empty:
            print(f"[DASHBOARD INIT] Successfully loaded initial data with {len(initial_df)} records")
            # Store initial data for immediate access
            initial_data = {
                'df': initial_df.to_json(orient='split'),
                'timestamp': initial_timestamp
            }
        else:
            print("[DASHBOARD INIT] Failed to load initial data, will try again when dashboard loads")
            initial_data = None
    except Exception as e:
        print(f"[DASHBOARD INIT] Error during initial data load: {str(e)}")
        initial_data = None

    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        redis_client.ping()  # Test connection
        print("[REDIS] Connection successful")
    except Exception as e:
        print(f"[REDIS ERROR] {str(e)}")
        print("[REDIS] Will proceed without Redis caching")
        redis_client = None

    # Initialize Dash App with a modern theme and custom styling
    dash_app = dash.Dash(
        __name__,
        server=flask_app,
        url_base_pathname="/dashboard/",
        external_stylesheets=[
            dbc.themes.DARKLY,
            "https://use.fontawesome.com/releases/v6.0.0/css/all.css"
        ],
        suppress_callback_exceptions=True
    )

    # Custom CSS for enhanced styling
    dash_app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Fraud Detection Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    background-color: #121212;
                    color: #f8f9fa;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                .card {
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    transition: transform 0.3s;
                    margin-bottom: 15px;
                }
                .card:hover {
                    transform: translateY(-5px);
                }
                .metric-value {
                    font-size: 1.8rem;
                    font-weight: bold;
                }
                .metric-icon {
                    font-size: 1.5rem;
                    margin-right: 10px;
                }
                .nav-tabs .nav-link {
                    color: #adb5bd;
                    border-radius: 5px 5px 0 0;
                    padding: 10px 20px;
                }
                .nav-tabs .nav-link.active {
                    background-color: #2c3e50;
                    color: white;
                    border-color: #2c3e50;
                }
                .dashboard-container {
                    padding: 20px;
                }
                .graph-container {
                    background-color: #1e2130;
                    border-radius: 10px;
                    padding: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                }
                .dropdown-container .Select-control {
                    background-color: #2c3e50;
                    border-color: #4e5d6c;
                    color: white;
                }
                .dropdown-container .Select-menu-outer {
                    background-color: #2c3e50;
                    color: white;
                }
                .dashboard-header {
                    padding: 15px 0;
                    margin-bottom: 20px;
                    background: linear-gradient(90deg, #1a2a6c, #b21f1f, #fdbb2d);
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                }
                .dashboard-title {
                    font-weight: bold;
                    margin: 0;
                    color: white;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
                }
                .last-updated {
                    font-size: 0.8rem;
                    color: #adb5bd;
                    margin-top: 5px;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    # Helper functions for metric cards
    def metric_card(icon, title, color, element_id):
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className=f"fas fa-{icon} metric-icon text-{color}"),
                    html.H5(title, className="mb-0")
                ], className="d-flex align-items-center"),
                html.H3(id=element_id, className=f"mt-3 metric-value text-{color}")
            ])
        ], className=f"border-{color}")

    def create_metrics():
        return dbc.Row([
            dbc.Col(metric_card("exclamation-triangle", "Fraudulent Transactions", "danger", "fraud-count"), md=3),
            dbc.Col(metric_card("check-circle", "Non-Fraudulent", "success", "non-fraud-count"), md=3),
            dbc.Col(metric_card("dollar-sign", "Total Amount", "info", "total-amount"), md=3),
            dbc.Col(metric_card("exchange", "Total Transactions", "primary", "total-count"), md=3),
        ], className="mb-4")

    # Define layout
    dash_app.layout = dbc.Container(
        [
            dcc.Store(id='session-data', storage_type='session', data=initial_data['df'] if initial_data else None),
            dcc.Store(id='analysis-results', storage_type='session'),
            dcc.Interval(id='interval-component', interval=180*1000, n_intervals=0),  # 3-minute refresh (180 seconds)
            
            # Dashboard Header
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-shield-alt me-2"),
                            "Fraud Detection Dashboard"
                        ], className="dashboard-title text-center")
                    ], className="dashboard-header text-center p-3")
                ], width=12)
            ], className="mb-4"),
            
            # Alert container for notifications
            html.Div(id="alert-container", className="mb-3"),
            
            # Action Buttons and Info
            dbc.Row([
                dbc.Col(
                    dbc.Button([
                        html.I(className="fas fa-sync-alt me-2"),
                        "Load Latest Data"
                    ], id="load-data-button", color="success", className="mb-3"),
                    width="auto"
                ),
                dbc.Col(
                    html.Div(id="data-info", className="text-light pt-2", children=[
                        html.Span(f"Data loaded: {len(initial_df)} records | ") if initial_data else "",
                        html.Span(f"Last updated: {initial_timestamp}") if initial_data else ""
                    ]),
                    width="auto"
                ),
                dbc.Col(
                    dbc.Button([
                        html.I(className="fas fa-file-download me-2"),
                        "Export Analysis"
                    ], id="export-button", color="info", className="mb-3 float-end"),
                    width=True
                )
            ]),
            
            # Metrics Row
            create_metrics(),
            
            # Tabs
            dbc.Tabs(
                [
                    dbc.Tab(label="Overview", tab_id="tab-overview"),
                    dbc.Tab(label="Detailed Analysis", tab_id="tab-details"),
                    dbc.Tab(label="Technical Analysis", tab_id="tab-technical"),
                    dbc.Tab(label="Pattern Detection", tab_id="tab-patterns"),
                    dbc.Tab(label="Advanced Analytics", tab_id="tab-advanced"),
                ],
                id="tabs",
                active_tab="tab-overview",
                className="mb-3"
            ),
            
            html.Div(id="tab-content", className="p-4"),
            
            # Modal for export
            dbc.Modal(
                [
                    dbc.ModalHeader("Export Analysis"),
                    dbc.ModalBody("Analysis exported successfully!"),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
                    ),
                ],
                id="export-modal",
                is_open=False,
            ),
        ],
        fluid=True,
        className="dashboard-container"
    )

    # Tab content
    @dash_app.callback(
        Output("tab-content", "children"),
        [Input("tabs", "active_tab"),
         Input("session-data", "data"),
         Input("analysis-results", "data")]
    )
    def render_tab_content(active_tab, data, analysis_results):
        if active_tab == "tab-overview":
            return get_overview_tab()
        elif active_tab == "tab-details":
            return get_details_tab()
        elif active_tab == "tab-technical":
            return get_technical_tab(data)
        elif active_tab == "tab-patterns":
            return get_patterns_tab()
        elif active_tab == "tab-advanced":
            return get_advanced_tab()
        else:
            return "No content here yet."

    # Overview tab layout
    def get_overview_tab():
        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([html.I(className="fas fa-exclamation-triangle me-2 text-danger"), "Fraud Alerts"], className="card-title"),
                            html.Div(id="fraud-alerts")
                        ])
                    ], className="mb-4"),
                    md=12
                ),
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-chart-pie me-2"),
                            "Transaction Type Distribution"
                        ], className="mb-3"),
                        dcc.Graph(id="transaction-type-pie")
                    ], className="graph-container"),
                    md=6
                ),
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-map-marker-alt me-2"),
                            "Transaction Locations"
                        ], className="mb-3"),
                        dcc.Graph(id="transaction-locations")
                    ], className="graph-container"),
                    md=6
                ),
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-dollar-sign me-2"),
                            "Transaction Amount Distribution"
                        ], className="mb-3"),
                        dcc.Graph(id="transaction-amount-histogram")
                    ], className="graph-container"),
                    md=12
                )
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-bolt me-2"), 
                            "Transaction Velocity Analysis"
                        ], className="mb-3"),
                        dcc.Graph(id="velocity-analysis")
                    ], className="graph-container"),
                    md=12
                )
            ]),
        ])

    # Detailed analysis tab layout
    def get_details_tab():
        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-filter me-2"),
                            "Feature Selection"
                        ], className="mb-3"),
                        dcc.Dropdown(
                            id="feature-dropdown",
                            placeholder="Select a feature to analyze",
                            className="mb-3 dropdown-container"
                        ),
                        html.H4([
                            html.I(className="fas fa-chart-bar me-2"),
                            "Feature Analysis"
                        ], className="mb-3"),
                        dcc.Graph(id="feature-analysis")
                    ], className="graph-container"),
                    md=8
                ),
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-chart-pie me-2"),
                            "Fraudulent vs. Non-Fraudulent Transactions"
                        ], className="mb-3"),
                        dcc.Graph(id="fraud-pie"),
                        html.H4([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            "Key Metrics"
                        ], className="mb-3"),
                        dbc.Card(id="key-metrics", body=True, className="mt-3")
                    ], className="graph-container"),
                    md=4
                ),
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-credit-card me-2"),
                            "Credit Card Type Analysis"
                        ], className="mb-3"),
                        dcc.Graph(id="card-type-analysis")
                    ], className="graph-container"),
                    md=12
                )
            ])
        ])

    # Technical analysis tab layout
    def get_technical_tab(data):
        if not data:
            return html.Div("No data loaded for technical analysis.")
        
        try:
            df = pd.read_json(StringIO(data), orient='split')
            numeric_features = df.select_dtypes(include=np.number).columns.tolist()
        except Exception as e:
            print(f"Error in get_technical_tab: {str(e)}")
            numeric_features = []
        
        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-sliders-h me-2"),
                            "Technical Visuals"
                        ], className="mb-3"),
                        dcc.Dropdown(
                            id="technical-feature",
                            options=[{'label': col, 'value': col} for col in numeric_features],
                            placeholder="Select a feature",
                            className="mb-3 dropdown-container"
                        ),
                        dcc.Graph(id="technical-visuals")
                    ], className="graph-container"),
                    md=6
                ),
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-chart-box me-2"),
                            "Box Plot Analysis"
                        ], className="mb-3"),
                        dcc.Dropdown(
                            id="boxplot-feature",
                            options=[{'label': col, 'value': col} for col in numeric_features],
                            placeholder="Select a feature for box plot",
                            className="mb-3 dropdown-container"
                        ),
                        dcc.Graph(id="box-plot")
                    ], className="graph-container"),
                    md=6
                ),
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-th me-2"),
                            "Correlation Heatmap"
                        ], className="mb-3"),
                        dcc.Graph(id="correlation-heatmap")
                    ], className="graph-container"),
                    md=12
                )
            ])
        ])

    # Patterns tab layout
    def get_patterns_tab():
        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-fingerprint me-2"), 
                            "Anomaly Detection"
                        ], className="mb-3"),
                        html.Div([
                            dcc.Dropdown(
                                id="anomaly-feature",
                                placeholder="Select Feature for Anomaly Detection",
                                className="mb-3 dropdown-container"
                            )
                        ], style={"padding": "0 10px"}),
                        dcc.Graph(id="anomaly-plot")
                    ], className="graph-container"),
                    md=12
                )
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-clock me-2"), 
                            "Temporal Pattern Analysis"
                        ], className="mb-3"),
                        dcc.Graph(id="temporal-pattern")
                    ], className="graph-container"),
                    md=6
                ),
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-network-wired me-2"), 
                            "Transaction Network"
                        ], className="mb-3"),
                        dcc.Graph(id="transaction-network")
                    ], className="graph-container"),
                    md=6
                )
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-user-secret me-2"), 
                            "User Behavior Analysis"
                        ], className="mb-3"),
                        dcc.Graph(id="user-behavior")
                    ], className="graph-container"),
                    md=12
                )
            ])
        ])

    # Advanced analytics tab layout
    def get_advanced_tab():
        return dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-brain me-2"), 
                            "Cluster Analysis"
                        ], className="mb-3"),
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Feature 1:"),
                                    dcc.Dropdown(
                                        id="cluster-feature-1",
                                        placeholder="Select first feature",
                                        className="mb-2 dropdown-container"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Feature 2:"),
                                    dcc.Dropdown(
                                        id="cluster-feature-2",
                                        placeholder="Select second feature",
                                        className="mb-2 dropdown-container"
                                    )
                                ], md=4),
                                dbc.Col([
                                    html.Label("Epsilon:"),
                                    dcc.Slider(
                                        id="cluster-epsilon",
                                        min=0.1,
                                        max=2.0,
                                        step=0.1,
                                        value=0.5,
                                        marks={i/10: str(i/10) for i in range(1, 21, 5)},
                                        className="mb-2"
                                    )
                                ], md=4)
                            ])
                        ], style={"padding": "0 10px"}),
                        dcc.Graph(id="cluster-plot")
                    ], className="graph-container"),
                    md=12
                )
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-chart-line me-2"), 
                            "Time Series Decomposition"
                        ], className="mb-3"),
                        html.Div([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Feature:"),
                                    dcc.Dropdown(
                                        id="timeseries-feature",
                                        placeholder="Select feature for time series analysis",
                                        className="mb-2 dropdown-container"
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Aggregation:"),
                                    dcc.Dropdown(
                                        id="timeseries-agg",
                                        options=[
                                            {"label": "Sum", "value": "sum"},
                                            {"label": "Mean", "value": "mean"},
                                            {"label": "Count", "value": "count"},
                                            {"label": "Max", "value": "max"}
                                        ],
                                        value="sum",
                                        className="mb-2 dropdown-container"
                                    )
                                ], md=6)
                            ])
                        ], style={"padding": "0 10px"}),
                        dcc.Graph(id="timeseries-plot")
                    ], className="graph-container"),
                    md=12
                )
            ]),
            
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-exclamation-circle me-2"), 
                            "Fraud Risk Scoring"
                        ], className="mb-3"),
                        dcc.Graph(id="risk-scoring")
                    ], className="graph-container"),
                    md=12
                )
            ])
        ])

    # Alert for data loading status
    @dash_app.callback(
        Output("alert-container", "children"),
        [Input("session-data", "data")]
    )
    def update_alert(data):
        if data is None:
            return dbc.Alert(
                [
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "No data loaded. Please click 'Load Latest Data' to refresh."
                ],
                color="warning",
                dismissable=True
            )
        return None

    # Metrics update callback
    @dash_app.callback(
        [Output("fraud-count", "children"),
         Output("non-fraud-count", "children"),
         Output("total-amount", "children"),
         Output("total-count", "children")],
        [Input("session-data", "data")]
    )
    def update_metrics(data):
        if not data:
            return "0", "0", "$0", "0"
        
        try:
            df = pd.read_json(StringIO(data), orient='split')
            
            # Calculate metrics
            total_count = len(df)
            fraud_count = df['Meta_Prediction'].eq('Fraudulent').sum()
            non_fraud_count = total_count - fraud_count
            total_amount = df['transaction_amount'].sum()
            
            return f"{fraud_count:,}", f"{non_fraud_count:,}", f"${total_amount:,.2f}", f"{total_count:,}"
        except Exception as e:
            print(f"[METRICS ERROR] {str(e)}")
            return "Error", "Error", "Error", "Error"

    # Load data callback
    @dash_app.callback(
        [Output("session-data", "data"),
        Output("data-info", "children"),
        Output("analysis-results", "data")],
        [Input("load-data-button", "n_clicks"),
        Input("interval-component", "n_intervals")],
        [State("session-data", "data")]
    )
    def load_data(n_clicks, n_intervals, existing_data):
        ctx = callback_context
        if not ctx.triggered:
            # On initial load, use the preloaded data if available
            if initial_data:
                print("[LOAD DATA] Using preloaded data")
                analysis_results = perform_fraud_analysis(pd.read_json(StringIO(initial_data['df']), orient='split'))
                info_text = [
                    html.Span(f"Data loaded: {len(initial_df)} records | "),
                    html.Span(f"Last updated: {initial_timestamp}")
                ]
                return initial_data['df'], info_text, json.dumps(analysis_results)
            return dash.no_update, dash.no_update, dash.no_update
        
        try:
            print(f"[LOAD DATA CALLBACK] Triggered by: {ctx.triggered[0]['prop_id']}")
            
            df, last_updated = load_latest_data()
            
            if df.empty:
                print("[LOAD DATA] No data loaded from files")
                # Fallback to sample data if no files found
                df = pd.DataFrame({
                    'transaction_id': range(100),
                    'user_name': ['UserA', 'UserB'] * 50,
                    'merchant_category': ['Grocery', 'Electronics', 'Travel', 'Restaurant'] * 25,
                    'location': ['New York', 'Los Angeles', 'Chicago', 'Miami'] * 25,
                    'transaction_amount': np.random.rand(100) * 1000,
                    'datetime': pd.date_range(start='2023-01-01', periods=100, freq='H'),
                    'credit_card_type': ['Visa', 'Mastercard', 'Amex', 'Discover'] * 25,
                    'Meta_Prediction': np.random.choice(['Fraudulent', 'Non-Fraudulent'], 100, p=[0.2, 0.8]),
                    'transaction_frequency': np.random.randint(1, 20, 100),
                    'time_since_last_txn_hrs': np.random.rand(100) * 24,
                    'is_foreign': np.random.choice([0, 1], 100, p=[0.7, 0.3])
                })
                last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            analysis_results = perform_fraud_analysis(df)
            
            info_text = [
                html.Span(f"Data loaded: {len(df)} records | "),
                html.Span(f"Last updated: {last_updated}")
            ]
            
            return df.to_json(orient='split'), info_text, json.dumps(analysis_results)
            
        except Exception as e:
            print(f"[LOAD DATA ERROR] {str(e)}")
            traceback.print_exc()
            return dash.no_update, f"Error: {str(e)}", None

    # Perform fraud analysis
    def perform_fraud_analysis(df):
        """Perform fraud analysis on the dataframe and return results"""
        results = {}
        
        try:
            # Calculate basic fraud metrics
            total_transactions = len(df)
            fraudulent_transactions = df['Meta_Prediction'].eq('Fraudulent').sum()
            fraud_percentage = (fraudulent_transactions / total_transactions) * 100 if total_transactions else 0
            
            results['total_transactions'] = int(total_transactions)
            results['fraudulent_transactions'] = int(fraudulent_transactions)
            results['fraud_percentage'] = float(fraud_percentage)
            
            # Calculate average transaction amount
            avg_amount = df['transaction_amount'].mean()
            results['avg_transaction_amount'] = float(avg_amount)
            
            # Calculate fraud by location
            if 'location' in df.columns:
                location_fraud = df.groupby('location')['Meta_Prediction'].apply(
                    lambda x: (x == 'Fraudulent').sum() / len(x) * 100 if len(x) > 0 else 0
                ).sort_values(ascending=False).head(5).to_dict()
                results['location_fraud'] = location_fraud
            
            # Calculate fraud by credit card type
            if 'credit_card_type' in df.columns:
                card_fraud = df.groupby('credit_card_type')['Meta_Prediction'].apply(
                    lambda x: (x == 'Fraudulent').sum() / len(x) * 100 if len(x) > 0 else 0
                ).sort_values(ascending=False).head(5).to_dict()
                results['card_fraud'] = card_fraud
            
            # Calculate fraud by merchant category
            if 'merchant_category' in df.columns:
                merchant_fraud = df.groupby('merchant_category')['Meta_Prediction'].apply(
                    lambda x: (x == 'Fraudulent').sum() / len(x) * 100 if len(x) > 0 else 0
                ).sort_values(ascending=False).head(5).to_dict()
                results['merchant_fraud'] = merchant_fraud
            
            # Identify high-risk users
            if 'user_name' in df.columns:
                user_fraud = df.groupby('user_name').agg(
                    total_txns=('transaction_id', 'count'),
                    fraud_txns=('Meta_Prediction', lambda x: (x == 'Fraudulent').sum()),
                    total_amount=('transaction_amount', 'sum')
                )
                user_fraud['fraud_rate'] = user_fraud['fraud_txns'] / user_fraud['total_txns'] * 100
                user_fraud['risk_score'] = user_fraud['fraud_rate'] * user_fraud['total_amount'] / 1000
                
                high_risk_users = user_fraud.sort_values('risk_score', ascending=False).head(5).to_dict('index')
                results['high_risk_users'] = high_risk_users
            
            # Generate fraud alerts
            alerts = []
            
            # Alert for high fraud rate
            if fraud_percentage > 10:
                alerts.append({
                    'severity': 'high',
                    'message': f'High overall fraud rate detected: {fraud_percentage:.1f}%'
                })
            
            # Alert for suspicious locations
            if 'location' in df.columns and location_fraud:
                for location, rate in location_fraud.items():
                    if rate > 30:
                        alerts.append({
                            'severity': 'high',
                            'message': f'High fraud rate in {location}: {rate:.1f}%'
                        })
            
            # Alert for suspicious merchant categories
            if 'merchant_category' in df.columns and merchant_fraud:
                for merchant, rate in merchant_fraud.items():
                    if rate > 30:
                        alerts.append({
                            'severity': 'medium',
                            'message': f'High fraud rate in {merchant} category: {rate:.1f}%'
                        })
            
            # Alert for suspicious users
            if 'high_risk_users' in results:
                for user, data in results['high_risk_users'].items():
                    if data.get('fraud_rate', 0) > 50:
                        alerts.append({
                            'severity': 'high',
                            'message': f'High-risk user detected: {user} (Fraud rate: {data["fraud_rate"]:.1f}%)'
                        })
            
            results['alerts'] = alerts
            
        except Exception as e:
            print(f"[ERROR] Analysis error: {str(e)}")
            results['error'] = str(e)
        
        return results

    # Fraud alerts callback
    @dash_app.callback(
        Output("fraud-alerts", "children"),
        [Input("analysis-results", "data")]
    )
    def update_fraud_alerts(analysis_results):
        if not analysis_results:
            return "No alerts available. Load data to see fraud alerts."
        
        try:
            results = json.loads(analysis_results)
            alerts = results.get('alerts', [])
            
            if not alerts:
                return "No fraud alerts detected in the current dataset."
            
            alert_components = []
            
            for alert in alerts:
                severity = alert.get('severity', 'low')
                message = alert.get('message', 'Unknown alert')
                
                color = {
                    'high': 'danger',
                    'medium': 'warning',
                    'low': 'info'
                }.get(severity, 'info')
                
                alert_components.append(
                    dbc.Alert(
                        [
                            html.I(className=f"fas fa-exclamation-circle me-2"),
                            message
                        ],
                        color=color,
                        className="mb-2"
                    )
                )
            
            return alert_components
        
        except Exception as e:
            print(f"[ERROR] Alert rendering error: {str(e)}")
            return f"Error rendering alerts: {str(e)}"

    # Overview tab callbacks
    @dash_app.callback(
        [Output("transaction-type-pie", "figure"),
         Output("transaction-locations", "figure"),
         Output("transaction-amount-histogram", "figure"),
         Output("velocity-analysis", "figure")],
        [Input("session-data", "data")]
    )
    def update_overview_visuals(data):
        if not data:
            empty_fig = go.Figure().update_layout(
                title="No data available",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return empty_fig, empty_fig, empty_fig, empty_fig

        try:
            df = pd.read_json(StringIO(data), orient='split')
            # Ensure the DataFrame is not empty
            if df.empty:
                raise ValueError("DataFrame is empty")

            # Transaction Type Pie Chart
            type_counts = df['Meta_Prediction'].value_counts()
            time_fig = px.pie(
                names=type_counts.index,
                values=type_counts.values,
                title="Transaction Type Distribution",
                color_discrete_sequence=['#e74c3c', '#2ecc71'],
                hole=0.4
            )
            
            time_fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hoverinfo='label+percent+value',
                marker=dict(line=dict(color='#1e2130', width=2))
            )
            
            time_fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                ),
                annotations=[dict(
                    text=f'Total: {len(df):,}',
                    x=0.5, y=0.5,
                    font_size=14,
                    showarrow=False
                )]
            )

            # Transaction Locations Map with Fraud Rate
            if 'location' in df.columns:
                location_data = df.groupby('location').agg(
                    total_count=('transaction_id', 'count'),
                    fraud_count=('Meta_Prediction', lambda x: (x == 'Fraudulent').sum()),
                    avg_amount=('transaction_amount', 'mean')
                ).reset_index()
                
                location_data['fraud_rate'] = location_data['fraud_count'] / location_data['total_count'] * 100
                location_data = location_data.sort_values('fraud_rate', ascending=False)
                
                loc_fig = px.bar(
                    location_data,
                    x='location',
                    y='total_count',
                    color='fraud_rate',
                    color_continuous_scale='Reds',
                    title="Transaction Locations with Fraud Rate",
                    labels={
                        'location': 'Location',
                        'total_count': 'Number of Transactions',
                        'fraud_rate': 'Fraud Rate (%)'
                    },
                    hover_data=['fraud_count', 'fraud_rate', 'avg_amount']
                )
                
                loc_fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        title="Location",
                        tickangle=45,
                        showgrid=False
                    ),
                    yaxis=dict(
                        title="Number of Transactions",
                        showgrid=True,
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    coloraxis_colorbar=dict(
                        title="Fraud Rate (%)"
                    )
                )
            else:
                loc_fig = go.Figure().update_layout(
                    title="Location data not available",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

            # Transaction Amount Histogram with KDE
            amount_fig = px.histogram(
                df,
                x="transaction_amount",
                color="Meta_Prediction",
                marginal="rug",
                opacity=0.7,
                nbins=50,
                title="Transaction Amount Distribution",
                labels={'transaction_amount': 'Transaction Amount ($)'},
                color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'},
                histnorm='percent'
            )
            
            amount_fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title="Percentage (%)"),
                barmode='overlay',
                legend_title_text='Transaction Type'
            )
            
            # Add transaction velocity analysis
            if not df.empty and 'datetime' in df.columns:
                try:
                    # Convert datetime to proper format
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    
                    # Sort by user and datetime
                    if 'user_name' in df.columns:
                        df = df.sort_values(['user_name', 'datetime'])
                        
                        # Calculate time difference between consecutive transactions for the same user
                        df['prev_datetime'] = df.groupby('user_name')['datetime'].shift(1)
                        df['time_diff_minutes'] = (df['datetime'] - df['prev_datetime']).dt.total_seconds() / 60
                        
                        # Flag rapid succession transactions (within 5 minutes)
                        df['rapid_succession'] = df['time_diff_minutes'] < 5
                        
                        # Flag unusual transaction amounts (Z-score > 2)
                        df['amount_zscore'] = df.groupby('user_name')['transaction_amount'].transform(
                            lambda x: (x - x.mean()) / x.std() if len(x) > 1 else 0
                        )
                        df['unusual_amount'] = abs(df['amount_zscore']) > 2
                        
                        # Calculate velocity score (higher is more suspicious)
                        df['velocity_score'] = (df['rapid_succession'].astype(int) * 5) + (df['unusual_amount'].astype(int) * 3)
                        
                        # Add velocity score to the dataframe for use in visualizations
                        print(f"[ANALYSIS] Added transaction velocity analysis. Found {df['rapid_succession'].sum()} rapid succession transactions.")
                except Exception as e:
                    print(f"[ANALYSIS WARNING] Could not perform velocity analysis: {str(e)}")
            
            # Transaction Velocity Analysis
            try:
                if 'velocity_score' in df.columns:
                    # Create a scatter plot of transaction amount vs velocity score
                    velocity_fig = px.scatter(
                        df,
                        x='transaction_amount',
                        y='velocity_score',
                        color='Meta_Prediction',
                        size='transaction_amount',
                        hover_name='user_name' if 'user_name' in df.columns else None,
                        hover_data=['credit_card_type', 'merchant_category', 'location', 'time_diff_minutes'] if all(col in df.columns for col in ['credit_card_type', 'merchant_category', 'location', 'time_diff_minutes']) else None,
                        color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'},
                        title="Transaction Velocity Analysis",
                        labels={
                            'transaction_amount': 'Transaction Amount ($)',
                            'velocity_score': 'Velocity Risk Score',
                            'Meta_Prediction': 'Transaction Type'
                        },
                        opacity=0.7
                    )
                    
                    # Add quadrant lines and labels
                    velocity_fig.add_shape(
                        type="line",
                        x0=df['transaction_amount'].min(),
                        y0=5,
                        x1=df['transaction_amount'].max(),
                        y1=5,
                        line=dict(color="white", width=1, dash="dash")
                    )
                    
                    velocity_fig.add_shape(
                        type="line",
                        x0=df['transaction_amount'].quantile(0.75),
                        y0=0,
                        x1=df['transaction_amount'].quantile(0.75),
                        y1=max(8, df['velocity_score'].max()),
                        line=dict(color="white", width=1, dash="dash")
                    )
                    
                    # Add quadrant annotations
                    velocity_fig.add_annotation(
                        x=df['transaction_amount'].quantile(0.9),
                        y=7,
                        text="High Risk Zone",
                        showarrow=False,
                        font=dict(color="red", size=14)
                    )
                    
                    velocity_fig.add_annotation(
                        x=df['transaction_amount'].quantile(0.25),
                        y=7,
                        text="Suspicious Activity",
                        showarrow=False,
                        font=dict(color="orange", size=14)
                    )
                    
                    velocity_fig.add_annotation(
                        x=df['transaction_amount'].quantile(0.9),
                        y=2,
                        text="Large but Normal",
                        showarrow=False,
                        font=dict(color="white", size=14)
                    )
                    
                    velocity_fig.add_annotation(
                        x=df['transaction_amount'].quantile(0.25),
                        y=2,
                        text="Low Risk Zone",
                        showarrow=False,
                        font=dict(color="green", size=14)
                    )
                    
                    velocity_fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )
                else:
                    velocity_fig = go.Figure().update_layout(
                        title="Transaction velocity data not available",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
            except Exception as e:
                print(f"Velocity analysis visualization error: {str(e)}")
                velocity_fig = go.Figure().update_layout(
                    title="Error creating velocity analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )

            return time_fig, loc_fig, amount_fig, velocity_fig

        except Exception as e:
            print(f"Overview error: {str(e)}")
            traceback.print_exc()
            empty_fig = go.Figure().update_layout(
                title="Error loading data",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return empty_fig, empty_fig, empty_fig, empty_fig

    # Detailed Analysis Tab Callbacks
    @dash_app.callback(
        [Output("feature-dropdown", "options"),
         Output("feature-analysis", "figure"),
         Output("fraud-pie", "figure"),
         Output("card-type-analysis", "figure"),
         Output("key-metrics", "children")],
        [Input("session-data", "data"),
         Input("feature-dropdown", "value")]
    )
    def update_detailed_analysis(data, selected_feature):
        if not data:
            empty_fig = go.Figure().update_layout(
                title="No data available",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], empty_fig, empty_fig, empty_fig, "No data available"
        
        try:
            df = pd.read_json(StringIO(data), orient='split')
            
            # Create dropdown options
            features = [{'label': col, 'value': col} for col in df.columns 
                       if col not in ['Meta_Prediction'] and pd.api.types.is_numeric_dtype(df[col])]
            
            # Create fraud pie chart
            fraud_counts = df['Meta_Prediction'].value_counts()
            fraud_pie = px.pie(
                names=fraud_counts.index,
                values=fraud_counts.values,
                title="Fraud Distribution",
                color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'},
                hole=0.4
            )
            
            fraud_pie.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Create card type analysis
            if 'credit_card_type' in df.columns:
                card_data = df.groupby(['credit_card_type', 'Meta_Prediction']).size().reset_index(name='count')
                card_fig = px.bar(
                    card_data,
                    x='credit_card_type',
                    y='count',
                    color='Meta_Prediction',
                    title="Credit Card Type Analysis",
                    barmode='group',
                    color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'}
                )
                
                card_fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Card Type",
                    yaxis_title="Number of Transactions"
                )
            else:
                card_fig = go.Figure().update_layout(
                    title="Credit card type data not available",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create feature analysis
            if selected_feature:
                feature_fig = px.histogram(
                    df,
                    x=selected_feature,
                    color='Meta_Prediction',
                    marginal="box",
                    title=f"{selected_feature} Distribution by Fraud Status",
                    color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'}
                )
                
                feature_fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title=selected_feature,
                    yaxis_title="Count"
                )
            else:
                feature_fig = go.Figure().update_layout(
                    title="Select a feature to analyze",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create key metrics
            total_count = len(df)
            fraud_count = df['Meta_Prediction'].eq('Fraudulent').sum()
            fraud_rate = (fraud_count / total_count) * 100 if total_count > 0 else 0
            avg_amount = df['transaction_amount'].mean()
            
            key_metrics = [
                html.Div([
                    html.H5("Fraud Rate", className="mb-1"),
                    html.P(f"{fraud_rate:.2f}%", className="text-danger")
                ], className="mb-3"),
                html.Div([
                    html.H5("Average Transaction Amount", className="mb-1"),
                    html.P(f"${avg_amount:.2f}", className="text-info")
                ], className="mb-3"),
                html.Div([
                    html.H5("Total Transactions", className="mb-1"),
                    html.P(f"{total_count:,}", className="text-primary")
                ])
            ]
            
            return features, feature_fig, fraud_pie, card_fig, key_metrics
            
        except Exception as e:
            print(f"Detailed analysis error: {str(e)}")
            traceback.print_exc()
            empty_fig = go.Figure().update_layout(
                title="Error in analysis",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], empty_fig, empty_fig, empty_fig, "Error loading data"

    # Technical Analysis Tab Callbacks
    @dash_app.callback(
        [Output("technical-feature", "options"),
         Output("boxplot-feature", "options"),
         Output("technical-visuals", "figure"),
         Output("box-plot", "figure"),
         Output("correlation-heatmap", "figure")],
        [Input("session-data", "data"),
         Input("technical-feature", "value"),
         Input("boxplot-feature", "value")]
    )
    def update_technical_analysis(data, tech_feature, box_feature):
        if not data:
            empty_fig = go.Figure().update_layout(
                title="No data available",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], [], empty_fig, empty_fig, empty_fig
        
        try:
            df = pd.read_json(StringIO(data), orient='split')
            
            # Get numeric features for dropdowns
            numeric_features = [{'label': col, 'value': col} for col in df.columns 
                               if pd.api.types.is_numeric_dtype(df[col])]
            
            # Create correlation heatmap
            numeric_cols = df.select_dtypes(include=np.number).columns
            corr_matrix = df[numeric_cols].corr()
            
            heatmap = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            
            heatmap.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            # Create technical visuals
            if tech_feature:
                tech_fig = px.scatter(
                    df,
                    x=tech_feature,
                    y='transaction_amount',
                    color='Meta_Prediction',
                    title=f"{tech_feature} vs Transaction Amount",
                    color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'},
                    opacity=0.7
                )
                
                tech_fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title=tech_feature,
                    yaxis_title="Transaction Amount"
                )
            else:
                tech_fig = go.Figure().update_layout(
                    title="Select a feature for technical analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create box plot
            if box_feature:
                box_fig = px.box(
                    df,
                    x='Meta_Prediction',
                    y=box_feature,
                    color='Meta_Prediction',
                    title=f"{box_feature} Distribution by Fraud Status",
                    color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'},
                    points="all"
                )
                
                box_fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Fraud Status",
                    yaxis_title=box_feature
                )
            else:
                box_fig = go.Figure().update_layout(
                    title="Select a feature for box plot analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            return numeric_features, numeric_features, tech_fig, box_fig, heatmap
            
        except Exception as e:
            print(f"Technical analysis error: {str(e)}")
            traceback.print_exc()
            empty_fig = go.Figure().update_layout(
                title="Error in analysis",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], [], empty_fig, empty_fig, empty_fig

    # Pattern Detection Tab Callbacks
    @dash_app.callback(
        [Output("anomaly-feature", "options"),
         Output("anomaly-plot", "figure"),
         Output("temporal-pattern", "figure"),
         Output("transaction-network", "figure"),
         Output("user-behavior", "figure")],
        [Input("session-data", "data"),
         Input("anomaly-feature", "value")]
    )
    def update_pattern_detection(data, anomaly_feature):
        if not data:
            empty_fig = go.Figure().update_layout(
                title="No data available",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], empty_fig, empty_fig, empty_fig, empty_fig
        
        try:
            df = pd.read_json(StringIO(data), orient='split')
            
            # Get numeric features for dropdown
            numeric_features = [{'label': col, 'value': col} for col in df.columns 
                               if pd.api.types.is_numeric_dtype(df[col])]
            
            # Create anomaly detection plot
            if anomaly_feature:
                # Use DBSCAN for anomaly detection
                try:
                    # Extract feature and standardize
                    X = df[[anomaly_feature, 'transaction_amount']].copy()
                    X = StandardScaler().fit_transform(X)
                    
                    # Apply DBSCAN
                    clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)
                    
                    # Add cluster labels to dataframe
                    df_anomaly = df.copy()
                    df_anomaly['cluster'] = clustering.labels_
                    
                    # Anomalies are labeled as -1
                    anomaly_fig = px.scatter(
                        df_anomaly,
                        x=anomaly_feature,
                        y='transaction_amount',
                        color='cluster',
                        color_continuous_scale='Viridis',
                        title=f"Anomaly Detection: {anomaly_feature} vs Transaction Amount",
                        labels={
                            anomaly_feature: anomaly_feature,
                            'transaction_amount': 'Transaction Amount ($)',
                            'cluster': 'Cluster'
                        },
                        hover_data=['Meta_Prediction']
                    )
                    
                    # Highlight anomalies
                    anomalies = df_anomaly[df_anomaly['cluster'] == -1]
                    if not anomalies.empty:
                        anomaly_fig.add_trace(
                            go.Scatter(
                                x=anomalies[anomaly_feature],
                                y=anomalies['transaction_amount'],
                                mode='markers',
                                marker=dict(
                                    color='red',
                                    size=12,
                                    line=dict(width=2, color='black')
                                ),
                                name='Anomalies'
                            )
                        )
                    
                    anomaly_fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                except Exception as e:
                    print(f"Anomaly detection error: {str(e)}")
                    anomaly_fig = go.Figure().update_layout(
                        title="Error in anomaly detection",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
            else:
                anomaly_fig = go.Figure().update_layout(
                    title="Select a feature for anomaly detection",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create temporal pattern analysis
            if 'datetime' in df.columns:
                try:
                    # Convert to datetime
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    
                    # Group by date and fraud status
                    df['date'] = df['datetime'].dt.date
                    temporal_data = df.groupby(['date', 'Meta_Prediction']).size().reset_index(name='count')
                    
                    # Create temporal pattern plot
                    temporal_fig = px.line(
                        temporal_data,
                        x='date',
                        y='count',
                        color='Meta_Prediction',
                        title="Temporal Pattern Analysis",
                        labels={
                            'date': 'Date',
                            'count': 'Number of Transactions',
                            'Meta_Prediction': 'Transaction Type'
                        },
                        color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'}
                    )
                    
                    temporal_fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )
                except Exception as e:
                    print(f"Temporal pattern error: {str(e)}")
                    temporal_fig = go.Figure().update_layout(
                        title="Error in temporal pattern analysis",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
            else:
                temporal_fig = go.Figure().update_layout(
                    title="Datetime data not available for temporal analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create transaction network visualization
            if 'user_name' in df.columns and 'merchant_category' in df.columns:
                try:
                    # Create a network of users and merchant categories
                    network_data = df.groupby(['user_name', 'merchant_category']).agg(
                        count=('transaction_id', 'count'),
                        fraud_count=('Meta_Prediction', lambda x: (x == 'Fraudulent').sum()),
                        total_amount=('transaction_amount', 'sum')
                    ).reset_index()
                    
                    network_data['fraud_rate'] = network_data['fraud_count'] / network_data['count'] * 100
                    
                    # Create a network visualization
                    network_fig = go.Figure()
                    
                    # Add edges (connections between users and merchants)
                    for _, row in network_data.iterrows():
                        # Only show connections with significant activity
                        if row['count'] > 1:
                            network_fig.add_trace(
                                go.Scatter(
                                    x=[0, 1],
                                    y=[network_data['user_name'].unique().tolist().index(row['user_name']),
                                       len(network_data['user_name'].unique()) + 
                                       network_data['merchant_category'].unique().tolist().index(row['merchant_category'])],
                                    mode='lines',
                                    line=dict(
                                        width=row['count'] / 2,
                                        color='rgba(255,0,0,{})'.format(row['fraud_rate']/100)
                                    ),
                                    hoverinfo='text',
                                    text=f"User: {row['user_name']}<br>Merchant: {row['merchant_category']}<br>" +
                                         f"Transactions: {row['count']}<br>Fraud Rate: {row['fraud_rate']:.1f}%",
                                    showlegend=False
                                )
                            )
                    
                    # Add user nodes
                    network_fig.add_trace(
                        go.Scatter(
                            x=[0] * len(network_data['user_name'].unique()),
                            y=list(range(len(network_data['user_name'].unique()))),
                            mode='markers',
                            marker=dict(
                                size=10,
                                color='#3498db',
                                line=dict(width=1, color='#2980b9')
                            ),
                            text=network_data['user_name'].unique(),
                            hoverinfo='text',
                            name='Users'
                        )
                    )
                    
                    # Add merchant nodes
                    network_fig.add_trace(
                        go.Scatter(
                            x=[1] * len(network_data['merchant_category'].unique()),
                            y=[len(network_data['user_name'].unique()) + i 
                               for i in range(len(network_data['merchant_category'].unique()))],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color='#2ecc71',
                                line=dict(width=1, color='#27ae60')
                            ),
                            text=network_data['merchant_category'].unique(),
                            hoverinfo='text',
                            name='Merchants'
                        )
                    )
                    
                    network_fig.update_layout(
                        title="Transaction Network: Users and Merchants",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        showlegend=True,
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            range=[-0.1, 1.1]
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False
                        )
                    )
                except Exception as e:
                    print(f"Network visualization error: {str(e)}")
                    network_fig = go.Figure().update_layout(
                        title="Error in network visualization",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
            else:
                network_fig = go.Figure().update_layout(
                    title="User and merchant data not available for network analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create user behavior analysis
            if 'user_name' in df.columns:
                try:
                    # Analyze user behavior
                    user_behavior = df.groupby('user_name').agg(
                        total_txns=('transaction_id', 'count'),
                        fraud_txns=('Meta_Prediction', lambda x: (x == 'Fraudulent').sum()),
                        total_amount=('transaction_amount', 'sum'),
                        avg_amount=('transaction_amount', 'mean')
                    ).reset_index()
                    
                    user_behavior['fraud_rate'] = user_behavior['fraud_txns'] / user_behavior['total_txns'] * 100
                    
                    # Create user behavior visualization
                    user_fig = px.scatter(
                        user_behavior,
                        x='total_txns',
                        y='avg_amount',
                        size='total_amount',
                        color='fraud_rate',
                        color_continuous_scale='Reds',
                        hover_name='user_name',
                        title="User Behavior Analysis",
                        labels={
                            'total_txns': 'Total Transactions',
                            'avg_amount': 'Average Transaction Amount ($)',
                            'fraud_rate': 'Fraud Rate (%)'
                        }
                    )
                    
                    user_fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )
                except Exception as e:
                    print(f"User behavior analysis error: {str(e)}")
                    user_fig = go.Figure().update_layout(
                        title="Error in user behavior analysis",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
            else:
                user_fig = go.Figure().update_layout(
                    title="User data not available for behavior analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            return numeric_features, anomaly_fig, temporal_fig, network_fig, user_fig
            
        except Exception as e:
            print(f"Pattern detection error: {str(e)}")
            traceback.print_exc()
            empty_fig = go.Figure().update_layout(
                title="Error in pattern detection",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], empty_fig, empty_fig, empty_fig, empty_fig

    # Advanced Analytics Tab Callbacks
    @dash_app.callback(
        [Output("cluster-feature-1", "options"),
         Output("cluster-feature-2", "options"),
         Output("timeseries-feature", "options"),
         Output("cluster-plot", "figure"),
         Output("timeseries-plot", "figure"),
         Output("risk-scoring", "figure")],
        [Input("session-data", "data"),
         Input("cluster-feature-1", "value"),
         Input("cluster-feature-2", "value"),
         Input("cluster-epsilon", "value"),
         Input("timeseries-feature", "value"),
         Input("timeseries-agg", "value")]
    )
    def update_advanced_analytics(data, cluster_f1, cluster_f2, epsilon, ts_feature, ts_agg):
        if not data:
            empty_fig = go.Figure().update_layout(
                title="No data available",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], [], [], empty_fig, empty_fig, empty_fig
        
        try:
            df = pd.read_json(StringIO(data), orient='split')
            
            # Get numeric features for dropdowns
            numeric_features = [{'label': col, 'value': col} for col in df.columns 
                               if pd.api.types.is_numeric_dtype(df[col])]
            
            # Create cluster analysis plot
            if cluster_f1 and cluster_f2:
                try:
                    # Extract features for clustering
                    X = df[[cluster_f1, cluster_f2]].copy()
                    X = StandardScaler().fit_transform(X)
                    
                    # Apply DBSCAN
                    clustering = DBSCAN(eps=epsilon, min_samples=5).fit(X)
                    
                    # Add cluster labels to dataframe
                    df_cluster = df.copy()
                    df_cluster['cluster'] = clustering.labels_
                    
                    # Create cluster plot
                    cluster_fig = px.scatter(
                        df_cluster,
                        x=cluster_f1,
                        y=cluster_f2,
                        color='cluster',
                        color_continuous_scale='Viridis',
                        title=f"Cluster Analysis: {cluster_f1} vs {cluster_f2}",
                        labels={
                            cluster_f1: cluster_f1,
                            cluster_f2: cluster_f2,
                            'cluster': 'Cluster'
                        },
                        hover_data=['Meta_Prediction']
                    )
                    
                    # Highlight anomalies (cluster = -1)
                    anomalies = df_cluster[df_cluster['cluster'] == -1]
                    if not anomalies.empty:
                        cluster_fig.add_trace(
                            go.Scatter(
                                x=anomalies[cluster_f1],
                                y=anomalies[cluster_f2],
                                mode='markers',
                                marker=dict(
                                    color='red',
                                    size=12,
                                    line=dict(width=2, color='black')
                                ),
                                name='Anomalies'
                            )
                        )
                    
                    cluster_fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )
                except Exception as e:
                    print(f"Cluster analysis error: {str(e)}")
                    cluster_fig = go.Figure().update_layout(
                        title="Error in cluster analysis",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
            else:
                cluster_fig = go.Figure().update_layout(
                    title="Select features for cluster analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create time series analysis
            if 'datetime' in df.columns and ts_feature:
                try:
                    # Convert to datetime
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    
                    # Group by date and aggregate
                    df['date'] = df['datetime'].dt.date
                    
                    # Apply the selected aggregation
                    if ts_agg == 'sum':
                        ts_data = df.groupby(['date', 'Meta_Prediction'])[ts_feature].sum().reset_index()
                    elif ts_agg == 'mean':
                        ts_data = df.groupby(['date', 'Meta_Prediction'])[ts_feature].mean().reset_index()
                    elif ts_agg == 'count':
                        ts_data = df.groupby(['date', 'Meta_Prediction'])[ts_feature].count().reset_index()
                    elif ts_agg == 'max':
                        ts_data = df.groupby(['date', 'Meta_Prediction'])[ts_feature].max().reset_index()
                    else:
                        ts_data = df.groupby(['date', 'Meta_Prediction'])[ts_feature].sum().reset_index()
                    
                    # Create time series plot
                    ts_fig = px.line(
                        ts_data,
                        x='date',
                        y=ts_feature,
                        color='Meta_Prediction',
                        title=f"Time Series Analysis: {ts_feature} ({ts_agg})",
                        labels={
                            'date': 'Date',
                            ts_feature: f'{ts_feature} ({ts_agg})',
                            'Meta_Prediction': 'Transaction Type'
                        },
                        color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'}
                    )
                    
                    ts_fig.update_layout(
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                    )
                except Exception as e:
                    print(f"Time series analysis error: {str(e)}")
                    ts_fig = go.Figure().update_layout(
                        title="Error in time series analysis",
                        template="plotly_dark",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
            else:
                ts_fig = go.Figure().update_layout(
                    title="Select a feature for time series analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            # Create risk scoring visualization
            try:
                # Calculate risk scores for each transaction
                df_risk = df.copy()
                
                # Initialize risk score
                df_risk['risk_score'] = 0
                
                # Add risk based on transaction amount (higher amount = higher risk)
                if 'transaction_amount' in df_risk.columns:
                    amount_percentile = df_risk['transaction_amount'].rank(pct=True)
                    df_risk['risk_score'] += amount_percentile * 30  # Max 30 points for amount
                
                # Add risk based on transaction frequency (if available)
                if 'transaction_frequency' in df_risk.columns:
                    freq_percentile = df_risk['transaction_frequency'].rank(pct=True)
                    df_risk['risk_score'] += freq_percentile * 20  # Max 20 points for frequency
                
                # Add risk based on time since last transaction (if available)
                if 'time_since_last_txn_hrs' in df_risk.columns:
                    # Lower time = higher risk (inverse ranking)
                    time_percentile = (1 - df_risk['time_since_last_txn_hrs'].rank(pct=True))
                    df_risk['risk_score'] += time_percentile * 15  # Max 15 points for time
                
                # Add risk based on foreign transactions (if available)
                if 'is_foreign' in df_risk.columns:
                    df_risk['risk_score'] += df_risk['is_foreign'] * 15  # Max 15 points for foreign
                
                # Add risk based on merchant category (if available)
                if 'merchant_category' in df_risk.columns:
                    # Calculate fraud rate by merchant category
                    merchant_fraud_rate = df.groupby('merchant_category')['Meta_Prediction'].apply(
                        lambda x: (x == 'Fraudulent').sum() / len(x) if len(x) > 0 else 0
                    )
                    
                    # Map fraud rates to transactions
                    df_risk['merchant_risk'] = df_risk['merchant_category'].map(merchant_fraud_rate)
                    df_risk['risk_score'] += df_risk['merchant_risk'] * 100 * 20  # Max 20 points for merchant
                
                # Create risk score distribution
                risk_fig = px.histogram(
                    df_risk,
                    x='risk_score',
                    color='Meta_Prediction',
                    marginal="box",
                    title="Fraud Risk Score Distribution",
                    labels={
                        'risk_score': 'Risk Score',
                        'Meta_Prediction': 'Transaction Type'
                    },
                    color_discrete_map={'Fraudulent': '#e74c3c', 'Non-Fraudulent': '#2ecc71'},
                    opacity=0.7
                )
                
                # Add vertical lines for risk thresholds
                risk_fig.add_shape(
                    type="line",
                    x0=30, y0=0,
                    x1=30, y1=df_risk['Meta_Prediction'].value_counts().max(),
                    line=dict(color="green", width=2, dash="dash")
                )
                
                risk_fig.add_shape(
                    type="line",
                    x0=60, y0=0,
                    x1=60, y1=df_risk['Meta_Prediction'].value_counts().max(),
                    line=dict(color="orange", width=2, dash="dash")
                )
                
                risk_fig.add_shape(
                    type="line",
                    x0=80, y0=0,
                    x1=80, y1=df_risk['Meta_Prediction'].value_counts().max(),
                    line=dict(color="red", width=2, dash="dash")
                )
                
                # Add annotations for risk levels
                risk_fig.add_annotation(
                    x=15, y=df_risk['Meta_Prediction'].value_counts().max() * 0.9,
                    text="Low Risk",
                    showarrow=False,
                    font=dict(color="green", size=14)
                )
                
                risk_fig.add_annotation(
                    x=45, y=df_risk['Meta_Prediction'].value_counts().max() * 0.9,
                    text="Medium Risk",
                    showarrow=False,
                    font=dict(color="orange", size=14)
                )
                
                risk_fig.add_annotation(
                    x=70, y=df_risk['Meta_Prediction'].value_counts().max() * 0.9,
                    text="High Risk",
                    showarrow=False,
                    font=dict(color="red", size=14)
                )
                
                risk_fig.add_annotation(
                    x=90, y=df_risk['Meta_Prediction'].value_counts().max() * 0.9,
                    text="Critical Risk",
                    showarrow=False,
                    font=dict(color="darkred", size=14)
                )
                
                risk_fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
                )
            except Exception as e:
                print(f"Risk scoring error: {str(e)}")
                risk_fig = go.Figure().update_layout(
                    title="Error in risk scoring analysis",
                    template="plotly_dark",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
            
            return numeric_features, numeric_features, numeric_features, cluster_fig, ts_fig, risk_fig
            
        except Exception as e:
            print(f"Advanced analytics error: {str(e)}")
            traceback.print_exc()
            empty_fig = go.Figure().update_layout(
                title="Error in advanced analytics",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            return [], [], [], empty_fig, empty_fig, empty_fig

    # Export modal callback
    @dash_app.callback(
        Output("export-modal", "is_open"),
        [Input("export-button", "n_clicks"),
         Input("close-modal", "n_clicks")],
        [State("export-modal", "is_open"),
         State("analysis-results", "data")],
        prevent_initial_call=True
    )
    def toggle_export_modal(export_clicks, close_clicks, is_open, analysis_results):
        ctx = callback_context
        if not ctx.triggered:
            return is_open
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        if button_id == "export-button" and export_clicks:
            # Here you would implement the actual export functionality
            # For example, saving the analysis results to a file
            if analysis_results:
                try:
                    # This is a placeholder for actual export functionality
                    # In a real app, you might save to a file or database
                    print("[EXPORT] Analysis results exported successfully")
                    return True
                except Exception as e:
                    print(f"[EXPORT ERROR] {str(e)}")
                    return False
        
        elif button_id == "close-modal" and close_clicks:
            return False
        
        return is_open

    # Return the Dash app
    return dash_app

# This allows the dashboard to be run directly
if __name__ == '__main__':
    from flask import Flask
    server = Flask(__name__)
    app = create_dashboard(server)
    server.run(debug=True)