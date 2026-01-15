from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import os
import numpy as np

# --------------------------------------------------
# App Initialization
# --------------------------------------------------
app = FastAPI(title="Wine Clustering App")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")),
    name="static"
)

# --------------------------------------------------
# Load Models
# --------------------------------------------------
kmeans = joblib.load(os.path.join(BASE_DIR, "models", "kmeans.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# --------------------------------------------------
# Helper function for confidence
# --------------------------------------------------
def compute_confidence(features_scaled, cluster_label):
    """
    Compute confidence as inverse normalized distance to cluster center
    """
    centroid = kmeans.cluster_centers_[cluster_label]
    dist = np.linalg.norm(features_scaled - centroid)
    # simple normalization: closer = higher confidence
    confidence = max(0, 100 - dist * 10)  # scale to 0-100%
    return f"{confidence:.1f}%"

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render homepage.
    """
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None, "confidence_interval": None}
    )

@app.post("/", response_class=HTMLResponse)
def predict(request: Request,
            alcohol: float = Form(...),
            malic_acid: float = Form(...),
            ash: float = Form(...),
            alcalinity_of_ash: float = Form(...),
            magnesium: float = Form(...),
            total_phenols: float = Form(...),
            flavanoids: float = Form(...),
            nonflavanoid_phenols: float = Form(...),
            proanthocyanins: float = Form(...),
            color_intensity: float = Form(...),
            hue: float = Form(...),
            od280_od315_of_diluted_wines: float = Form(...),
            proline: float = Form(...)):
    """
    Predict cluster and compute confidence.
    """
    features = np.array([[alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
                          total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins,
                          color_intensity, hue, od280_od315_of_diluted_wines, proline]])
    
    # Scale input
    features_scaled = scaler.transform(features)
    
    # Predict cluster
    cluster = kmeans.predict(features_scaled)[0]
    
    # Compute confidence
    confidence = compute_confidence(features_scaled, cluster)
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": f"Predicted Cluster: {cluster}",
            "confidence_interval": confidence
        }
    )
