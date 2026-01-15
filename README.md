Wine Quality Clustering Web App
A machine learning web application that uses Unsupervised Learning to group different types of wines based on their chemical composition. The project compares multiple clustering algorithms to determine which best identifies the underlying cultivars in the dataset.

🚀 Features
Machine Learning Models: Implementation of K-Means, Agglomerative Clustering, and DBSCAN.

Model Evaluation: Comparison using Silhouette Score, Davies-Bouldin Index, and Adjusted Rand Index (ARI).

Web Framework: Built with FastAPI for high-performance API routing and HTML rendering.

Preprocessing: Automated feature scaling using Scikit-Learn's StandardScaler.

📂 Project Structure
Plaintext

wine_clustering_project.py/
├── wine_clustering_app/
│   ├── app/
│   │   ├── main.py          # FastAPI application logic
│   │   ├── clustering.py    # Model training and evaluation logic
│   │   ├── templates/       # HTML files
│   │   └── static/          # CSS and JavaScript
│   ├── models/
│   │   ├── kmeans.pkl       # Serialized K-Means model
│   │   └── scaler.pkl       # Serialized scaler object
│   └── notebooks/           # Jupyter notebooks for data analysis
├── venv/                    # Virtual environment
├── requirements.txt         # Project dependencies
└── README.md
🛠️ Installation & Setup
Clone the repository:

Bash

git clone https://https://https://github.com/Urielthebane/Wine-Clustering-Web-App-with-FastAPI
cd wine-clustering-project.py
Create and Activate Virtual Environment:

Bash

python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux
Install Dependencies:

Bash

pip install -r requirements.txt
Run the Application:

Bash

cd wine_clustering_app/app
uvicorn main:app --reload
📊 Methodology
The application analyzes the UCI Wine Dataset, consisting of 13 chemical features (e.g., Alcohol, Magnesium, Phenols).

K-Means & Agglomerative Clustering: These algorithms showed clearer cluster separation for this specific dataset.

DBSCAN: Utilized primarily to identify noise points and outliers that do not strictly adhere to a specific cultivar group.

📝 Author
Akinmusire Oluwankorinola Data Science & AI Portfolio Project
