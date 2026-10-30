# %% [markdown]
# # UMAP Embeddings for Inference APIs
#
# ## Non-Parametric vs Parametric UMAP
#
# ### Learning Objectives
#
# By the end of this notebook you should be able to:
#
# - Explain what embeddings are
# - Compare PCA, UMAP, and Parametric UMAP
# - Generate 2D and 3D embeddings
# - Understand fit() vs transform()
# - Persist trained UMAP models
# - Integrate embeddings into machine learning APIs
# - Evaluate production deployment considerations
#
# ---
#
# Homework Alignment:
#
# This notebook directly supports the assignment requiring:
#
# - UMAP embedding calculation
# - Configurable UMAP parameters
# - Persistence across API calls
# - Returning embeddings with predictions
# - Error handling
# - Performance considerations

# %%
# Uncomment if needed
# !pip install umap-learn
# !pip install plotly
# !pip install ipywidgets

# %% [markdown]
# # Instructor Notes
#
# Common Misconceptions:
#
# 1. UMAP should be retrained during every API request.
#
# 2. Embedding coordinates have direct semantic meaning.
#
# 3. Embeddings are stable when training repeatedly.
#
# Key Concepts:
#
# - fit() learns the manifold
# - transform() projects onto an existing manifold
# - persistence is critical for consistency
# - embeddings support explainability and monitoring

# %% [markdown]
# # What Are Embeddings?
#
# Suppose our model contains 13 input features:
#
# ```text
# Alcohol
# Malic Acid
# Ash
# Alcalinity
# ...
# ```
#
# Visualizing a 13-dimensional space is impossible.
#
# UMAP creates a lower dimensional representation:
#
# ```text
# 13 Features
#      ↓
#     UMAP
#      ↓
#    (x,y)
# ```
#
# allowing us to:
#
# - visualize data
# - discover clusters
# - monitor drift
# - identify anomalies
# - enhance interpretability

# %% [markdown]
# # PCA vs UMAP
#
# | Method | Linear | Nonlinear | Transform New Data |
# |----------|----------|----------|----------|
# | PCA | Yes | No | Yes |
# | t-SNE | No | Yes | No |
# | UMAP | No | Yes | Yes |
# | Parametric UMAP | No | Yes | Yes |

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA

import umap

# %%
wine = load_wine()

X = wine.data
y = wine.target

print("Shape:", X.shape)

# %% [markdown]
# # PCA Projection

# %%
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(8,6))
plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y,
    cmap="viridis"
)

plt.title("PCA Projection")
plt.show()

# %% [markdown]
# # Basic UMAP Projection

# %%
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

X_umap = reducer.fit_transform(X)

# %%
plt.figure(figsize=(8,6))

plt.scatter(
    X_umap[:,0],
    X_umap[:,1],
    c=y,
    cmap="viridis"
)

plt.title("UMAP Projection")
plt.show()

# %% [markdown]
# ## Discussion Prompt
#
# Compare PCA and UMAP:
#
# - Which clusters appear more distinct?
# - Which technique better preserves local neighborhoods?
# - Why might UMAP be preferred for visualization?

# %% [markdown]
# # Understanding UMAP Parameters
#
# ## n_neighbors
#
# Controls:
#
# - Local structure
# - Global structure
#
# Smaller values:
#
# - focus on local neighbors
#
# Larger values:
#
# - preserve broader structure
#
# ---
#
# ## min_dist
#
# Controls:
#
# - tightness of clusters

# %% [markdown]
# # Interactive Sweep: n_neighbors

# %%
from ipywidgets import interact

# %%
@interact(n_neighbors=(2,100,1))
def explore_neighbors(n_neighbors=15):

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        random_state=42
    )

    emb = reducer.fit_transform(X)

    plt.figure(figsize=(8,5))

    plt.scatter(
        emb[:,0],
        emb[:,1],
        c=y,
        cmap="viridis"
    )

    plt.title(f"n_neighbors={n_neighbors}")
    plt.show()

# %% [markdown]
# ### Checkpoint
#
# Compare:
#
# - n_neighbors=5
# - n_neighbors=50
#
# How do the clusters change?

# %% [markdown]
# # Interactive Sweep: min_dist

# %%
@interact(min_dist=(0.0,1.0,0.01))
def explore_distance(min_dist=0.1):

    reducer = umap.UMAP(
        min_dist=min_dist,
        random_state=42
    )

    emb = reducer.fit_transform(X)

    plt.figure(figsize=(8,5))

    plt.scatter(
        emb[:,0],
        emb[:,1],
        c=y,
        cmap="viridis"
    )

    plt.title(f"min_dist={min_dist:.2f}")
    plt.show()

# %% [markdown]
# ### Checkpoint
#
# Compare:
#
# - min_dist=0.01
# - min_dist=0.75
#
# What changes do you observe?

# %% [markdown]
# # 3D UMAP Embeddings

# %%
reducer_3d = umap.UMAP(
    n_components=3,
    random_state=42
)

embedding_3d = reducer_3d.fit_transform(X)

embedding_3d.shape

# %%
import plotly.express as px

# %%
df_3d = pd.DataFrame(
    {
        "x": embedding_3d[:,0],
        "y": embedding_3d[:,1],
        "z": embedding_3d[:,2],
        "class": y
    }
)

fig = px.scatter_3d(
    df_3d,
    x="x",
    y="y",
    z="z",
    color="class"
)

fig.show()

# %% [markdown]
# ## Discussion
#
# When might a 3D embedding be preferable to a 2D embedding?

# %% [markdown]
# # Non-Parametric UMAP
#
# Standard UMAP is non-parametric.
#
# It learns:
#
# ```text
# Data
#   ↓
# Manifold Graph
#   ↓
# Embedding
# ```
#
# No neural network is trained.

# %%
reducer = umap.UMAP(random_state=42)

reducer.fit(X)

sample = X[0].reshape(1,-1)

embedded_sample = reducer.transform(sample)

embedded_sample

# %% [markdown]
# # fit() vs transform()
#
# Production systems should:
#
# ```python
# reducer.fit(training_data)
# ```
#
# once
#
# and
#
# ```python
# reducer.transform(new_data)
# ```
#
# repeatedly

# %% [markdown]
# # Why Persistence Matters
#
# BAD
#
# ```text
# request
#    ↓
# fit UMAP
#    ↓
# transform
# ```
#
# GOOD
#
# ```text
# training
#    ↓
# fit
#    ↓
# save model
#    ↓
# load
#    ↓
# transform requests
# ```

# %% [markdown]
# # Persisting UMAP Models with joblib

# %%
import joblib

# %%
joblib.dump(
    reducer,
    "umap_model.pkl"
)

# %%
loaded_reducer = joblib.load(
    "umap_model.pkl"
)

# %%
loaded_reducer.transform(sample)

# %% [markdown]
# ## Checkpoint
#
# Why is model persistence required by the assignment?

# %% [markdown]
# # Parametric UMAP
#
# Parametric UMAP trains a neural network.
#
# ```text
# Features
#    ↓
# Neural Network
#    ↓
# Embedding
# ```
#
# Benefits:
#
# - faster inference
# - scalable deployment
# - neural network representation
#
# Drawback:
#
# - longer training time

# %%
# Optional
# !pip install "umap-learn[parametric_umap]"

# %%
PARAMETRIC_AVAILABLE = True

try:
    from umap.parametric_umap import ParametricUMAP
except:
    PARAMETRIC_AVAILABLE = False

# %%
if PARAMETRIC_AVAILABLE:

    pumap = ParametricUMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2
    )

    parametric_embedding = pumap.fit_transform(X)

    print(parametric_embedding.shape)

else:

    print("Parametric UMAP not installed")

# %% [markdown]
# # Visual Comparison
#
# PCA vs UMAP vs Parametric UMAP

# %%
fig, axs = plt.subplots(
    1,
    3,
    figsize=(18,5)
)

axs[0].scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y
)
axs[0].set_title("PCA")

axs[1].scatter(
    X_umap[:,0],
    X_umap[:,1],
    c=y
)
axs[1].set_title("UMAP")

if PARAMETRIC_AVAILABLE:

    axs[2].scatter(
        parametric_embedding[:,0],
        parametric_embedding[:,1],
        c=y
    )

    axs[2].set_title(
        "Parametric UMAP"
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Checkpoint
#
# Which projection best separates the classes?
#
# Why?

# %% [markdown]
# # Performance Comparison

# %%
import time

# %%
start = time.time()

_ = reducer.transform(X)

nonparam_time = time.time() - start

print(
    "Non Parametric:",
    nonparam_time
)

# %%
if PARAMETRIC_AVAILABLE:

    start = time.time()

    _ = pumap.transform(X)

    param_time = (
        time.time() - start
    )

    print(
        "Parametric:",
        param_time
    )

# %% [markdown]
# ## Discussion
#
# Which approach would be preferred for:
#
# - a dashboard?
# - a streaming system?
# - millions of transformations per day?

# %% [markdown]
# # API Integration Example

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
classifier = RandomForestClassifier(
    random_state=42
)

classifier.fit(X, y)

# %% [markdown]
# ## Production Architecture
#
# ```text
# Request
#    │
#    ▼
# Classifier
#    │
#    ├──── Label
#    │
#    ▼
# UMAP Transform
#    │
#    ▼
# Embedding
#    │
#    ▼
# Response
# ```

# %%
def predict(features):

    prediction = int(
        classifier.predict(
            [features]
        )[0]
    )

    embedding = reducer.transform(
        [features]
    )[0]

    return {
        "label": prediction,
        "embeddings": {
            "kind": "inputs",
            "x": float(embedding[0]),
            "y": float(embedding[1])
        }
    }

# %%
predict(X[0])

# %% [markdown]
# # FastAPI Example
#
# This directly maps to the homework assignment.

# %%
fastapi_example = """
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

classifier = joblib.load(
    'classifier.pkl'
)

umap_model = joblib.load(
    'umap_model.pkl'
)

class PredictionRequest(
    BaseModel
):
    features: list[float]

@app.post('/predict')
def predict(request):

    label = classifier.predict(
        [request.features]
    )[0]

    embedding = (
        umap_model.transform(
            [request.features]
        )[0]
    )

    return {
        'label': int(label),
        'embeddings': {
            'kind':'inputs',
            'x': float(embedding[0]),
            'y': float(embedding[1])
        }
    }
"""

print(fastapi_example)

# %% [markdown]
# # Example Assignment Response

# %%
example_response = {
    "label": 1,
    "embeddings": {
        "kind": "inputs",
        "x": 0.24,
        "y": -0.38
    }
}

example_response

# %% [markdown]
# # Error Handling

# %%
def safe_predict(features):

    try:

        return predict(
            features
        )

    except Exception as ex:

        return {
            "error": str(ex)
        }

# %%
safe_predict([1,2,3])

# %% [markdown]
# Potential Errors:
#
# - wrong feature count
# - missing values
# - unloaded model
# - bad input types

# %% [markdown]
# # Scaling Considerations
#
# Strategies:
#
# 1. Cache models in memory
#
# 2. Batch transform records
#
# 3. Avoid retraining during inference
#
# 4. Use Parametric UMAP for very large deployments
#
# 5. Persist fitted models

# %% [markdown]
# ## Checkpoint
#
# When would Parametric UMAP be superior to standard UMAP?

# %% [markdown]
# # Assignment Mapping

# %%
mapping = pd.DataFrame(
    {
        "Requirement": [
            "UMAP Calculation",
            "Persistence",
            "Response Format",
            "Error Handling",
            "Performance",
            "Configurable Parameters"
        ],
        "Notebook Section":[
            "UMAP",
            "joblib",
            "API Example",
            "safe_predict",
            "Benchmark",
            "Interactive Sweeps"
        ]
    }
)

mapping

# %% [markdown]
# # Lab Exercise
#
# Build an API that:
#
# 1. Loads a persisted classifier
# 2. Loads a persisted UMAP model
# 3. Returns label and embeddings
# 4. Handles errors
# 5. Supports configurable UMAP dimensions
# 6. Benchmarks performance

# %% [markdown]
# # Grading Rubric

# %%
rubric = pd.DataFrame(
    {
        "Category":[
            "UMAP Integration",
            "Model Persistence",
            "API Response",
            "Error Handling",
            "Performance",
            "Code Quality"
        ],
        "Points":[
            25,
            15,
            20,
            10,
            10,
            20
        ]
    }
)

rubric

# %% [markdown]
# # Final Reflection
#
# Answer the following:
#
# 1. What problem does UMAP solve?
# 2. Why should UMAP be persisted?
# 3. What is the difference between Parametric and Non-Parametric UMAP?
# 4. How could embeddings be used for drift detection?
# 5. How would you scale this system for millions of requests?
#
# ---
#
# End of notebook.