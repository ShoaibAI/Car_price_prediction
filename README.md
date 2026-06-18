# Car Price Prediction Using Scikit-Learn Pipelines

This project builds an end-to-end machine learning regression model to predict used car prices based on mechanical and brand specifications. It transitions from an initial exploratory script into a production-grade Scikit-Learn preprocessing pipeline and includes an interactive web interface built with Gradio.

# Core Features & Architecture

* *Automated Data Pipelines:* Uses 'ColumnTransformer' and 'make_pipeline' to manage numeric scaling, median imputation, and categorical one-hot encoding seamlessly. 
* *Target Transformation:* Integrates a logarithmic transformation ('np.log1p') directly into the model training phase using 'TransformedTargetRegressor' to handle right-skewed pricing distributions and prevent the underprediction of high-end vehicles.
* *Production Resilience:* Configured with 'handle_unknown='ignore' on structural categorical steps to ensure the application can handle unseen categories (like rare luxury car brands in validation sets) gracefully without crashing.
* *Interactive Interface:* Wraps the entire finalized pipeline inside a local Gradio UI dashboard, enabling immediate real-time inferences via slider and dropdown components inside Jupyter Notebooks.

# Project Structure

* *Feature Engineering:* Computes explicit automotive metrics including a Power-to-Weight/Efficiency Index ('horsepower / city-mpg') and a true harmonic Combined MPG.
* *Feature Selection:* Evaluates full-spectrum correlation matrices using absolute values ('.abs() > 0.5') to preserve strong inverse relationships (e.g., high mileage driving prices down) that standard positive filters skip.
* *Model Baseline:* Trains a regularized linear 'Ridge' regression estimator that optimizes relative percentage error uniform across all price brackets rather than focusing purely on raw distance metrics.

# Setup & Requirements

Make sure you have the required packages installed before running the notebook:


pip install pandas numpy scikit-learn matplotlib seaborn gradio
