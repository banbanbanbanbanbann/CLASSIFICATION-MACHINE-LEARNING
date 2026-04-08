import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import shap
import joblib
import os
import xgboost as xgb
import tensorflow as tf
import keras_tuner as kt
import warnings
import re
import tempfile
from sklearn.exceptions import ConvergenceWarning
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, f1_score, average_precision_score,
    confusion_matrix, balanced_accuracy_score, brier_score_loss,
    precision_score, recall_score, precision_recall_curve
)
from sklearn.calibration import CalibrationDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Suppress ConvergenceWarnings during Logistic Regression Early Stopping
warnings.filterwarnings("ignore", category=ConvergenceWarning)

st.set_page_config(page_title="Multi-Model Pipeline", layout="wide")

# Vocaloid Hex Colors
miku_turquoise = "#39C5BB"
luka_pink = "#FFB8D0"
teto_red = "#D43654"

st.title("CLASSIFICATION(ONLY) ML Pipeline")

# ==========================================
# 0. APP MODE SELECTION
# ==========================================
app_mode = st.radio(
    "Choose Workflow:", 
    ["🏋️ Train New Model", " Load Saved Model"], 
    horizontal=True
)
st.divider()

# ==========================================
# 1. THE HEAVY LIFTING (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Training model with Early Stopping... This might take a while!")
def train_pipeline(df, model_choice):
    X = df.drop('target', axis=1)
    y = df['target']

    # --- THE 4 SPLITS ARCHITECTURE ---
    # 1. Separate unseen TEST set (20%)
    X_temp1, X_test, y_temp1, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # 2. Separate VALIDATION set for hyperparameter tuning (20% overall)
    X_temp2, X_val, y_temp2, y_val = train_test_split(X_temp1, y_temp1, stratify=y_temp1, test_size=0.25, random_state=42)
    # 3. Separate EARLY STOPPING set (12% overall) from the TRAINING set (48% overall)
    X_train, X_early_stop, y_train, y_early_stop = train_test_split(X_temp2, y_temp2, stratify=y_temp2, test_size=0.2, random_state=42)

    # Preprocessing
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Create a sub-pipeline for numeric data (Fills missing with the median, then scales)
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Create a sub-pipeline for categorical data (Fills missing with the word 'missing', then encodes)
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ], remainder='passthrough'
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_early_stop_processed = preprocessor.transform(X_early_stop)
    X_val_processed = preprocessor.transform(X_val) 
    X_test_processed = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out()
    # XGBoost hates [, ], and < in column names. Replace them with underscores.
    safe_feature_names = [re.sub(r'[\[\]<]', '_', name) for name in feature_names]

    X_train_processed = pd.DataFrame(X_train_processed, columns=safe_feature_names, index=X_train.index)
    X_early_stop_processed = pd.DataFrame(X_early_stop_processed, columns=safe_feature_names, index=X_early_stop.index)
    X_test_processed = pd.DataFrame(X_test_processed, columns=safe_feature_names, index=X_test.index)
    X_val_processed = pd.DataFrame(X_val_processed, columns=safe_feature_names, index=X_val.index)

    # Label Encoding
    le = LabelEncoder()
    y_train_processed = pd.Series(le.fit_transform(y_train), index=y_train.index)
    y_early_stop_processed = pd.Series(le.transform(y_early_stop), index=y_early_stop.index)
    y_val_processed = pd.Series(le.transform(y_val), index=y_val.index)
    y_test_processed = pd.Series(le.transform(y_test), index=y_test.index)

    best_val_auprc = -1
    best_model = None
    
# ---------------------------------------------------------
    # MODEL: LOGISTIC REGRESSION (With Cross Validation)
    # ---------------------------------------------------------
    if model_choice == "Logistic Regression":
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
        
        for params in ParameterGrid(param_grid):
            # Train strictly on the 48% Train split
            lr = LogisticRegression(**params, class_weight='balanced', random_state=42, solver='liblinear')
            lr.fit(X_train_processed, y_train_processed)
            
            # Grade hyperparameters using the 12% Early Stop / Tuning split
            tune_auprc = average_precision_score(y_early_stop_processed, lr.predict_proba(X_early_stop_processed)[:, 1])
            
            if tune_auprc > best_val_auprc:
                best_val_auprc = tune_auprc
                best_model = lr

    # ---------------------------------------------------------
    # MODEL: RANDOM FOREST
    # ---------------------------------------------------------
    elif model_choice == "Random Forest":
        param_grid = {
            'max_depth': [5, 10, None], 
            'min_samples_leaf': [1, 4], 
            'min_samples_split': [2, 10]
        }
        
        for params in ParameterGrid(param_grid):
            # Train strictly on the 48% Train split
            rf = RandomForestClassifier(
                **params, 
                n_estimators=200, 
                class_weight='balanced', 
                random_state=42, 
                n_jobs=-1
            )
            rf.fit(X_train_processed, y_train_processed)
            
            # Grade hyperparameters using the 12% Early Stop / Tuning split
            tune_auprc = average_precision_score(y_early_stop_processed, rf.predict_proba(X_early_stop_processed)[:, 1])
            
            if tune_auprc > best_val_auprc:
                best_val_auprc = tune_auprc
                best_model = rf
    # ---------------------------------------------------------
    # MODEL: XGBOOST (Native Early Stopping)
    # ---------------------------------------------------------
    elif model_choice == "XGBoost":
        neg_count = (y_train_processed == 0).sum()
        pos_count = (y_train_processed == 1).sum()
        spw = neg_count / pos_count if pos_count > 0 else 1

        param_grid = {
            'max_depth': [3, 5, 7], 
            'learning_rate': [0.01, 0.05, 0.1]
        }
        for params in ParameterGrid(param_grid):
            xgb_model = xgb.XGBClassifier(
                **params, n_estimators=500, scale_pos_weight=spw, 
                random_state=42, eval_metric='aucpr', early_stopping_rounds=15, 
                tree_method="hist", n_jobs=-1
            )
            # Evaluate against the early_stopping split, NOT the validation split!
            xgb_model.fit(X_train_processed, y_train_processed, eval_set=[(X_early_stop_processed, y_early_stop_processed)], verbose=False)
            
            tune_auprc = average_precision_score(y_early_stop_processed, xgb_model.predict_proba(X_early_stop_processed)[:, 1])
            if tune_auprc > best_val_auprc:
                best_val_auprc = tune_auprc
                best_model = xgb_model

    # ---------------------------------------------------------
    # MODEL: NEURAL NETWORK (Keras Tuner)
    # ---------------------------------------------------------
    elif model_choice == "Neural Network (Focal Loss)":
        neg_count = (y_train_processed == 0).sum()
        pos_count = (y_train_processed == 1).sum()
        initial_bias = np.log([pos_count / neg_count]) if pos_count > 0 else [0]
        output_bias = tf.keras.initializers.Constant(initial_bias)

        classes = np.unique(y_train_processed)
        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_processed)
        global_class_weights = {classes[i]: weights[i] for i in range(len(classes))}

        def build_dense_model(hp):
            input_layer = Input(shape=(X_train_processed.shape[1],))
            x = input_layer
            hp_neurons = hp.Int('neurons', min_value=64, max_value=256, step=64)
            hp_dropout_1 = hp.Float('dropout_1', min_value=0.3, max_value=0.5, step=0.1)
            hp_dropout_rest = hp.Float('dropout_rest', min_value=0.2, max_value=0.4, step=0.1)

            x = Dense(hp_neurons, activation='swish')(x)
            x = BatchNormalization()(x)
            x = Dropout(hp_dropout_1)(x)

            for i in range(hp.Int('num_layers', 1, 3)):
                x = Dense(hp_neurons // (2**(i+1)), activation='swish')(x)
                x = BatchNormalization()(x)
                x = Dropout(hp_dropout_rest)(x)

            output = Dense(1, activation='sigmoid', bias_initializer=output_bias)(x)
            model = tf.keras.Model(inputs=input_layer, outputs=output)
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])
            
            focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
                gamma=2.0, alpha=0.25, label_smoothing=0.1, apply_class_balancing=False
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=focal_loss,
                metrics=[tf.keras.metrics.AUC(curve='PR', name='auprc')]
            )
            return model

        tuner = kt.RandomSearch(
            build_dense_model, 
            objective=kt.Objective("val_auprc", direction="max"), 
            max_trials=10, 
            directory=tempfile.gettempdir(), # <-- Sends the junk to your OS's hidden temp folder!
            project_name='reactive_dense', 
            overwrite=True 
        )

        early_stop = EarlyStopping(monitor='val_auprc', mode='max', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_auprc', factor=0.2, patience=3, min_lr=1e-6, mode='max')

        # Tuner now uses X_early_stop so X_val remains untainted!
        tuner.search(
            X_train_processed.values, y_train_processed.values, epochs=30, batch_size=512,
            class_weight=global_class_weights, 
            validation_data=(X_early_stop_processed.values, y_early_stop_processed.values), 
            callbacks=[early_stop, reduce_lr], verbose=0
        )
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.predict_proba = lambda X: np.hstack([1 - best_model.predict(X, verbose=0), best_model.predict(X, verbose=0)])

    # ---------------------------------------------------------
    # EVALUATION & THRESHOLD MOVING (On Validation Split)
    # ---------------------------------------------------------

    # 1. Generate probabilities for the Validation Set
    # We use the Validation set ONLY here to avoid "double-dipping" 
    # (using the same data to tune hyperparameters and thresholds).
    if model_choice == "Neural Network (Focal Loss)":
        # Ensure we use .values for Keras to avoid feature name mismatch warnings
        val_probs = best_model.predict_proba(X_val_processed.values)[:, 1]
    else:
        val_probs = best_model.predict_proba(X_val_processed)[:, 1]

    # 2. Calculate Precision-Recall Curve
    precisions_val, recalls_val, thresholds_val = precision_recall_curve(y_val_processed, val_probs)

    # 3. Calculate F1 Scores for every threshold
    # We add a tiny epsilon (1e-8) to prevent division by zero errors
    f1_scores_val = 2 * (precisions_val * recalls_val) / (precisions_val + recalls_val + 1e-8)

    # 4. Find the optimal threshold
    # Slicing [:-1] is critical because thresholds_val is 1 element shorter than precision/recall
    best_threshold = thresholds_val[np.argmax(f1_scores_val[:-1])]

    # ---------------------------------------------------------
    # FINAL TEST SET EVALUATION (The "Golden" Evaluation)
    # ---------------------------------------------------------

    # 1. Generate probabilities for the Test Set
    if model_choice == "Neural Network (Focal Loss)":
        test_probs = best_model.predict_proba(X_test_processed.values)[:, 1]
    else:
        test_probs = best_model.predict_proba(X_test_processed)[:, 1]

    # 2. Apply the "Validation-Optimized" threshold to the Test Set
    optimal_preds_test = (test_probs >= best_threshold).astype(int)

    # 3. Consolidate Metrics
    # This provides a 360-degree view of how the model performs on truly unseen data
    metrics = {
        "Best Threshold": best_threshold,
        "F1 Score": f1_score(y_test_processed, optimal_preds_test),
        "AUPRC": average_precision_score(y_test_processed, test_probs),
        "Precision": precision_score(y_test_processed, optimal_preds_test, zero_division=0),
        "Recall": recall_score(y_test_processed, optimal_preds_test),
        "Balanced Accuracy": balanced_accuracy_score(y_test_processed, optimal_preds_test),
        "ROC AUC": roc_auc_score(y_test_processed, test_probs)
    }

    # ---------------------------------------------------------
    # SHAP FEATURE IMPORTANCE
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # SHAP FEATURE IMPORTANCE
    # ---------------------------------------------------------
    shap_values = None
    if model_choice in ["Logistic Regression"]:
        explainer = shap.LinearExplainer(best_model, X_train_processed)
        shap_values = explainer(X_test_processed)
        
    elif model_choice in ["XGBoost", "Random Forest"]:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer(X_test_processed)
        
        # --- FIX FOR RANDOM FOREST 3D SHAP VALUES ---
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
            
    elif model_choice == "Neural Network (Focal Loss)":
        # 1.Use K-Means to summarize the background into just 10 representative rows!
        # This makes the explainer drastically faster while maintaining high accuracy.
        background = shap.kmeans(X_train_processed, 10)
        
        # 2. Create a safe prediction function
        predict_fn = lambda x: best_model.predict_proba(x if isinstance(x, np.ndarray) else x.values)[:, 1]
        
        # 3. Initialize the heavy-duty KernelExplainer
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # 4. Take a small 50-row sample of the test set for speed
        test_sample = shap.sample(X_test_processed, 50)
        
        # 5. Calculate the raw SHAP values
        shap_values_raw = explainer.shap_values(test_sample)
        
        # 6. Convert the raw numpy array back into a SHAP Explanation object 
        shap_values = shap.Explanation(
            values=shap_values_raw,
            base_values=explainer.expected_value,
            data=test_sample,
            feature_names=test_sample.columns.tolist()
        )

    #  ADD THIS RETURN STATEMENT BACK IN! 
    return (best_model, preprocessor, le, best_threshold, metrics, 
            thresholds_val, f1_scores_val, precisions_val, recalls_val, 
            X_test_processed, y_test_processed, shap_values, X, test_probs)

# ==========================================
# 2. WORKFLOW: TRAIN NEW MODEL
# ==========================================
if app_mode == "🏋️ Train New Model":
    
    selected_model = st.selectbox(
        " CHOOSE YOUR MODEL!", 
        ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network (Focal Loss)"]
    )
    
    st.write("Upload a CSV file with your data. Ensure your target variable is named **`target`**.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'target' not in df.columns:
            st.error("Error: Could not find a column named 'target'. Please rename your label column and try again.")
        else:
            (model, preprocessor, le, threshold, metrics, 
             thresholds_val, f1_scores_val, precisions_val, recalls_val, 
             X_test_processed, y_test_processed, shap_values, raw_X, test_probs) = train_pipeline(df, selected_model)

            st.success(f"**{selected_model}** trained successfully!")

            # --- Show Metrics ---
            st.subheader("Test Set Performance")
            cols = st.columns(5)
            cols[0].metric("F1-Score", f"{metrics['F1 Score']:.3f}")
            cols[1].metric("AUPRC", f"{metrics['AUPRC']:.3f}")
            cols[2].metric("Precision", f"{metrics['Precision']:.3f}")
            cols[3].metric("Recall", f"{metrics['Recall']:.3f}")
            cols[4].metric("Bal. Accuracy", f"{metrics['Balanced Accuracy']:.3f}")
            
            target_classes = le.classes_
            st.info(f"**Target Mapping:** {target_classes[0]} -> 0 | {target_classes[1]} -> 1. **Optimal Threshold:** {threshold:.3f}")

            # --- Visualizations ---
            st.subheader("Visualizations")
            tab1, tab2, tab3 = st.tabs(["Threshold Tuning", "Calibration", "SHAP Importance"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(thresholds_val, f1_scores_val[:-1], label="F1 Score", color=miku_turquoise, linewidth=2.5)
                ax.plot(thresholds_val, precisions_val[:-1], label="Precision", color=luka_pink, linestyle="--", linewidth=2)
                ax.plot(thresholds_val, recalls_val[:-1], label="Recall", color=teto_red, linestyle="--", linewidth=2)
                ax.axvline(x=threshold, color='#31333F', linestyle=':', label=f'Optimal ({threshold:.3f})')
                ax.set_title('Validation Metrics across Thresholds', color=miku_turquoise)
                ax.legend()
                ax.grid(True, alpha=0.2, color=miku_turquoise)
                st.pyplot(fig)

            with tab2:
                fig, ax = plt.subplots(figsize=(6, 6))
                CalibrationDisplay.from_predictions(y_test_processed, test_probs, n_bins=10, ax=ax, color=miku_turquoise)
                ax.set_title("Calibration Curve", color=miku_turquoise)
                ax.grid(True, alpha=0.2, color=miku_turquoise)
                st.pyplot(fig)
                
            with tab3:
                if shap_values is not None:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    vocaloid_cmap = mcolors.LinearSegmentedColormap.from_list("Vocaloid", [miku_turquoise, luka_pink, teto_red])
                    shap.plots.beeswarm(shap_values, max_display=10, show=False, color=vocaloid_cmap)
                    st.pyplot(fig)
                else:
                    st.warning("SHAP values are hidden for Neural Networks to prevent long rendering times.")

            st.divider()

            # --- OPTIONAL: SAVE MODEL TO FOLDER ---
            st.header(" Save Model Pipeline")
            save_path = st.text_input("Enter folder path to save (e.g., ./saved_models/miku_run):", value="./miku_model")
            
            if st.button("Create Folder & Save Files"):
                try:
                    os.makedirs(save_path, exist_ok=True)
                    if selected_model == "Neural Network (Focal Loss)":
                        model.save(os.path.join(save_path, 'keras_model.h5'))
                    else:
                        joblib.dump(model, os.path.join(save_path, 'model.pkl'))
                        
                    joblib.dump(preprocessor, os.path.join(save_path, 'preprocessor.pkl'))
                    joblib.dump(le, os.path.join(save_path, 'label_encoder.pkl'))
                    
                    with open(os.path.join(save_path, 'optimal_threshold.txt'), 'w') as f:
                        f.write(str(threshold))
                        
                    with open(os.path.join(save_path, 'model_type.txt'), 'w') as f:
                        f.write(selected_model)
                        
                    st.success(f"Successfully created folder `{save_path}` and saved all files!")
                except Exception as e:
                    st.error(f"Error saving files: {e}")

# ==========================================
# 3. WORKFLOW: LOAD SAVED MODEL
# ==========================================
else:
    st.header("📂 Load Pre-Trained Pipeline")
    
    col1, col2 = st.columns(2)
    model_file = col1.file_uploader("1. Upload Model (.pkl or .h5)", type=["pkl", "h5"])
    prep_file = col2.file_uploader("2. Upload Preprocessor (.pkl)", type="pkl")
    le_file = col1.file_uploader("3. Upload Label Encoder (.pkl)", type="pkl")
    thresh_file = col2.file_uploader("4. Upload Threshold (.txt)", type="txt")
    type_file = st.file_uploader("5. Upload Model Type (.txt)", type="txt")

    if model_file and prep_file and le_file and thresh_file and type_file:
        selected_model = type_file.read().decode("utf-8").strip()
        
        if selected_model == "Neural Network (Focal Loss)":
            with open("temp_keras.h5", "wb") as f:
                f.write(model_file.getbuffer())
            model = tf.keras.models.load_model(
                "temp_keras.h5", 
                custom_objects={"BinaryFocalCrossentropy": tf.keras.losses.BinaryFocalCrossentropy}
            )
            model.predict_proba = lambda X: np.hstack([1 - model.predict(X, verbose=0), model.predict(X, verbose=0)])
        else:
            model = joblib.load(model_file)
            
        preprocessor = joblib.load(prep_file)
        le = joblib.load(le_file)
        threshold = float(thresh_file.read().decode("utf-8").strip())
        target_classes = le.classes_

        st.success(f"**{selected_model}** and Preprocessor loaded! (Threshold: {threshold:.3f})")
        st.divider()

# ==========================================
# 4. DYNAMIC PREDICTION FORM
# ==========================================
if ('model' in locals() and 'preprocessor' in locals()):
    st.header(" Make a Prediction")

    user_input_data = {}
    with st.form("prediction_form"):
        form_cols = st.columns(2)
        idx = 0
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num':
                for col in cols:
                    ui_col = form_cols[idx % 2]
                    user_input_data[col] = ui_col.number_input(f"{col}", value=0.0)
                    idx += 1
            elif name == 'cat':
                for i, col in enumerate(cols):
                    ui_col = form_cols[idx % 2]
                    
                    # --- Extract the encoder from the pipeline ---
                    # Check if it's a pipeline (has named_steps) or a direct encoder
                    if hasattr(transformer, 'named_steps'):
                        actual_encoder = transformer.named_steps['encoder']
                    else:
                        actual_encoder = transformer
                        
                    unique_vals = actual_encoder.categories_[i]
                    # --------------------------------------------------------
                    
                    user_input_data[col] = ui_col.selectbox(f"{col}", options=unique_vals)
                    idx += 1
        
        submit_button = st.form_submit_button(label="Predict Outcome")

    if submit_button:
        user_df = pd.DataFrame([user_input_data])
        user_processed = preprocessor.transform(user_df)
        
        # --- THE FIX: Clean the feature names exactly like we did in training ---
        raw_feature_names = preprocessor.get_feature_names_out()
        safe_prediction_names = [re.sub(r'[\[\]<]', '_', name) for name in raw_feature_names]
        
        user_processed_df = pd.DataFrame(user_processed, columns=safe_prediction_names)
        # ------------------------------------------------------------------------
        
        if selected_model == "Neural Network (Focal Loss)":
            prob = model.predict_proba(user_processed_df.values)[:, 1][0]
        else:
            prob = model.predict_proba(user_processed_df)[:, 1][0]
            
        st.write(f"**Probability of Class '{target_classes[1]}':** {prob:.2%}")
        st.write(f"*(Decision Threshold used: {threshold:.2%})*")
