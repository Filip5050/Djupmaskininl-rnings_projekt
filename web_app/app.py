import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import torch
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from src.config import Config

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

config = Config()

@st.cache_resource
def load_models_and_results():
    """Load trained PyTorch models, preprocessors, training and evaluation results"""
    try:
        from src.model_builder import load_model
        import joblib
        
        model_path = config.MODELS_DIR / 'fraud_model.pt'
        results_path = config.MODELS_DIR / 'training_results.pkl'
        eval_path = config.MODELS_DIR / 'evaluation_results.pkl'
        preprocessor_path = config.MODELS_DIR / 'preprocessor.pkl'
        
        model = None
        results = None
        eval_results = None
        preprocessor = None
        error_msg = None
        
        if model_path.exists():

            model = load_model(model_path, len(Config.FEATURE_COLS))
            model.eval()
            st.sidebar.success("PyTorch Model loaded!")
        else:
            error_msg = "Model not found"
            
        if results_path.exists():
            results = joblib.load(str(results_path))
            st.sidebar.success("Training results loaded!")
        else:
            error_msg = "Training results not found"
        
        if eval_path.exists():
            eval_results = joblib.load(str(eval_path))
            st.sidebar.success("Evaluation results loaded!")
            
        if preprocessor_path.exists():
            preprocessor_data = joblib.load(str(preprocessor_path))
            preprocessor = preprocessor_data['scaler']
            st.sidebar.success("Preprocessor loaded!")
        
        return model, results, eval_results, preprocessor, error_msg
        
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"

model, results, eval_results, preprocessor, error_msg = load_models_and_results()

st.title("Credit Card Fraud Detection Dashboard")
st.markdown("Real-time fraud detection powered by deep learning")
st.markdown("---")


if error_msg:
    st.warning(error_msg)
    st.info("Run: `python main.py train` to train the model")
else:
    st.success("System operational - Ready to detect fraud!")

st.sidebar.header("Model Info")
if results:
    st.sidebar.metric("Training Date", results.get('training_date', 'N/A'))
    st.sidebar.metric("Val PR-AUC", f"{results.get('val_pr_auc', 0):.4f}")
    if eval_results:
        st.sidebar.metric("Test PR-AUC", f"{eval_results.get('pr_auc', 0):.4f}")
        st.sidebar.metric("Test ROC-AUC", f"{eval_results.get('roc_auc', 0):.4f}")
    st.sidebar.metric("Training Time", results.get('training_time', 'N/A'))


tab1, tab2= st.tabs(["Overview", "Fraud Detection"])

with tab1:
    st.header("Training Overview")
    
    if results:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Training Samples", f"{results.get('total_train_samples', 0):,}")
        with col2:
            test_pr_auc = eval_results.get('pr_auc', 0) if eval_results else 0
            st.metric("Test PR-AUC (Primary)", f"{test_pr_auc:.4f}", help="Precision-Recall AUC on test set")
        with col3:
            test_roc_auc = eval_results.get('roc_auc', 0) if eval_results else 0
            st.metric("Test ROC-AUC", f"{test_roc_auc:.4f}", help="ROC AUC on test set")
        with col4:
            test_accuracy = eval_results.get('accuracy', 0) if eval_results else 0
            st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
        with col5:
            epochs = results.get('epochs_trained', 0)
            st.metric("Epochs Trained", epochs)
        
        # Training History
        st.subheader("Training History")
        
        history = results.get('history', {})
        
        if history:
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy plot
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=history['accuracy'],
                    name='Train Accuracy',
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig_acc.add_trace(go.Scatter(
                    y=history['val_accuracy'],
                    name='Val Accuracy',
                    mode='lines',
                    line=dict(color='red')
                ))
                fig_acc.update_layout(
                    title="Model Accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # Loss plot
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=history['loss'],
                    name='Train Loss',
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig_loss.add_trace(go.Scatter(
                    y=history['val_loss'],
                    name='Val Loss',
                    mode='lines',
                    line=dict(color='red')
                ))
                fig_loss.update_layout(
                    title="Model Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            # PR-AUC plot (primary metric)
            if 'pr_auc' in history:
                fig_pr_auc = go.Figure()
                fig_pr_auc.add_trace(go.Scatter(
                    y=history['pr_auc'],
                    name='Train PR-AUC',
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig_pr_auc.add_trace(go.Scatter(
                    y=history['val_pr_auc'],
                    name='Val PR-AUC',
                    mode='lines',
                    line=dict(color='red')
                ))
                fig_pr_auc.update_layout(
                    title="PR-AUC (Precision-Recall) - Primary Metric for Imbalanced Data",
                    xaxis_title="Epoch",
                    yaxis_title="PR-AUC",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_pr_auc, use_container_width=True)
            
            # ROC-AUC plot (secondary metric)
            if 'auc' in history:
                fig_auc = go.Figure()
                fig_auc.add_trace(go.Scatter(
                    y=history['auc'],
                    name='Train ROC-AUC',
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig_auc.add_trace(go.Scatter(
                    y=history['val_auc'],
                    name='Val ROC-AUC',
                    mode='lines',
                    line=dict(color='red')
                ))
                fig_auc.update_layout(
                    title="ROC-AUC (Secondary Metric)",
                    xaxis_title="Epoch",
                    yaxis_title="ROC-AUC",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_auc, use_container_width=True)
        
        # Class Distribution
        st.subheader("Training Data Distribution")
        class_dist = results.get('class_distribution', {})
        if class_dist:
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=list(class_dist.keys()),
                    y=list(class_dist.values()),
                    marker_color=['green', 'red'],
                    text=list(class_dist.values()),
                    textposition='auto'
                )
            ])
            fig_dist.update_layout(
                title="Training Data Class Distribution (Weighted Sampling Applied)",
                xaxis_title="Class",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Confusion Matrix from Test Data
        st.subheader("Confusion Matrix (Test Set)")
        
        from src.config import Config
        threshold = Config.THRESHOLD
        
        try:
            import joblib
            from sklearn.metrics import confusion_matrix
            
            test_data_path = config.MODELS_DIR / 'test_data.pkl'
            if test_data_path.exists():
                # test_data is saved as tuple (X_test, y_test) in train.py
                X_test, y_test = joblib.load(str(test_data_path))
                
                # Predict on test set with PyTorch
                X_test_tensor = torch.FloatTensor(X_test)
                with torch.no_grad():
                    y_pred_prob = model(X_test_tensor).numpy().flatten()
                y_pred = (y_pred_prob > threshold).astype(int)
                
                # Calculate confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Create heatmap
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Predicted Normal', 'Predicted Fraud'],
                    y=['Actual Normal', 'Actual Fraud'],
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 20},
                    colorscale='Blues',
                    showscale=False
                ))
                
                fig_cm.update_layout(
                    title="Confusion Matrix",
                    xaxis_title="Predicted Label",
                    yaxis_title="Actual Label",
                    height=400
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Show metrics
                tn, fp, fn, tp = cm.ravel()
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("True Negatives", f"{tn:,}", help="Correctly identified legitimate transactions")
                with col2:
                    st.metric("False Positives", f"{fp:,}", help="Legitimate flagged as fraud", delta=f"-{fp}", delta_color="inverse")
                with col3:
                    st.metric("False Negatives", f"{fn:,}", help="Fraud missed by model", delta=f"-{fn}", delta_color="inverse")
                with col4:
                    st.metric("True Positives", f"{tp:,}", help="Correctly detected fraud", delta=f"+{tp}", delta_color="normal")
                
                # Calculate additional metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                st.write("---")
                st.subheader("Performance Metrics")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{accuracy*100:.2f}%", help="Overall correctness")
                with col2:
                    st.metric("Precision", f"{precision*100:.2f}%", help="Of predicted fraud, how many were correct?")
                with col3:
                    st.metric("Recall", f"{recall*100:.2f}%", help="Of actual fraud, how many did we catch?")
                with col4:
                    st.metric("F1-Score", f"{f1:.4f}", help="Harmonic mean of precision and recall")
                with col5:
                    st.metric("FPR", f"{fpr*100:.2f}%", help="False Positive Rate - % of normal flagged as fraud")
                
                st.write("---")
                st.subheader("Business Impact Analysis")
                
                avg_fraud_amount = 122  # From dataset analysis
                check_cost = 2
                
                # Realistic calculation: ALL flagged transactions must be checked (TP + FP)
                total_flagged = tp + fp
                fraud_saved = tp * avg_fraud_amount
                fraud_lost = fn * avg_fraud_amount
                check_costs = total_flagged * check_cost  
                net_benefit = fraud_saved - fraud_lost - check_costs
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Fraud Prevented", f"${fraud_saved:,.0f}", help=f"Caught {tp} frauds × ${avg_fraud_amount}")
                with col2:
                    st.metric("Fraud Missed", f"${fraud_lost:,.0f}", help=f"Missed {fn} frauds × ${avg_fraud_amount}", delta=f"-${fraud_lost:,.0f}", delta_color="inverse")
                with col3:
                    st.metric("Check Costs", f"${check_costs:,.0f}", help=f"{total_flagged} flagged transactions (TP+FP) × ${check_cost}")
                with col4:
                    st.metric("Net Benefit", f"${net_benefit:,.0f}", help="Saved - Lost - Costs", delta=f"+${net_benefit:,.0f}" if net_benefit > 0 else f"${net_benefit:,.0f}", delta_color="normal" if net_benefit > 0 else "inverse")
                
                
            else:
                st.warning("Test data not found. Run evaluation first: `python main.py evaluate`")
        
        except Exception as e:
            st.error(f"Error loading confusion matrix: {str(e)}")
    else:
        st.error("Train the model first: `python main.py train`")

# Tab 2: Fraud Detection
with tab2:
    if model and preprocessor:
        st.subheader("Single Transaction Testing")
        st.info("V1-V28 are PCA-transformed features (anonymized credit card data). Use preset examples below!")
        
        # Quick test presets
        st.write("**Load Test Transaction:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("Normal Transaction", use_container_width=True):
                st.session_state['preset'] = 'normal'
                st.rerun()
        
        with preset_col2:
            if st.button("Fraud Transaction", use_container_width=True):
                st.session_state['preset'] = 'fraud'
                st.rerun()
        
        with preset_col3:
            if st.button("Random from Dataset", use_container_width=True):
                st.session_state['preset'] = 'random'
                st.rerun()
        
        st.write("---")
        
        # Determine which transaction to use
        transaction = None
        actual_label = None
        
        if 'preset' in st.session_state and st.session_state['preset']:
            try:
                data_file = config.DATA_FILE
                if not data_file.exists():
                    st.error("Dataset not found!")
                    transaction = None
                else:
                    # Load dataset sample
                    df_sample = pd.read_csv(data_file, nrows=100000)
                    
                    if st.session_state['preset'] == 'normal':
                        # Select random NORMAL transaction
                        normal_df = df_sample[df_sample['Class'] == 0]
                        if len(normal_df) > 0:
                            random_row = normal_df.sample(n=1).iloc[0]
                            actual_label = 0
                            transaction = random_row.drop('Class').values.reshape(1, -1)
                            st.info(f"Loaded: **Random Normal Transaction** (Actual: LEGITIMATE)")
                        else:
                            st.error("No normal transactions found!")
                    
                    elif st.session_state['preset'] == 'fraud':
                        # Select random FRAUD transaction
                        fraud_df = df_sample[df_sample['Class'] == 1]
                        if len(fraud_df) > 0:
                            random_row = fraud_df.sample(n=1).iloc[0]
                            actual_label = 1
                            transaction = random_row.drop('Class').values.reshape(1, -1)
                            st.info(f"Loaded: **Random Fraud Transaction** (Actual: FRAUD)")
                        else:
                            st.error("No fraud transactions found!")
                    
                    elif st.session_state['preset'] == 'random':
                        # Select 50/50 between fraud and normal (better for demo than 0.17% fraud!)
                        coin_flip = np.random.randint(0, 2)  # 0 or 1
                        
                        if coin_flip == 0:
                            # Select normal
                            normal_df = df_sample[df_sample['Class'] == 0]
                            if len(normal_df) > 0:
                                random_row = normal_df.sample(n=1).iloc[0]
                                actual_label = 0
                            else:
                                random_row = df_sample.sample(n=1).iloc[0]
                                actual_label = int(random_row['Class'])
                        else:
                            # Select fraud
                            fraud_df = df_sample[df_sample['Class'] == 1]
                            if len(fraud_df) > 0:
                                random_row = fraud_df.sample(n=1).iloc[0]
                                actual_label = 1
                            else:
                                random_row = df_sample.sample(n=1).iloc[0]
                                actual_label = int(random_row['Class'])
                        
                        transaction = random_row.drop('Class').values.reshape(1, -1)
                        st.info(f"Loaded: **Random Transaction (50/50 split)** (Actual: {'FRAUD' if actual_label == 1 else 'LEGITIMATE'})")
            
            except Exception as e:
                st.error(f"Error loading transaction: {str(e)}")
                st.exception(e)
                transaction = None
        
        # Show transaction if loaded
        if transaction is not None:
            # Key fraud indicators explanation
            st.info("""
            **Key Fraud Indicators (Based on Permutation Importance Analysis):**
            - **V14** (Most important, 1.78% PR-AUC drop when shuffled): Fraud shows extreme negative values (< -6)
            - **V4** (Second most important, 1.72% PR-AUC drop): Fraud displays unusual deviations from normal ≈ 0
            - **V8** (Third most important, 0.56% PR-AUC drop): Fraud exhibits anomalous patterns
            
            These PCA-transformed features capture hidden patterns in transaction behavior that are strong predictors of fraud.
            """)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Time (seconds since start)", f"{transaction[0][0]:.0f}s")
                st.metric("Amount", f"${transaction[0][29]:.2f}")
            with col2:
                # V14 is at index 14 (Time=0, V1=1, V2=2, ..., V14=14)
                v14_value = transaction[0][14]
                st.metric("V14 (Strongest Indicator)", f"{v14_value:.3f}",
                         delta="High Risk" if v14_value < -6 else "Normal",
                         delta_color="inverse" if v14_value < -6 else "normal")
                # V4 is at index 4
                v4_value = transaction[0][4]
                st.metric("V4 (2nd Strongest)", f"{v4_value:.3f}",
                         delta="Suspicious" if abs(v4_value) > 3 else "Normal",
                         delta_color="inverse" if abs(v4_value) > 3 else "normal")
            with col3:
                # V8 is at index 8
                v8_value = transaction[0][8]
                st.metric("V8 (3rd Strongest)", f"{v8_value:.3f}",
                         delta="Anomaly" if abs(v8_value) > 2 else "Normal",
                         delta_color="inverse" if abs(v8_value) > 2 else "normal")
            
            # Predict button
            if st.button("Check for Fraud", type="primary", use_container_width=True):
                try:
                    import joblib
                    
                    preprocessor_path = config.MODELS_DIR / 'preprocessor.pkl'
                    
                    if preprocessor_path.exists():
                        preprocessor_data = joblib.load(str(preprocessor_path))
                        scaler = preprocessor_data['scaler']

                        time_amount = transaction[:, [0, 29]]  
                        
                        time_amount_scaled = scaler.transform(time_amount)  # Shape: (1, 2)
                        
                        transaction_scaled = transaction.copy()
                        transaction_scaled[:, 0] = time_amount_scaled[:, 0]   
                        transaction_scaled[:, 29] = time_amount_scaled[:, 1]  

                        transaction_tensor = torch.FloatTensor(transaction_scaled)
                        with torch.no_grad():
                            prediction = model(transaction_tensor).numpy()
                        fraud_probability = float(prediction[0][0])

                        st.write(f"Fraud probability: {fraud_probability:.4f} ({fraud_probability*100:.2f}%)")
                        st.write(f"Amount: ${transaction[0][29]:.2f} | ⏱️ Time: {int(transaction[0][0])}s since first transaction")

                        st.write("---")
                        

                        st.subheader("Analyzing Transaction...")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        import time
                        
     
                        for i in range(101):
                            progress_bar.progress(i)
                            status_text.text(f"Processing... {i}%")
                            time.sleep(0.01)  
                        
                        status_text.text("✅ Analysis Complete!")
                        time.sleep(0.3)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.write("---")

                        gauge_placeholder = st.empty()
                        
                        steps = 20
                        for step in range(steps + 1):
                            current_value = (fraud_probability * 100 * step) / steps
                            
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=current_value,
                                number={'suffix': "%", 'font': {'size': 40}},
                                title={'text': "Fraud Probability", 'font': {'size': 24}},
                                gauge={
                                    'axis': {'range': [0, 100], 'tickwidth': 2},
                                    'bar': {'color': "darkred" if current_value > 50 else "darkgreen", 'thickness': 0.7},
                                    'steps': [
                                        {'range': [0, 25], 'color': "rgba(144, 238, 144, 0.3)"},
                                        {'range': [25, 60], 'color': "rgba(255, 255, 0, 0.3)"},
                                        {'range': [60, 100], 'color': "rgba(240, 128, 128, 0.3)"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': Config.THRESHOLD * 100
                                    }
                                }
                            ))
                            fig.update_layout(
                                height=350,
                                margin=dict(l=20, r=20, t=50, b=20)
                            )
                            gauge_placeholder.plotly_chart(fig, use_container_width=True, key=f"gauge_{step}")
                            time.sleep(0.05)  
                        
                        # Result
                        predicted_class = 1 if fraud_probability > 0.5 else 0
                        
                        st.write("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if predicted_class == 1:
                                st.error(f"**FRAUD DETECTED!**")
                                st.error(f"Confidence: {fraud_probability*100:.2f}%")
                            else:
                                st.success(f"**LEGITIMATE TRANSACTION**")
                                st.success(f"Confidence: {(1-fraud_probability)*100:.2f}%")
                        
                        with col2:
                            if actual_label is not None:
                                if predicted_class == actual_label:
                                    st.success("**CORRECT PREDICTION!**")
                                else:
                                    st.error("**INCORRECT PREDICTION**")
                                
                                st.write(f"**Actual:** {'FRAUD' if actual_label == 1 else 'LEGITIMATE'}")
                                st.write(f"**Predicted:** {'FRAUD' if predicted_class == 1 else 'LEGITIMATE'}")
                    
                    else:
                        st.error("Preprocessor not found!")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.exception(e)
        
        else:
            st.info("Click a button above to load a transaction, or enter values manually below")
        
