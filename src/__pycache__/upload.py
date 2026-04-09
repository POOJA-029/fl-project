import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import warnings

# Ignore Streamlit beta warnings and Torch user warnings
warnings.filterwarnings("ignore")

# Import custom modules
from src.dataset import load_domain_data, partition_data_for_clients, get_dataloaders
from src.model import FederatedNN, evaluate_model
from src.federated import simulate_fl_round
from src.fairness import calculate_demographic_parity, calculate_equal_opportunity, get_fair_dataloader
from src.efficiency import get_actual_size_kb, execute_green_ai_pipeline

st.set_page_config(page_title="Multi-Domain FL Sandbox", layout="wide", page_icon="🌍")

# Styling to make it look premium
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2E86AB; text-align: center; margin-bottom: 0px; }
    .sub-header { text-align: center; color: #6C7A89; font-size: 1.2rem; margin-bottom: 30px; }
    .metric-box { background-color: #F8F9FA; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); border: 1px solid #EAEAEA; }
    div[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #cbd5e1;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Privacy-Preserving, Fair, and Energy-Efficient Federated Learning</div>', unsafe_allow_html=True)

# ====== SIDEBAR ======
st.sidebar.title("🌍 Domain Configuration")
selected_domain = st.sidebar.selectbox(
    "Choose Application Domain",
    ("Healthcare", "Finance", "Benchmark")
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Healthcare**: Heart Disease Dataset\n\n"
    "**Finance**: Credit Default Risk (Credit-g)\n\n"
    "**Benchmark**: Adult Census Income"
)

st.markdown(f'<div class="sub-header">Current Domain: {selected_domain} Application</div>', unsafe_allow_html=True)

# Initialization function
@st.cache_data
def load_data(domain):
    try:
        X, y, protected, df = load_domain_data(domain)
        return X, y, protected, df
    except Exception as e:
        st.error(f"Error loading {domain} data: {str(e)}")
        return None, None, None, None

X, y, protected, df = load_data(selected_domain)

if X is None:
    st.stop()

# Reset state when domain changes
if 'current_domain' not in st.session_state or st.session_state['current_domain'] != selected_domain:
    st.session_state['current_domain'] = selected_domain
    for key in ['trained_model', 'evaluation_data', 'partitions', 'fl_metrics', 'pre_fairness']:
        if key in st.session_state:
            del st.session_state[key]

# Navigation
tabs = st.tabs(["1. Data Overview", "2. Federated FL Training", "3. Bias Mitigation", "4. Green AI (Efficiency)"])

# ====== TAB 1: DATA OVERVIEW ======
with tabs[0]:
    st.markdown(f"### 📊 Dataset Demographics ({selected_domain})")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("**Raw Data Sample:**")
        st.dataframe(df.head(10))
        st.write(f"**Total Samples:** {len(df)}")
        st.write(f"**Total Features:** {X.shape[1]}")
    with col2:
        st.write("**Target & Protected Attribute Distribution:**")
        if selected_domain == "Healthcare":
            x_label = "Protected: Age > 55"
            target_label = "Disease Risk"
        elif selected_domain == "Finance":
            x_label = "Protected: Age > 30"
            target_label = "Default Risk"
        else:
            x_label = "Protected: Male"
            target_label = "Income >50K"
            
        fig = px.histogram(
            df, x="protected_group", color="target", barmode='group',
            labels={'protected_group': x_label, 'target': target_label, 'count': 'Number of Individuals'},
            title=f"Distribution of {target_label} by Protected Attribute"
        )
        newnames = {'0': "Negative Outcome (0)", '1': "Positive Risk (1)"}
        fig.for_each_trace(lambda t: t.update(name = newnames[t.name], legendgroup = newnames[t.name], hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])))

        st.plotly_chart(fig, use_container_width=True)
        
    st.info("💡 **Why this matters:** Data is often inherently biased against certain protected groups. Our FL system will detect and mitigate this inequality.")

# ====== TAB 2: FEDERATED LEARNING ======
with tabs[1]:
    st.markdown("### 🌐 Privacy-Preserving Federated Learning")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("#### Settings")
        num_clients = st.slider("Number of Nodes (Clients)", 2, 10, 3)
        num_rounds = st.slider("Communication Rounds", 1, 20, 5)
        local_epochs = st.slider("Local Epochs per Round", 1, 10, 2)
        
        start_fl = st.button("🚀 Start Federated Training", type="primary")
        
    with col2:
        if start_fl:
            partitions = partition_data_for_clients(X, y, protected, num_clients)
            
            # Unpack test set from Client 0 as Global Evaluation Set
            X_test_global, y_test_global, p_test_global = partitions[0]['test']
            
            dataloaders = [get_dataloaders(p['train'][0], p['train'][1]) for p in partitions]
            
            global_model = FederatedNN(input_dim=X.shape[1])
            
            st.session_state['fl_metrics'] = {'round': [], 'accuracy': []}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            metrics_placeholder = st.empty()
            
            for r in range(num_rounds):
                status_text.text(f"Round {r+1}/{num_rounds} - Aggregating Client Updates...")
                
                # Simulate training + aggregation
                global_model = simulate_fl_round(global_model, dataloaders, client_epochs=local_epochs)
                
                # Evaluate global model
                acc, preds, probs = evaluate_model(global_model, X_test_global, y_test_global)
                
                st.session_state['fl_metrics']['round'].append(r+1)
                st.session_state['fl_metrics']['accuracy'].append(acc)
                
                progress_bar.progress((r + 1) / num_rounds)
                
                # Real-time chart update
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=st.session_state['fl_metrics']['round'], 
                                         y=st.session_state['fl_metrics']['accuracy'], mode='lines+markers', name='Accuracy', marker=dict(color='#2E86AB')))
                fig.update_layout(title=f"Federated Model Accuracy Over Rounds (Current: {acc:.2%})", xaxis_title="Communication Round", yaxis_title="Test Accuracy")
                
                with metrics_placeholder.container():
                    st.plotly_chart(fig, use_container_width=True)
                
            status_text.text("Federated Learning Complete! Model is ready for Fairness testing.")
            st.session_state['trained_model'] = global_model
            st.session_state['evaluation_data'] = (X_test_global, y_test_global, p_test_global)
            st.session_state['partitions'] = partitions
            st.success("✅ Model Trained!")


# ====== TAB 3: FAIRNESS & BIAS MITIGATION ======
with tabs[2]:
    st.markdown("### ⚖️ Bias Detection & Mitigation")
    
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train the Federated Model first in the previous tab.")
    else:
        st.write("Ensuring model predictions are fair across protected sub-populations.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1. Analyze Original Model")
            if st.button("Calculate Fairness Metrics"):
                model = st.session_state['trained_model']
                X_test, y_test, p_test = st.session_state['evaluation_data']
                
                acc, preds, probs = evaluate_model(model, X_test, y_test)
                
                dp_diff = calculate_demographic_parity(preds, p_test)
                eo_diff = calculate_equal_opportunity(y_test, preds, p_test)
                
                st.session_state['pre_fairness'] = {'dp': dp_diff, 'eo': eo_diff, 'acc': acc}
                
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Demographic Parity Difference", f"{dp_diff:.3f}", delta_color="inverse")
                st.caption("Closer to 0.0 is fairer. Represents prediction rate difference between groups.")
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Equal Opportunity Difference", f"{eo_diff:.3f}", delta_color="inverse")
                st.caption("Closer to 0.0 is fairer. Represents True Positive Rate difference.")
                st.markdown('</div>', unsafe_allow_html=True)
                
        with col2:
            st.markdown("#### 2. Apply Mitigation (Reweighting)")
            if st.button("Retrain with Fairness Mitigation", type="primary"):
                partitions = st.session_state['partitions']
                fair_dataloaders = []
                for p in partitions:
                    fair_dl = get_fair_dataloader(p['train'][0], p['train'][1], p['train'][2])
                    fair_dataloaders.append(fair_dl)
                
                new_model = FederatedNN(input_dim=X.shape[1])
                st.info("Training a balanced model over 3 rounds...")
                
                bar = st.progress(0)
                for r in range(3):
                    new_model = simulate_fl_round(new_model, fair_dataloaders, client_epochs=2)
                    bar.progress((r+1)/3)
                
                st.session_state['trained_model'] = new_model
                
                X_test, y_test, p_test = st.session_state['evaluation_data']
                acc_new, preds_new, probs_new = evaluate_model(new_model, X_test, y_test)
                
                dp_diff_new = calculate_demographic_parity(preds_new, p_test)
                eo_diff_new = calculate_equal_opportunity(y_test, preds_new, p_test)
                
                st.success("✅ Model Resampled and Retrained!")
                
                col_m1, col_m2 = st.columns(2)
                
                if 'pre_fairness' in st.session_state:
                    delta_dp = dp_diff_new - st.session_state['pre_fairness']['dp']
                    delta_eo = eo_diff_new - st.session_state['pre_fairness']['eo']
                else:
                    delta_dp = delta_eo = None
                
                col_m1.metric("New DP Difference", f"{dp_diff_new:.3f}", delta=f"{delta_dp:.3f}" if delta_dp is not None else None, delta_color="inverse")
                col_m2.metric("New EO Difference", f"{eo_diff_new:.3f}", delta=f"{delta_eo:.3f}" if delta_eo is not None else None, delta_color="inverse")
                
                st.markdown(f"**Trade-off Check:** New Accuracy is **{acc_new:.2%}**.")


# ====== TAB 4: GREEN AI ======
with tabs[3]:
    st.markdown("### 🍃 Green AI (Energy Efficiency)")
    
    if 'trained_model' not in st.session_state:
        st.warning("⚠️ Please train the Federated Model first in the 'Federated FL Training' tab.")
    else:
        st.write("Applying Pruning and Quantization to reduce carbon footprint and computational overhead.")
        
        if st.button("Optimize Model (Prune + Quantize)", type="primary"):
            model = st.session_state['trained_model'].cpu()
            
            with st.spinner("Calculating optimization paths..."):
                original_size = get_actual_size_kb(model)
                X_test, y_test, p_test = st.session_state['evaluation_data']
                original_acc, _, _ = evaluate_model(model, X_test, y_test)
                
                test_tensor = torch.tensor(X_test, dtype=torch.float32)
                start_time = time.time()
                _ = model(test_tensor)
                original_inference = (time.time() - start_time) * 1000
                
                optimized_model = execute_green_ai_pipeline(model, prune_amount=0.3)
                
                new_size = get_actual_size_kb(optimized_model)
                new_acc, _, _ = evaluate_model(optimized_model, X_test, y_test)
                
                start_time = time.time()
                _ = optimized_model(test_tensor)
                new_inference = (time.time() - start_time) * 1000
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Model Size", f"{new_size:.1f} KB", delta=f"{new_size - original_size:.1f} KB", delta_color="inverse")
            with c2:
                st.metric("Inference Time (ms)", f"{new_inference:.2f} ms", delta=f"{new_inference - original_inference:.2f} ms", delta_color="inverse")
            with c3:
                st.metric("Accuracy Maintained", f"{new_acc:.2%}", delta=f"{new_acc - original_acc:.2%}")
                
            fig = go.Figure(data=[
                go.Bar(name='Original Model', x=['Model Size (KB)'], y=[original_size], marker_color='#34495E'),
                go.Bar(name='Green Model', x=['Model Size (KB)'], y=[new_size], marker_color='#2ECC71')
            ])
            fig.update_layout(barmode='group', title="Footprint Comparison", width=600)
            st.plotly_chart(fig)
            st.success("Successfully Optimized! Ready for edge deployment.")
