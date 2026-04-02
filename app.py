#!/usr/bin/env python3
"""
TNBC Drug Discovery Pipeline
STRING-db target discovery and ranking for triple negative breast cancer
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import io

# Page config
st.set_page_config(
    page_title="TNBC Drug Discovery",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subheader {
        color: #555;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Target selection
st.sidebar.subheader("Primary Targets")
primary_targets = st.sidebar.multiselect(
    "Select targets to query:",
    ["EGFR", "MMP1", "MMP7", "MMP12", "DNMT1"],
    default=["EGFR", "MMP1", "MMP7", "MMP12", "DNMT1"]
)

# Confidence threshold
confidence_threshold = st.sidebar.slider(
    "STRING-db Confidence Threshold:",
    min_value=0.4,
    max_value=1.0,
    value=0.7,
    step=0.05,
    help="Higher values = more confident interactions"
)

# Druggable targets filter
st.sidebar.subheader("Druggable Surface Receptors")
druggable_targets = {
    "EGFR Family": ["EGFR", "ERBB2", "ERBB3", "ERBB4"],
    "Growth Factor Receptors": ["MET", "IGF1R", "FGFR1", "FGFR2", "FGFR3", "PDGFRA", "PDGFRB"],
    "Proteases": ["MMP1", "MMP2", "MMP3", "MMP7", "MMP9", "MMP12", "MMP13", "MMP14", "ADAM10", "ADAM17"],
    "Immune Checkpoints": ["CD40", "TNFRSF1A", "TNFRSF1B", "IL6R", "IL10R"],
    "Notch/Wnt": ["NOTCH1", "NOTCH2", "NOTCH3", "NOTCH4", "FZD1", "FZD2", "FZD3", "FZD4", "FZD5"],
    "Cell Adhesion": ["ITGA1", "ITGA5", "ITGAV", "ITGB1", "L1CAM", "NCAM1"]
}

selected_categories = st.sidebar.multiselect(
    "Select receptor categories:",
    list(druggable_targets.keys()),
    default=list(druggable_targets.keys())
)

# Flatten selected druggable targets
druggable_set = set()
for category in selected_categories:
    druggable_set.update(druggable_targets[category])

# Main title
st.markdown('<div class="main-header">🧬 TNBC Drug Discovery Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">STRING-db Target Discovery for Triple Negative Breast Cancer</div>', unsafe_allow_html=True)

st.markdown("---")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### 📋 Pipeline Overview")
    st.markdown("""
    This app queries the **STRING-db** protein interaction database to discover novel druggable targets 
    that interact with key TNBC drivers (EGFR, MMPs, DNMT). Results are ranked by interaction confidence 
    and experimental evidence.
    
    **Workflow:**
    1. Query STRING-db for protein interactions
    2. Filter for druggable surface receptors
    3. Rank by binding score + experimental evidence
    4. Export results to CSV
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### ⏱️ Status")
    st.markdown(f"""
    **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    **Configuration:**
    - Targets: {len(primary_targets)}
    - Threshold: {confidence_threshold}
    - Categories: {len(selected_categories)}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Query STRING-db
if st.button("🔍 Run Discovery Pipeline", key="run_button", use_container_width=True):
    with st.spinner("Querying STRING-db for interactions..."):
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, target in enumerate(primary_targets):
            status_text.text(f"Querying {target}... ({idx+1}/{len(primary_targets)})")
            
            try:
                url = f"https://string-db.org/api/json/interaction_partners?identifiers={target}&species=9606&required_score={confidence_threshold}&format=json"
                response = requests.get(url, timeout=10)
                interactions = response.json()
                
                for interaction in interactions:
                    gene = interaction.get("stringId", "").split(".")[-1].upper()
                    score = float(interaction.get("score", 0))
                    
                    if gene in druggable_set and gene not in primary_targets:
                        all_results.append({
                            "Gene": gene,
                            "Score": round(score, 3),
                            "InteractsWith": target,
                            "Experiments": round(float(interaction.get("expertscore", 0)), 3),
                            "Coexpression": round(float(interaction.get("escore", 0)), 3),
                            "Database": round(float(interaction.get("dscore", 0)), 3),
                            "StringID": interaction.get("stringId", "")
                        })
                
                progress_bar.progress((idx + 1) / len(primary_targets))
                
            except Exception as e:
                st.error(f"Error querying {target}: {str(e)}")
        
        status_text.empty()
        progress_bar.empty()
        
        if all_results:
            # Remove duplicates, keep highest score
            df = pd.DataFrame(all_results)
            df_unique = df.loc[df.groupby('Gene')['Score'].idxmax()].reset_index(drop=True)
            
            # Sort by score
            df_unique = df_unique.sort_values('Score', ascending=False).reset_index(drop=True)
            df_unique['Rank'] = range(1, len(df_unique) + 1)
            
            # Reorder columns
            df_unique = df_unique[['Rank', 'Gene', 'Score', 'Experiments', 'Coexpression', 'Database', 'InteractsWith', 'StringID']]
            
            # Store in session state
            st.session_state.results_df = df_unique
            
            st.success(f"✅ Discovery complete! Found {len(df_unique)} unique druggable targets")
        else:
            st.warning("No targets found with current filters. Try adjusting parameters.")

# Display results
if 'results_df' in st.session_state:
    st.markdown("---")
    st.markdown("### 📊 Results")
    
    df = st.session_state.results_df
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Targets", len(df))
    col2.metric("Avg Score", f"{df['Score'].mean():.3f}")
    col3.metric("With Experiments", (df['Experiments'] > 0).sum())
    col4.metric("Top Target", df.iloc[0]['Gene'] if len(df) > 0 else "N/A")
    
    st.markdown("---")
    
    # Results table
    st.markdown("#### Top 20 Druggable Targets")
    
    # Format display dataframe
    display_df = df.head(20).copy()
    display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.3f}")
    display_df['Experiments'] = display_df['Experiments'].apply(lambda x: f"{x:.3f}")
    display_df['Coexpression'] = display_df['Coexpression'].apply(lambda x: f"{x:.3f}")
    display_df['Database'] = display_df['Database'].apply(lambda x: f"{x:.3f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Download options
    st.markdown("#### 📥 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download as CSV",
            data=csv,
            file_name=f"tnbc_targets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download as JSON",
            data=json_str,
            file_name=f"tnbc_targets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Interaction breakdown
    st.markdown("#### 🔗 Interactions by Primary Target")
    interaction_counts = df['InteractsWith'].value_counts()
    st.bar_chart(interaction_counts, use_container_width=True)
    
    # Statistics
    st.markdown("#### 📈 Score Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Min Score", f"{df['Score'].min():.3f}")
        st.metric("Max Score", f"{df['Score'].max():.3f}")
    
    with col2:
        st.metric("Median Score", f"{df['Score'].median():.3f}")
        st.metric("Std Dev", f"{df['Score'].std():.3f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em; margin-top: 20px;">
    <p>🧬 TNBC Drug Discovery Pipeline | Powered by STRING-db | Built with Streamlit</p>
    <p><a href="https://string-db.org" target="_blank">STRING-db Documentation</a> | 
    <a href="https://github.com" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
