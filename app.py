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
import tempfile
import os
import zipfile

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from meeko import MoleculePreparation
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False

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
st.markdown('<div class="subheader">STRING-db Target Discovery for Triple Negative Breast Cancer | Developed by Yashwant Nama</div>', unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Target Discovery", "💊 3D Ligand Preparation", "🧪 Protein Prep", "⚗️ Docking Prep", "🚀 Run Docking", "🔬 3D Visualization"])

with tab1:
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
                    # STRING-db requires score on 0-1000 scale
                    score_threshold = int(confidence_threshold * 1000)
                    url = f"https://string-db.org/api/json/interaction_partners?identifiers={target}&species=9606&required_score={score_threshold}&limit=200"
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    interactions = response.json()

                    for interaction in interactions:
                        # preferredName_B is the interacting partner
                        gene = interaction.get("preferredName_B", "").upper()
                        if not gene:
                            gene = interaction.get("stringId_B", "").split(".")[-1].upper()
                        score = float(interaction.get("score", 0))

                        if gene in druggable_set and gene not in primary_targets:
                            all_results.append({
                                "Gene": gene,
                                "Score": round(score, 3),
                                "InteractsWith": target,
                                "Experiments": round(float(interaction.get("experimentsscore", 0)), 3),
                                "Coexpression": round(float(interaction.get("coexpressionscore", 0)), 3),
                                "Database": round(float(interaction.get("databasescore", 0)), 3),
                                "StringID": interaction.get("stringId_B", "")
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


with tab2:
    st.markdown("### 💊 Automated 3D Ligand Preparation Pipeline")
    st.markdown("""
    This tool automatically converts 2D SDF files into 3D conformers, adds polar hydrogens, 
    and performs energy minimization (MMFF94) for AutoDock Vina docking.
    
    **Workflow:** Upload 2D SDF → Add Hydrogens → Generate 3D Conformers → MMFF94 Minimization → Download ZIP
    """)
    
    if not RDKIT_AVAILABLE:
        st.error("❌ RDKit not installed. Add `rdkit` to requirements.txt and redeploy.")
    else:
        uploaded_sdfs = st.file_uploader(
            "Upload 2D SDF Files", 
            type=["sdf"], 
            accept_multiple_files=True,
            help="Download 2D SDF files from PubChem or ChemDraw"
        )
        
        if uploaded_sdfs:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.info(f"**{len(uploaded_sdfs)} file(s) uploaded**")
                for f in uploaded_sdfs:
                    st.write(f"📄 {f.name}")
            
            with col2:
                st.markdown("""
                **Processing steps:**
                1. Parse 2D SDF structure
                2. Add polar hydrogens (`Chem.AddHs`)
                3. Generate 3D coordinates (ETKDGv3)
                4. Minimize energy (MMFF94, 500 iterations)
                5. Export as 3D SDF
                """)
            
            if st.button("🚀 Process & Minimize Ligands", use_container_width=True):
                with tempfile.TemporaryDirectory() as temp_dir:
                    processed_files = []
                    failed_files = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, uploaded_file in enumerate(uploaded_sdfs):
                        filename = uploaded_file.name
                        status_text.text(f"⚙️ Processing {filename}... ({i+1}/{len(uploaded_sdfs)})")
                        
                        try:
                            file_content = uploaded_file.read()
                            temp_input_path = os.path.join(temp_dir, f"raw_{filename}")
                            with open(temp_input_path, "wb") as f:
                                f.write(file_content)
                            
                            supplier = Chem.SDMolSupplier(temp_input_path)
                            mol = next(iter(supplier))
                            
                            if mol is not None:
                                mol_h = Chem.AddHs(mol)
                                result = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
                                
                                if result == -1:
                                    # Fallback to ETKDG if ETKDGv3 fails
                                    AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())
                                
                                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500, nonBondedThresh=100.0)
                                
                                output_path = os.path.join(temp_dir, f"3D_minimized_{filename}")
                                writer = Chem.SDWriter(output_path)
                                writer.write(mol_h)
                                writer.close()
                                
                                processed_files.append(output_path)
                                st.success(f"✅ {filename} — minimized successfully")
                            else:
                                failed_files.append(filename)
                                st.error(f"⚠️ Could not parse {filename} — check SDF format")
                                
                        except Exception as e:
                            failed_files.append(filename)
                            st.error(f"❌ Error with {filename}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_sdfs))
                    
                    status_text.text("🎉 Processing complete!")
                    
                    if processed_files:
                        # Create ZIP
                        zip_path = os.path.join(temp_dir, "minimized_ligands.zip")
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in processed_files:
                                zipf.write(file, os.path.basename(file))
                        
                        with open(zip_path, "rb") as f:
                            zip_data = f.read()
                        
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("✅ Processed", len(processed_files))
                        col2.metric("❌ Failed", len(failed_files))
                        col3.metric("Total", len(uploaded_sdfs))
                        
                        st.download_button(
                            label="📥 Download All 3D Minimized Ligands (ZIP)",
                            data=zip_data,
                            file_name=f"minimized_ligands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                    else:
                        st.error("No files were successfully processed.")

with tab3:
    st.markdown("### 🧪 Protein Structure Preparation")
    st.markdown("""
    Upload a raw PDB file to automatically remove water molecules (HOH) and existing ligands (HETATM records),
    keeping only standard protein ATOM records — ready for AutoDock Vina.
    
    **Workflow:** Upload raw PDB → Remove HOH + HETATM → Keep ATOM/TER/END → Download clean PDB
    """)
    
    uploaded_pdb = st.file_uploader(
        "Upload Raw PDB File",
        type=["pdb"],
        help="Download from RCSB PDB (rcsb.org)"
    )
    
    if uploaded_pdb:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.info(f"**File:** {uploaded_pdb.name}  \n**Size:** {uploaded_pdb.size / 1024:.1f} KB")
        with col2:
            st.markdown("""
            **What gets removed:**
            - 💧 Water molecules (`HOH`)
            - 💊 Existing ligands (all `HETATM` records)
            
            **What is kept:**
            - 🧬 Standard amino acid atoms (`ATOM`)
            - 🔚 Chain terminators (`TER`, `END`)
            """)
        
        if st.button("🧹 Clean Protein Structure", use_container_width=True):
            try:
                lines = uploaded_pdb.read().decode("utf-8").splitlines(keepends=True)
                
                removed_water = 0
                removed_ligand = 0
                kept_atoms = 0
                clean_lines = []
                
                for line in lines:
                    if line.startswith("HETATM"):
                        if "HOH" in line:
                            removed_water += 1
                        else:
                            removed_ligand += 1
                        continue
                    elif line.startswith("ATOM"):
                        clean_lines.append(line)
                        kept_atoms += 1
                    elif line.startswith("TER") or line.startswith("END"):
                        clean_lines.append(line)
                
                clean_pdb_content = "".join(clean_lines)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                col1.metric("💧 Water Removed", removed_water)
                col2.metric("💊 Ligands Removed", removed_ligand)
                col3.metric("🧬 Atoms Kept", kept_atoms)
                
                if kept_atoms > 0:
                    st.success("✅ Protein cleaned successfully! Ready for docking.")
                    
                    output_name = uploaded_pdb.name.replace(".pdb", "_clean.pdb")
                    st.download_button(
                        label=f"📥 Download Clean PDB ({output_name})",
                        data=clean_pdb_content,
                        file_name=output_name,
                        mime="chemical/x-pdb",
                        use_container_width=True
                    )
                    
                    # Preview first 10 ATOM lines
                    st.markdown("#### 🔎 Preview (first 10 ATOM lines)")
                    preview_lines = [l for l in clean_lines if l.startswith("ATOM")][:10]
                    st.code("".join(preview_lines), language="text")
                else:
                    st.error("No ATOM records found. Check if the file is a valid PDB.")
                    
            except Exception as e:
                st.error(f"❌ Error processing PDB: {str(e)}")


with tab4:
    st.markdown("### ⚗️ AutoDock Vina Docking Preparation")
    st.markdown("""
    Generate a **Grid Box config file** for AutoDock Vina and convert **3D SDF ligands → PDBQT format** 
    using Meeko (Gasteiger charges + AutoDock atom types). 
    """)

    # --- Section A: Grid Box Config ---
    st.subheader("A. Grid Box Configuration")
    st.markdown("Set the docking search space coordinates. For EGFR kinase domain (PDB: 1M17), ATP-binding pocket defaults are pre-filled.")

    col1, col2, col3 = st.columns(3)
    with col1:
        center_x = st.number_input("Center X (Å)", value=22.5, step=0.5)
        size_x = st.number_input("Size X (Å)", value=25, step=1)
    with col2:
        center_y = st.number_input("Center Y (Å)", value=4.5, step=0.5)
        size_y = st.number_input("Size Y (Å)", value=25, step=1)
    with col3:
        center_z = st.number_input("Center Z (Å)", value=51.5, step=0.5)
        size_z = st.number_input("Size Z (Å)", value=25, step=1)

    exhaustiveness = st.slider("Exhaustiveness (Search Rigor)", min_value=4, max_value=32, value=8, step=4,
                               help="Higher = more thorough search, slower. 8 is standard.")

    receptor_name = st.text_input("Receptor PDBQT filename", value="receptor.pdbqt")

    if st.button("📄 Generate config.txt", use_container_width=True):
        config_content = f"""receptor = {receptor_name}
center_x = {center_x}
center_y = {center_y}
center_z = {center_z}
size_x = {size_x}
size_y = {size_y}
size_z = {size_z}
exhaustiveness = {exhaustiveness}
"""
        st.success("✅ config.txt generated!")
        st.code(config_content, language="text")
        st.download_button(
            label="📥 Download config.txt",
            data=config_content,
            file_name="config.txt",
            mime="text/plain",
            use_container_width=True
        )

    st.divider()

    # --- Section B: Ligand PDBQT Conversion ---
    st.subheader("B. Batch Ligand Conversion (SDF → PDBQT)")
    st.markdown("Upload your 3D minimized SDF files (from Tab 2). Uses **Meeko** to assign Gasteiger charges and AutoDock atom types.")

    if not MEEKO_AVAILABLE:
        st.error("❌ Meeko not installed. Add `meeko` to requirements.txt and redeploy.")
    elif not RDKIT_AVAILABLE:
        st.error("❌ RDKit not installed. Add `rdkit` to requirements.txt and redeploy.")
    else:
        uploaded_3d_sdfs = st.file_uploader(
            "Upload 3D Minimized SDF Files",
            type=["sdf"],
            accept_multiple_files=True,
            key="pdbqt_uploader",
            help="Use files from Tab 2 (3D Ligand Preparation output)"
        )

        if uploaded_3d_sdfs:
            st.info(f"**{len(uploaded_3d_sdfs)} file(s)** ready for PDBQT conversion")

            if st.button("⚙️ Convert All Ligands to PDBQT", use_container_width=True):
                pdbqt_files = {}
                failed = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_3d_sdfs):
                    ligand_name = uploaded_file.name.replace(".sdf", "")
                    status_text.text(f"Converting {ligand_name}... ({i+1}/{len(uploaded_3d_sdfs)})")

                    try:
                        with tempfile.NamedTemporaryFile(suffix=".sdf", delete=False) as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        supplier = Chem.SDMolSupplier(tmp_path, removeHs=False)
                        mol = supplier[0] if supplier and len(supplier) > 0 else None
                        os.unlink(tmp_path)

                        if mol is not None:
                            preparator = MoleculePreparation()
                            preparator.prepare(mol)
                            pdbqt_string = preparator.write_pdbqt_string()
                            pdbqt_files[f"{ligand_name}.pdbqt"] = pdbqt_string
                            st.success(f"✅ {ligand_name}.pdbqt — converted")
                        else:
                            failed.append(ligand_name)
                            st.error(f"⚠️ Could not parse {ligand_name}")

                    except Exception as e:
                        failed.append(ligand_name)
                        st.error(f"❌ {ligand_name}: {str(e)}")

                    progress_bar.progress((i + 1) / len(uploaded_3d_sdfs))

                status_text.text("✅ Conversion complete!")

                if pdbqt_files:
                    # Pack all PDBQT into ZIP
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                        for fname, content_str in pdbqt_files.items():
                            zf.writestr(fname, content_str)
                    zip_buffer.seek(0)

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("✅ Converted", len(pdbqt_files))
                    col2.metric("❌ Failed", len(failed))
                    col3.metric("Total", len(uploaded_3d_sdfs))

                    st.download_button(
                        label="📥 Download All PDBQT Files (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"vina_ligands_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

                    st.info("💡 Next step: Run AutoDock Vina with these PDBQT files + the config.txt from Section A")


with tab5:
    st.markdown("### 🚀 Run AutoDock Vina in Cloud")
    st.markdown("""
    Upload your prepared files and run **AutoDock Vina** directly in the cloud — no local installation needed.
    Vina binary is auto-downloaded from the official GitHub release.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **Required files:**
        - 🧬 Receptor PDBQT (from Tab 3 → cleaned, then converted externally)
        - 💊 Ligand PDBQT (from Tab 4 → Section B output)
        - ⚙️ Config TXT (from Tab 4 → Section A output)
        """)
    with col2:
        st.warning("""
        **Note:** First run downloads Vina (~4MB). 
        Docking takes **1-3 minutes** per ligand depending on exhaustiveness.
        Results are the top binding poses ranked by ΔG (kcal/mol).
        """)

    st.markdown("---")

    receptor_file = st.file_uploader("🧬 Upload Receptor (PDBQT)", type=['pdbqt'], key="rec_upload")
    ligand_file = st.file_uploader("💊 Upload Ligand (PDBQT)", type=['pdbqt'], key="lig_upload")
    config_file = st.file_uploader("⚙️ Upload Config (TXT)", type=['txt'], key="conf_upload")

    # Show config preview
    if config_file:
        config_text = config_file.read().decode("utf-8")
        config_file.seek(0)
        st.markdown("**Config preview:**")
        st.code(config_text, language="text")

    if st.button("🚀 Run AutoDock Vina Docking", use_container_width=True):
        if receptor_file and ligand_file and config_file:
            
            # Save uploaded files to disk (Vina needs physical files)
            with open("receptor.pdbqt", "wb") as f:
                f.write(receptor_file.getbuffer())
            with open("ligand.pdbqt", "wb") as f:
                f.write(ligand_file.getbuffer())
            with open("config.txt", "wb") as f:
                f.write(config_file.getbuffer())

            vina_path = "./vina"

            # Download Vina binary if not present
            if not os.path.exists(vina_path):
                with st.spinner("⬇️ Downloading AutoDock Vina binary (~4MB)..."):
                    try:
                        import urllib.request
                        url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
                        urllib.request.urlretrieve(url, vina_path)
                        os.system("chmod +x " + vina_path)
                        st.success("✅ Vina downloaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Failed to download Vina: {str(e)}")
                        st.stop()

            with st.spinner("🔬 Running docking... (1-3 minutes)"):
                import subprocess
                cmd = [
                    vina_path,
                    "--receptor", "receptor.pdbqt",
                    "--ligand", "ligand.pdbqt",
                    "--config", "config.txt",
                    "--out", "result_out.pdbqt"
                ]
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if process.returncode == 0 or "Writing output" in process.stdout:
                st.success("✅ Docking Completed Successfully!")
                st.markdown("---")

                # Parse binding affinities from Vina output
                st.markdown("#### 📊 Binding Affinity Results")
                lines = process.stdout.split("\n")
                result_lines = []
                capture = False
                for line in lines:
                    if "mode" in line and "affinity" in line:
                        capture = True
                    if capture and line.strip():
                        result_lines.append(line)

                if result_lines:
                    st.code("\n".join(result_lines[:15]), language="text")

                    # Try to parse into a table
                    try:
                        import re
                        rows = []
                        for line in result_lines[2:]:
                            parts = line.split()
                            if len(parts) >= 4 and parts[0].isdigit():
                                rows.append({
                                    "Mode": int(parts[0]),
                                    "Affinity (kcal/mol)": float(parts[1]),
                                    "RMSD l.b.": float(parts[2]),
                                    "RMSD u.b.": float(parts[3])
                                })
                        if rows:
                            results_df = pd.DataFrame(rows)
                            st.dataframe(results_df, use_container_width=True, hide_index=True)
                            best = results_df.iloc[0]
                            st.metric(
                                "🏆 Best Binding Affinity",
                                f"{best['Affinity (kcal/mol)']} kcal/mol",
                                help="More negative = stronger binding"
                            )
                    except:
                        pass

                # Terminal logs
                with st.expander("📋 Full Vina Terminal Output"):
                    st.text_area("Logs:", process.stdout, height=300)

                # Download result
                if os.path.exists("result_out.pdbqt"):
                    with open("result_out.pdbqt", "rb") as f:
                        result_data = f.read()
                    st.download_button(
                        label="📥 Download Docked Poses (PDBQT)",
                        data=result_data,
                        file_name=f"docked_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdbqt",
                        mime="chemical/x-pdb",
                        use_container_width=True
                    )
            else:
                st.error("❌ Docking Failed!")
                with st.expander("🔍 Error Logs"):
                    st.text_area("stderr:", process.stderr, height=300)
                    st.text_area("stdout:", process.stdout, height=200)

        else:
            st.warning("⚠️ Please upload all 3 files: Receptor PDBQT, Ligand PDBQT, and Config TXT")


with tab6:
    st.markdown("### 🔬 3D Visualization of Docked Complex")
    st.markdown("""
    Visualize the receptor-ligand docked complex in interactive 3D.
    Use files from **Tab 5** output, or upload directly here.
    """)

    if not PY3DMOL_AVAILABLE:
        st.error("❌ py3Dmol not installed. Add `py3Dmol` to requirements.txt and redeploy.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Option A: Use files from Tab 5 docking run**")
            use_previous = st.checkbox("Use receptor.pdbqt + result_out.pdbqt from Tab 5", value=True)
        with col2:
            st.markdown("**Option B: Upload files manually**")
            viz_receptor = st.file_uploader("Upload Receptor PDBQT", type=["pdbqt"], key="viz_rec")
            viz_ligand = st.file_uploader("Upload Docked Ligand (PDBQT or SDF)", type=["pdbqt", "sdf"], key="viz_lig")

        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            receptor_style = st.selectbox("Receptor Style", ["cartoon", "surface", "line", "stick"], index=0)
            receptor_color = st.color_picker("Receptor Color", "#aaaaaa")
        with col2:
            ligand_style = st.selectbox("Ligand Style", ["stick", "sphere", "line", "cross"], index=0)
            ligand_colorscheme = st.selectbox("Ligand Color Scheme", ["greenCarbon", "cyanCarbon", "magentaCarbon", "yellowCarbon"], index=0)
        with col3:
            bg_color = st.color_picker("Background Color", "#1a1a2e")
            viewer_height = st.slider("Viewer Height (px)", 400, 800, 550, step=50)

        if st.button("🎨 Render 3D Structure", use_container_width=True):
            receptor_data = None
            ligand_data = None

            ligand_filename = "ligand.pdbqt"
            if use_previous and not viz_receptor and not viz_ligand:
                try:
                    with open("receptor.pdbqt", "r") as f:
                        receptor_data = f.read()
                    with open("result_out.pdbqt", "r") as f:
                        ligand_data = f.read()
                    ligand_filename = "result_out.pdbqt"
                    st.success("✅ Loaded files from Tab 5 docking run")
                except FileNotFoundError:
                    st.warning("⚠️ Tab 5 files not found. Upload files manually below.")

            if viz_receptor and viz_ligand:
                receptor_data = viz_receptor.read().decode("utf-8")
                ligand_data = viz_ligand.read().decode("utf-8")
                ligand_filename = viz_ligand.name
                st.success("✅ Loaded uploaded files")

            if receptor_data and ligand_data:
                try:
                    import streamlit.components.v1 as components

                    # Escape for JS
                    rec_escaped = receptor_data.replace("\\", "\\\\").replace("`", "\\`")
                    lig_escaped = ligand_data.replace("\\", "\\\\").replace("`", "\\`")

                    lig_fname = ligand_filename if 'ligand_filename' in dir() else 'ligand.pdbqt'
                    html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
  <style>
    body {{ margin: 0; padding: 0; background: {bg_color}; }}
    #viewer {{ width: 100%; height: {viewer_height}px; position: relative; }}
  </style>
</head>
<body>
  <div id="viewer"></div>
  <script>
    let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "{bg_color}"}});
    
    let recData = `{rec_escaped}`;
    let ligData = `{lig_escaped}`;
    
    viewer.addModel(recData, "pdbqt");
    viewer.setStyle({{model: 0}}, {{"{receptor_style}": {{color: "{receptor_color}"}}}});
    
    viewer.addModel(ligData, "pdbqt");
    viewer.setStyle({{model: 1}}, {{"{ligand_style}": {{colorscheme: "{ligand_colorscheme}"}}}});
    
    viewer.zoomTo({{model: 1}});
    viewer.render();
  </script>
</body>
</html>
"""
                    components.html(html_content, height=viewer_height + 20)

                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    col1.info("🖱️ **Rotate:** Left click + drag")
                    col2.info("🔍 **Zoom:** Scroll wheel")
                    col3.info("✋ **Pan:** Right click + drag")

                except Exception as e:
                    st.error(f"❌ Visualization error: {str(e)}")
            else:
                st.warning("⚠️ No structure data available. Run docking in Tab 5 or upload files.")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9em; margin-top: 20px;">
    <p>🧬 TNBC Drug Discovery Pipeline | Powered by STRING-db | Built with Streamlit</p>
    <p><a href="https://string-db.org" target="_blank">STRING-db Documentation</a> | 
    <a href="https://github.com" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
