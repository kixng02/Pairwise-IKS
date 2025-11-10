import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import io

# Set page configuration
st.set_page_config(
    page_title="Pairwise Agreement Analysis - AI Bias Study",
    page_icon="🇿🇦",
    layout="wide"
)

class PairwiseAgreementAnalyzer:
    """
    A comprehensive analyzer for pairwise agreement between AI models
    on indigenous plant classification.
    """
    
    def __init__(self):
        self.model_names = ["ChatGPT", "Gemini", "Mistral AI"]
        self.pairwise_agreement = None
        self.overall_agreement = None
        
    def calculate_pairwise_agreement(self, ratings):
        """
        Calculate pairwise agreement between all model combinations
        """
        n_models = len(ratings[0])
        n_plants = len(ratings)
        
        # Initialize agreement matrix
        agreement_matrix = np.zeros((n_models, n_models))
        total_comparisons = np.zeros((n_models, n_models))
        
        # Calculate pairwise agreement
        for i in range(n_models):
            for j in range(i + 1, n_models):
                agreements = 0
                valid_comparisons = 0
                
                for plant_ratings in ratings:
                    rating_i = plant_ratings[i]
                    rating_j = plant_ratings[j]
                    
                    # Only count if both models provided valid responses
                    if rating_i != -1 and rating_j != -1:
                        valid_comparisons += 1
                        if rating_i == rating_j:
                            agreements += 1
                
                if valid_comparisons > 0:
                    agreement_rate = agreements / valid_comparisons
                    agreement_matrix[i, j] = agreement_rate
                    agreement_matrix[j, i] = agreement_rate
                    total_comparisons[i, j] = valid_comparisons
                    total_comparisons[j, i] = valid_comparisons
        
        return agreement_matrix, total_comparisons
    
    def calculate_overall_agreement(self, ratings):
        """Calculate overall agreement across all models"""
        n_plants = len(ratings)
        total_agreements = 0
        total_valid_plants = 0
        
        for plant_ratings in ratings:
            # Count only plants with at least 2 valid responses
            valid_ratings = [r for r in plant_ratings if r != -1]
            if len(valid_ratings) >= 2:
                total_valid_plants += 1
                if len(set(valid_ratings)) == 1:  # All valid ratings are the same
                    total_agreements += 1
        
        if total_valid_plants > 0:
            return total_agreements / total_valid_plants
        return 0

def prepare_table10_data():
    """
    Prepare data from Table 10 for pairwise agreement analysis
    """
    plant_names = [
        "Aloe ferox", "African ginger", "Wild rosemary", "Devil's claw", 
        "African wormwood", "Pepperbark tree", "Pineapple flower", "Spekboom",
        "False horsewood", "Sand raisin", "Mountain nettle", "Acacia",
        "River karee", "Kudu lily", "Waterberg raisin", "Sweet wild garlic",
        "Cyrtanthus sanguineus", "Ruttya fruticosa", "Sesamum trilobum", "Aloe hahnii"
    ]
    
    # Using the same classification data as Table 1 (since Table 10 builds on it)
    classifications = [
        [0, 0, 1],       # Aloe ferox: Medicinal, Medicinal, Edible
        [0, 0, 0],       # African ginger: All Medicinal
        [-1, -1, -1],    # Wild rosemary: All No results
        [0, 0, 0],       # Devil's claw: All Medicinal
        [0, 0, 0],       # African wormwood: All Medicinal
        [-1, -1, -1],    # Pepperbark tree: All Not accurate
        [0, 0, 0],       # Pineapple flower: All Medicinal
        [-1, -1, -1],    # Spekboom: All Not accurate
        [2, -1, -1],     # False horsewood: Poisonous, No results, Not accurate
        [-1, -1, 1],     # Sand raisin: No results, Not accurate, Edible
        [-1, -1, -1],    # Mountain nettle: All Not accurate
        [0, 2, 2],       # Acacia: Medicinal, Poisonous, Poisonous
        [0, -1, -1],     # River karee: Medicinal, No results, Not accurate
        [0, -1, 0],      # Kudu lily: Medicinal, Not accurate, Medicinal
        [-1, -1, -1],    # Waterberg raisin: All Not accurate
        [-1, -1, -1],    # Sweet wild garlic: All No results
        [-1, -1, 1],     # Cyrtanthus sanguineus: Not accurate, Not accurate, Edible
        [-1, 0, -1],     # Ruttya fruticosa: No results, Medicinal, Not accurate
        [-1, -1, 1],     # Sesamum trilobum: No results, No results, Edible
        [-1, 0, -1]      # Aloe hahnii: No results, Medicinal, Not accurate
    ]
    
    return classifications, plant_names

def text_to_code(text_response):
    """Convert text responses to numerical codes"""
    response_map = {
        'Medicinal': 0,
        'Edible': 1,
        'Poisonous': 2,
        'No Results': -1
    }
    return response_map.get(text_response, -1)

def code_to_text(code):
    """Convert numerical codes back to text"""
    code_map = {
        0: 'Medicinal',
        1: 'Edible',
        2: 'Poisonous',
        -1: 'No Results'
    }
    return code_map.get(code, 'No Results')

def create_pairwise_heatmap(agreement_matrix, model_names):
    """Create a heatmap visualization of pairwise agreement"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create custom colormap from red to green
    cmap = LinearSegmentedColormap.from_list('agreement_cmap', ['red', 'yellow', 'green'])
    
    # Create mask for upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(agreement_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(agreement_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap=cmap,
                cbar_kws={'label': 'Agreement Rate'},
                xticklabels=model_names,
                yticklabels=model_names,
                ax=ax,
                vmin=0, 
                vmax=1)
    
    ax.set_title('Pairwise Agreement Between AI Models\n(Higher values indicate better agreement)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    return fig

def create_comparison_matrix(agreement_matrix, total_comparisons, model_names):
    """Create a detailed comparison matrix with counts"""
    comparisons = []
    n_models = len(model_names)
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            agreement = agreement_matrix[i, j]
            comparisons_count = total_comparisons[i, j]
            comparisons.append({
                'Model Pair': f"{model_names[i]} vs {model_names[j]}",
                'Agreement Rate': f"{agreement:.3f}",
                'Agreement Percentage': f"{agreement*100:.1f}%",
                'Valid Comparisons': int(comparisons_count),
                'Interpretation': interpret_agreement(agreement)
            })
    
    return pd.DataFrame(comparisons)

def interpret_agreement(rate):
    """Interpret agreement rate"""
    if rate >= 0.8:
        return "Excellent Agreement"
    elif rate >= 0.6:
        return "Good Agreement"
    elif rate >= 0.4:
        return "Moderate Agreement"
    elif rate >= 0.2:
        return "Fair Agreement"
    else:
        return "Poor Agreement"

def create_model_performance_chart(agreement_matrix, model_names):
    """Create a bar chart showing average agreement for each model"""
    # Calculate average agreement for each model (excluding self-comparison)
    avg_agreements = []
    for i in range(len(model_names)):
        other_agreements = [agreement_matrix[i, j] for j in range(len(model_names)) if i != j]
        avg_agreements.append(np.mean(other_agreements))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    bars = ax.bar(model_names, avg_agreements, color=colors, alpha=0.7)
    
    ax.set_ylabel('Average Agreement Rate')
    ax.set_title('Average Pairwise Agreement for Each AI Model', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_agreements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    return fig, avg_agreements

def create_detailed_comparison_heatmap(ratings, plant_names, model_names):
    """Create a detailed heatmap showing agreement patterns per plant"""
    n_plants = len(plant_names)
    n_models = len(model_names)
    
    # Create agreement matrix per plant
    plant_agreement = np.zeros((n_plants, n_models))
    
    for i, plant_ratings in enumerate(ratings):
        for j in range(n_models):
            # Count how many other models agree with this model for this plant
            agreements = 0
            valid_comparisons = 0
            
            for k in range(n_models):
                if j != k and plant_ratings[j] != -1 and plant_ratings[k] != -1:
                    valid_comparisons += 1
                    if plant_ratings[j] == plant_ratings[k]:
                        agreements += 1
            
            if valid_comparisons > 0:
                plant_agreement[i, j] = agreements / valid_comparisons
            else:
                plant_agreement[i, j] = -1  # No valid comparisons
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('plant_agreement', ['lightgray', 'red', 'yellow', 'green'])
    
    # Plot heatmap
    im = ax.imshow(plant_agreement, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Customize the plot
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_yticks(range(len(plant_names)))
    ax.set_yticklabels(plant_names)
    ax.set_title('Model Agreement Patterns per Plant\n(Green = High Agreement, Red = Low Agreement)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Agreement Rate with Other Models')
    
    return fig

def get_existing_plant_names():
    """Get list of existing plant names to prevent duplicates"""
    if 'user_plants_pairwise' not in st.session_state:
        return []
    return [plant['plant_name'].lower().strip() for plant in st.session_state.user_plants_pairwise]

def user_data_input_pairwise():
    """Allow users to input their own plant classification data for pairwise analysis"""
    st.header(" Insert Your Own Data")
    
    st.markdown("""
    **Instructions:**
    1. Enter at least **3 plant names** (each plant name must be unique)
    2. Select the classification response from each AI model for each plant
    3. Available responses: **Medicinal, Edible, Poisonous, No Results**
    4. Click 'Add Plant' after entering each plant's data
    5. Click 'Run Pairwise Analysis with Your Data' when you have at least 3 plants
    """)
    
    # Initialize session state for user data
    if 'user_plants_pairwise' not in st.session_state:
        st.session_state.user_plants_pairwise = []
    
    # Get existing plant names for duplicate checking
    existing_plants = get_existing_plant_names()
    
    # Input form for new plant
    with st.form("plant_input_form_pairwise", clear_on_submit=True):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            plant_name = st.text_input("Plant Name *", placeholder="e.g., Aloe vera", key="plant_name_pairwise")
        with col2:
            chatgpt_response = st.selectbox("ChatGPT *", ["No Results", "Medicinal", "Edible", "Poisonous"], key="chatgpt_pairwise")
        with col3:
            gemini_response = st.selectbox("Gemini *", ["No Results", "Medicinal", "Edible", "Poisonous"], key="gemini_pairwise")
        with col4:
            mistral_response = st.selectbox("Mistral AI *", ["No Results", "Medicinal", "Edible", "Poisonous"], key="mistral_pairwise")
        
        st.markdown("**All fields are required**")
        
        submitted = st.form_submit_button("Add Plant")
        
        if submitted:
            # Validate all fields are filled
            if not plant_name.strip():
                st.error("❌ Please enter a plant name.")
            elif plant_name.lower().strip() in existing_plants:
                st.error(f"❌ Plant '{plant_name}' already exists in your dataset. Please use a different name.")
            else:
                # All validations passed, add the plant
                plant_data = {
                    'plant_name': plant_name.strip(),
                    'chatgpt': text_to_code(chatgpt_response),
                    'gemini': text_to_code(gemini_response),
                    'mistral': text_to_code(mistral_response)
                }
                st.session_state.user_plants_pairwise.append(plant_data)
                st.success(f"✅ Added '{plant_name}' to the dataset!")
    
    # Display current user data
    if st.session_state.user_plants_pairwise:
        st.subheader(" Your Current Plant Data")
        display_data = []
        for plant in st.session_state.user_plants_pairwise:
            display_data.append({
                'Plant Name': plant['plant_name'],
                'ChatGPT': code_to_text(plant['chatgpt']),
                'Gemini': code_to_text(plant['gemini']),
                'Mistral AI': code_to_text(plant['mistral'])
            })
        
        df_display = pd.DataFrame(display_data)
        st.dataframe(df_display, use_container_width=True)
        
        # Show data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Plants", len(st.session_state.user_plants_pairwise))
        with col2:
            unique_plants = len(set([p['plant_name'].lower() for p in st.session_state.user_plants_pairwise]))
            st.metric("Unique Plants", unique_plants)
        with col3:
            if len(st.session_state.user_plants_pairwise) >= 3:
                st.metric("Status", "Ready for Analysis ✅")
            else:
                st.metric("Status", "Need More Data ⚠️")
        
        # Clear data button
        if st.button("🗑️ Clear All Data", key="clear_pairwise"):
            st.session_state.user_plants_pairwise = []
            st.rerun()
        
        # Check if we have enough data for analysis
        if len(st.session_state.user_plants_pairwise) >= 3:
            st.success(f"✅ You have {len(st.session_state.user_plants_pairwise)} plants. Ready for pairwise analysis!")
            return True
        else:
            st.warning(f"⚠️ You have {len(st.session_state.user_plants_pairwise)} plants. Need at least 3 for pairwise analysis.")
            return False
    else:
        st.info(" Start by adding your first plant using the form above.")
        return False

def prepare_user_data_pairwise():
    """Convert user data to the format needed for pairwise analysis"""
    if 'user_plants_pairwise' not in st.session_state:
        return [], []
    
    plant_names = []
    classifications = []
    
    for plant in st.session_state.user_plants_pairwise:
        plant_names.append(plant['plant_name'])
        classifications.append([
            plant['chatgpt'],
            plant['gemini'], 
            plant['mistral']
        ])
    
    return classifications, plant_names

def run_study_data_analysis(analyzer):
    """Run and display the study data pairwise analysis"""
    st.header("📋 Research Study Data - Pairwise Agreement Analysis")
    st.markdown("""
    This analysis shows **pairwise agreement** between AI models using the original research data from **Table 10**.
    
    Pairwise agreement measures how often two models provide the same classification when both give valid responses.
    """)
    
    # Load study data
    ratings, plant_names = prepare_table10_data()
    model_names = ["ChatGPT", "Gemini", "Mistral AI"]
    
    # Display data table
    st.subheader("Research Data Overview")
    display_data = []
    for i, plant in enumerate(plant_names):
        display_data.append({
            'Plant Name': plant,
            'ChatGPT': code_to_text(ratings[i][0]),
            'Gemini': code_to_text(ratings[i][1]),
            'Mistral AI': code_to_text(ratings[i][2])
        })
    
    df_display = pd.DataFrame(display_data)
    st.dataframe(df_display, use_container_width=True)
    
    # Run analysis automatically
    if st.button("Run Pairwise Agreement Analysis on Study Data", type="primary"):
        with st.spinner("Calculating pairwise agreement for research data..."):
            # Calculate pairwise agreement
            agreement_matrix, total_comparisons = analyzer.calculate_pairwise_agreement(ratings)
            overall_agreement = analyzer.calculate_overall_agreement(ratings)
            
            # Store results
            analyzer.pairwise_agreement = agreement_matrix
            analyzer.overall_agreement = overall_agreement
        
        if agreement_matrix is not None:
            # Display results
            st.header(" Pairwise Agreement Results")
            
            # Overall agreement
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Agreement", f"{overall_agreement:.3f}")
            with col2:
                st.metric("Overall Agreement %", f"{overall_agreement*100:.1f}%")
            with col3:
                st.metric("Interpretation", interpret_agreement(overall_agreement))
            
            # Pairwise heatmap
            st.subheader("Pairwise Agreement Heatmap")
            heatmap_fig = create_pairwise_heatmap(agreement_matrix, model_names)
            st.pyplot(heatmap_fig)
            
            # Detailed comparison matrix
            st.subheader("Detailed Pairwise Comparisons")
            comparison_df = create_comparison_matrix(agreement_matrix, total_comparisons, model_names)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Model performance chart
            st.subheader("Model Performance Comparison")
            performance_fig, avg_agreements = create_model_performance_chart(agreement_matrix, model_names)
            st.pyplot(performance_fig)
            
            # Detailed plant-level analysis
            st.subheader(" Plant-Level Agreement Patterns")
            plant_heatmap_fig = create_detailed_comparison_heatmap(ratings, plant_names, model_names)
            st.pyplot(plant_heatmap_fig)
            
            # Research insights
            st.subheader(" Research Insights")
            
            # Find best and worst agreement pairs
            comparisons = []
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    comparisons.append({
                        'pair': f"{model_names[i]} - {model_names[j]}",
                        'agreement': agreement_matrix[i, j]
                    })
            
            best_pair = max(comparisons, key=lambda x: x['agreement'])
            worst_pair = min(comparisons, key=lambda x: x['agreement'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Highest Agreement**: {best_pair['pair']} ({best_pair['agreement']:.3f})")
            with col2:
                st.warning(f"**Lowest Agreement**: {worst_pair['pair']} ({worst_pair['agreement']:.3f})")
            
            st.markdown("""
            **Key Findings:**
            - Pairwise agreement reveals which AI models are most consistent with each other
            - Low agreement rates indicate systematic differences in classification behavior
            - Patterns may reveal biases in training data or model architectures
            - High variability suggests the need for standardized evaluation frameworks
            """)

def main():
    # Main title and description
    st.title(" Pairwise Agreement Analysis: AI Model Consistency")
    st.markdown("""
    This application analyzes **pairwise agreement** between AI models (ChatGPT, Gemini, Mistral AI) 
    on plant classification tasks.
    
    **Pairwise agreement** measures how often two models provide identical classifications 
    when both give valid responses, revealing patterns of consistency and disagreement.
    """)
    
    # Initialize analyzer
    analyzer = PairwiseAgreementAnalyzer()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Set default selection to Study Data Analysis
    if 'pairwise_app_mode' not in st.session_state:
        st.session_state.pairwise_app_mode = "Study Data Analysis"
    
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Study Data Analysis", "Insert Your Own Data", "About Pairwise Analysis"],
        index=0  # Default to Study Data Analysis
    )
    
    # Update session state
    st.session_state.pairwise_app_mode = app_mode
    
    if app_mode == "Study Data Analysis":
        # Show study data analysis by default
        run_study_data_analysis(analyzer)
    
    elif app_mode == "Insert Your Own Data":
        # User data input and analysis
        ready_for_analysis = user_data_input_pairwise()
        
        if ready_for_analysis:
            st.header("Analyze Your Data")
            
            if st.button("Run Pairwise Analysis with Your Data", type="primary"):
                with st.spinner("Calculating pairwise agreement..."):
                    # Prepare user data
                    ratings, plant_names = prepare_user_data_pairwise()
                    model_names = ["ChatGPT", "Gemini", "Mistral AI"]
                    
                    # Calculate pairwise agreement
                    agreement_matrix, total_comparisons = analyzer.calculate_pairwise_agreement(ratings)
                    overall_agreement = analyzer.calculate_overall_agreement(ratings)
                    
                    # Store results
                    analyzer.pairwise_agreement = agreement_matrix
                    analyzer.overall_agreement = overall_agreement
                
                if agreement_matrix is not None:
                    # Display results
                    st.header("Your Pairwise Analysis Results")
                    
                    # Overall agreement
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Agreement", f"{overall_agreement:.3f}")
                    with col2:
                        st.metric("Overall Agreement %", f"{overall_agreement*100:.1f}%")
                    with col3:
                        st.metric("Interpretation", interpret_agreement(overall_agreement))
                    
                    # Pairwise heatmap
                    st.subheader("Pairwise Agreement Heatmap")
                    heatmap_fig = create_pairwise_heatmap(agreement_matrix, model_names)
                    st.pyplot(heatmap_fig)
                    
                    # Detailed comparison matrix
                    st.subheader("Detailed Pairwise Comparisons")
                    comparison_df = create_comparison_matrix(agreement_matrix, total_comparisons, model_names)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Model performance chart
                    st.subheader("Model Performance Comparison")
                    performance_fig, avg_agreements = create_model_performance_chart(agreement_matrix, model_names)
                    st.pyplot(performance_fig)
                    
                    # Export results
                    st.subheader("Export Your Results")
                    
                    # Create downloadable data
                    export_data = []
                    for i, plant in enumerate(plant_names):
                        export_data.append({
                            'Plant_Name': plant,
                            'ChatGPT': code_to_text(ratings[i][0]),
                            'Gemini': code_to_text(ratings[i][1]),
                            'Mistral_AI': code_to_text(ratings[i][2])
                        })
                    
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Your Data as CSV",
                        data=csv,
                        file_name="pairwise_analysis_data.csv",
                        mime="text/csv"
                    )
    
    else:  # About mode
        st.header("About Pairwise Agreement Analysis")
        st.markdown("""
        ### What is Pairwise Agreement?
        Pairwise agreement measures how often two raters (AI models) provide identical classifications 
        when both give valid responses. It's particularly useful for understanding:
        
        - **Model Consistency**: Which models tend to agree with each other
        - **Systematic Differences**: Patterns of disagreement that may indicate biases
        - **Reliability Assessment**: How dependable the models are relative to each other
        
        ### Methodology
        - **Calculation**: Agreement rate = (Number of matching classifications) / (Number of valid comparisons)
        - **Models**: ChatGPT, Google Gemini, Mistral AI
        - **Classifications**: Medicinal, Edible, Poisonous, or No Results
        - **Valid Comparisons**: Only counts instances where both models provided valid responses
        
        ### Interpretation Guide
        - **≥ 0.80**: Excellent Agreement (Models are very consistent)
        - **0.60 - 0.79**: Good Agreement (Models are generally consistent)  
        - **0.40 - 0.59**: Moderate Agreement (Some inconsistency present)
        - **0.20 - 0.39**: Fair Agreement (Significant differences)
        - **< 0.20**: Poor Agreement (Models frequently disagree)
        
        ### Research Context
        This analysis complements the Fleiss' Kappa analysis by providing detailed insights 
        into specific model-to-model relationships and consistency patterns.
        """)

if __name__ == "__main__":
    main()
