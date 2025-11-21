import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---

local_files = [
    'breakdown_results_local_11n_1.csv',
    'breakdown_results_local_11n_3.csv',
    'breakdown_results_local_11s_1.csv',
    'breakdown_results_local_11s_6.csv'
]

edge_files = [
    'breakdown_results_edge_11n.csv',
    'breakdown_results_edge_11s.csv',
    'breakdown_results_edge_11x.csv'
]

# 1. Local files columns (no header in file, so we assign these)
local_cols = ['Preprocessing', 'Inference', 'Postprocessing', 'Tracking', 'OpFlow']

# 2. Edge files columns (headers exist in file, usually lowercase)
# We will map these to the Capitalized versions later to match local files
edge_cols_source  = ['preprocessing_time', 'inference_time', 'postprocessing_time', 'tracking_time', 'network_time']

# Dictionary to map edge CSV headers to the final plot labels
edge_map = {
    'preprocessing_time': 'Preprocessing',
    'inference_time': 'Inference',
    'postprocessing_time': 'Postprocessing',
    'tracking_time': 'Tracking',
    'network_time': 'Network'
}

data_to_plot = {}
all_files_found = True

# --- Data Processing ---

print("Starting data processing...")

# Process "local" files
for file in local_files:
    try:
        df = pd.read_csv(file, header=None)
        values = df.iloc[0].values
        s = pd.Series(values, index=local_cols)
        
        key_name = file.replace('breakdown_results_', '').replace('.csv', '')
        s.name = key_name
        data_to_plot[key_name] = s
        print(f"Processed {file} (local)")
        
    except FileNotFoundError:
        print(f"Error: File not found: {file}")
        all_files_found = False

# Process "edge" files
for file in edge_files:
    try:
        df = pd.read_csv(file)
        
        # --- FIX IS HERE: Clean column names ---
        # This removes leading/trailing spaces (e.g., " tracking_time" becomes "tracking_time")
        df.columns = df.columns.str.strip()
        
        # Check if columns exist before proceeding
        missing = [c for c in edge_cols_source if c not in df.columns]
        if missing:
            print(f"WARNING: Missing columns in {file}: {missing}")
            print(f"Found columns: {df.columns.tolist()}")
            continue

        # Calculate mean
        values = df[edge_cols_source].mean()
        
        # Rename index to match Local files (Capitalized)
        values = values.rename(edge_map)
        
        key_name = file.replace('breakdown_results_', '').replace('.csv', '')
        values.name = key_name
        data_to_plot[key_name] = values
        print(f"Processed {file} (edge)")

    except FileNotFoundError:
        print(f"Error: File not found: {file}")
        all_files_found = False

# --- Plotting ---

if data_to_plot:
    print("\nGenerating plot...")
    
    plot_df = pd.concat(data_to_plot.values(), axis=1).T
    plot_df = plot_df.fillna(0)

    file_order = [
        'local_11n_1', 'local_11n_3',
        'local_11s_1', 'local_11s_6',
        'edge_11n', 'edge_11s', 'edge_11x'
    ]
    processed_order = [f for f in file_order if f in plot_df.index]
    plot_df = plot_df.reindex(processed_order)

    # 'Network' comes from edge files, 'OpFlow' from local. 
    # Preprocessing, Inference, etc. are now shared because we renamed the edge columns.
    final_col_order = ['Preprocessing', 'Inference', 'Postprocessing', 'Tracking', 'OpFlow', 'Network']
    
    cols_to_use = [c for c in final_col_order if c in plot_df.columns]
    plot_df = plot_df[cols_to_use]

    plt.figure(figsize=(12, 7))
    plot_df.plot(kind='bar', stacked=True, ax=plt.gca())

    plt.title('Latency Breakdown by Experiment', fontsize=24)
    plt.ylabel('Average Latency (ms)', fontsize=22)
    plt.xlabel('Experiment', fontsize=22)
    plt.xticks(rotation=45, ha='right', fontsize=22)
    plt.yticks(fontsize=20)
    
    plt.legend(title='Time Component', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)
    plt.tight_layout()

    plt.savefig('latency_breakdown_final.png')
    print("Plot saved as 'latency_breakdown_final.png'")
else:
    print("No data processed.")