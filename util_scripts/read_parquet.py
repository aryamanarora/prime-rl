import os
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

def read_parquet_files_from_folder(folder_path):
    """
    Read all parquet files from a folder and combine them into a single DataFrame.
    
    Args:
        folder_path (str): Path to the folder containing parquet files
    
    Returns:
        pandas.DataFrame: Combined DataFrame from all parquet files
    """
    # Convert to Path object for better path handling
    folder = Path(folder_path)
    
    # Get all parquet files in the folder
    parquet_files = list(folder.glob('*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return None
    
    # Read and combine all parquet files
    dfs = []
    for file in parquet_files:
        print(f"Reading {file}")
        df = pd.read_parquet(file)
        dfs.append(df)
    
    # Combine all dataframes
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Successfully read {len(parquet_files)} parquet files with {len(combined_df)} total rows")
        return combined_df
    else:
        return None

if __name__ == "__main__":
    # Example usage
    folder_path = "../data_rollout/step_0"
    df = read_parquet_files_from_folder(folder_path)
    
    if df is not None:
        # Do something with the combined DataFrame
        print(df.head())
        print(f"DataFrame shape: {df.shape}")
        
        #for i in range(20):
        
        #    print(tok.decode(df.loc[i, "output_tokens"][-50:]))
        #    print("\n\n####\n\n")
