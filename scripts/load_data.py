import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.data_loader import CresciDataLoader

def main():
    print("="*60)
    print("NeuroShield: Loading Cresci-2017 Dataset")
    print("="*60)
    
    # Path to your dataset
    data_path = "data/raw/cresci-2017"
    
    # Initialize loader
    loader = CresciDataLoader(data_path)
    
    # Load users
    users_df = loader.load_users()
    
    if users_df is None:
        print("\n✗ Failed to load dataset!")
        return
    
    # Optionally load tweets
    tweets_df = loader.load_tweets()
    
    # Extract features
    features_df = loader.get_user_features(users_df)
    
    # Show sample
    print("\n" + "="*60)
    print("Sample Data (first 5 rows)")
    print("="*60)
    print(features_df.head())
    
    print("\n" + "="*60)
    print("Data Types")
    print("="*60)
    print(features_df.dtypes)
    
    # Save processed data
    output_file = loader.save_processed(features_df, "data/processed")
    
    print("\n" + "="*60)
    print("✓ Data loading complete!")
    print("="*60)
    print(f"Processed data saved to: {output_file}")
    print("\nNext step: Build graph structure from this data")

if __name__ == "__main__":
    main()
