import pandas as pd
import os
from pathlib import Path
import numpy as np

class CresciDataLoader:
    """Load and combine Cresci-2017 dataset"""
    
    def __init__(self, data_root):
        """
        Args:
            data_root: Path to cresci-2017 dataset root folder
        """
        self.data_root = Path(data_root)
        
        # Define bot categories
        self.bot_categories = [
            'social_spambots_1',
            'social_spambots_2', 
            'social_spambots_3',
            'traditional_spambots_1',
            'traditional_spambots_2',
            'traditional_spambots_3',
            'traditional_spambots_4',
            'fake_followers'
        ]
        
        self.human_categories = ['genuine_accounts']
    
    def load_users(self):
        """Load all user data and combine with labels"""
        all_users = []
        
        # Load human accounts
        print("Loading genuine accounts...")
        for category in self.human_categories:
            try:
                users_path = self.data_root / category / "users.csv"
                
                if users_path.exists():
                    df = pd.read_csv(users_path)
                    df['label'] = 0  # Human
                    df['category'] = category
                    all_users.append(df)
                    print(f"  ✓ {category}: {len(df)} users")
                else:
                    print(f"  ✗ File not found: {users_path}")
            except Exception as e:
                print(f"  ✗ Error loading {category}: {e}")
        
        # Load bot accounts
        print("\nLoading bot accounts...")
        for category in self.bot_categories:
            try:
                users_path = self.data_root / category / "users.csv"
                
                if users_path.exists():
                    df = pd.read_csv(users_path)
                    df['label'] = 1  # Bot
                    df['category'] = category
                    all_users.append(df)
                    print(f"  ✓ {category}: {len(df)} users")
                else:
                    print(f"  ✗ File not found: {users_path}")
            except Exception as e:
                print(f"  ✗ Error loading {category}: {e}")
        
        # Combine all users
        if all_users:
            combined_df = pd.concat(all_users, ignore_index=True)
            print(f"\n✓ Total users loaded: {len(combined_df)}")
            print(f"  - Humans: {(combined_df['label'] == 0).sum()}")
            print(f"  - Bots: {(combined_df['label'] == 1).sum()}")
            return combined_df
        else:
            print("✗ No users loaded!")
            return None
    
    def load_tweets(self):
        """Load all tweet data (optional - some categories don't have tweets)"""
        all_tweets = []
        
        categories = self.human_categories + self.bot_categories
        
        print("\nLoading tweets...")
        for category in categories:
            try:
                tweets_path = self.data_root / category / "tweets.csv"
                
                if tweets_path.exists():
                    df = pd.read_csv(tweets_path)
                    df['category'] = category
                    all_tweets.append(df)
                    print(f"  ✓ {category}: {len(df)} tweets")
            except Exception as e:
                print(f"  - {category}: No tweets (expected for some categories)")
        
        if all_tweets:
            combined_tweets = pd.concat(all_tweets, ignore_index=True)
            print(f"\n✓ Total tweets loaded: {len(combined_tweets)}")
            return combined_tweets
        else:
            print("\n- No tweets loaded (optional)")
            return None
    
    def get_user_features(self, users_df):
        """Extract relevant features from users dataframe"""
        
        print("\n" + "="*50)
        print("Extracting Features")
        print("="*50)
        
        # Print available columns
        print(f"Available columns: {len(users_df.columns)} columns")
        
        # Key features for bot detection (check which exist)
        desired_features = [
            'id',
            'followers_count',
            'friends_count',
            'statuses_count',
            'favourites_count',
            'listed_count',
            'verified',
            'default_profile',
            'default_profile_image',
            'geo_enabled',
            'profile_use_background_image',
            'protected'
        ]
        
        # Select only existing features
        existing_features = [col for col in desired_features if col in users_df.columns]
        features_df = users_df[existing_features].copy()
        
        # Add label
        features_df['label'] = users_df['label']
        
        print(f"\nBase features selected: {len(existing_features)} features")
        
        # Create computed features
        print("\nComputing derived features...")
        
        if 'followers_count' in features_df.columns and 'friends_count' in features_df.columns:
            # Follower/following ratio (handle division by zero)
            features_df['follower_friend_ratio'] = features_df['followers_count'] / (features_df['friends_count'] + 1)
            print("  ✓ follower_friend_ratio")
        
        # Handle boolean columns BEFORE computing age features
        print("\nProcessing boolean features...")
        bool_columns = ['verified', 'default_profile', 'default_profile_image', 
                       'geo_enabled', 'profile_use_background_image', 'protected']
        for col in bool_columns:
            if col in features_df.columns:
                # Fill NaN first, then convert to int
                features_df[col] = features_df[col].fillna(0).astype(int)
                print(f"  ✓ {col}")
        
        # Handle created_at for account age
        if 'created_at' in users_df.columns:
            try:
                print("\nComputing account age features...")
                # Convert to datetime, removing timezone info
                users_df['created_at_parsed'] = pd.to_datetime(users_df['created_at'], errors='coerce', utc=True)
                users_df['created_at_parsed'] = users_df['created_at_parsed'].dt.tz_localize(None)
                
                # Calculate account age in days (from 2017-01-01 as reference)
                from datetime import datetime
                reference_date = pd.Timestamp('2017-01-01')
                features_df['account_age_days'] = (reference_date - users_df['created_at_parsed']).dt.days
                
                # Handle negative ages (accounts created after reference date)
                features_df['account_age_days'] = features_df['account_age_days'].clip(lower=1)
                features_df['account_age_days'] = features_df['account_age_days'].fillna(365)  # Default to 1 year if missing
                
                # Tweets per day
                if 'statuses_count' in features_df.columns:
                    features_df['tweets_per_day'] = features_df['statuses_count'] / features_df['account_age_days']
                    print("  ✓ account_age_days")
                    print("  ✓ tweets_per_day")
            except Exception as e:
                print(f"  - Could not compute age features: {e}")
        
        # Fill any remaining NaN values
        print("\nHandling missing values...")
        features_df = features_df.fillna(0)
        
        # Replace any inf values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        print(f"\n✓ Final feature count: {len(features_df.columns) - 1} features")
        print(f"Features: {[col for col in features_df.columns if col not in ['label', 'id']]}")
        
        return features_df
    
    def save_processed(self, features_df, output_dir):
        """Save processed data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        output_file = output_path / 'users_processed.csv'
        features_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved processed data to {output_file}")
        
        # Print statistics
        print("\n" + "="*50)
        print("Dataset Statistics")
        print("="*50)
        print(f"Total samples: {len(features_df)}")
        print(f"Features: {len(features_df.columns) - 2}")  # Exclude id and label
        print(f"\nClass distribution:")
        label_counts = features_df['label'].value_counts()
        print(f"  Humans (0): {label_counts[0]} ({label_counts[0]/len(features_df)*100:.1f}%)")
        print(f"  Bots (1): {label_counts[1]} ({label_counts[1]/len(features_df)*100:.1f}%)")
        
        # Feature statistics
        print(f"\nFeature ranges:")
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'label']]
        for col in numeric_cols[:5]:  # Show first 5
            print(f"  {col}: min={features_df[col].min():.2f}, max={features_df[col].max():.2f}, mean={features_df[col].mean():.2f}")
        
        return output_file
