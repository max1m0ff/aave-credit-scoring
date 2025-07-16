# DeFi Credit Scoring Script 
#
# The script will create a folder named 'analysis_results'

#Necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json
import os
import matplotlib.pyplot as plt

#Input path
INPUT_JSON_PATH = 'user-transactions.json'
OUTPUT_DIR = 'analysis_results'


def load_and_preprocess_data(filepath):
    #Loads transaction data from a JSON file, flattens it, and performs initial cleaning.
    print(f"\nLoading data from {filepath}...")
    if not os.path.exists(filepath):
        print(f"FATAL ERROR: The file was not found at the path: {filepath}")
        print("Please make sure the file is in the same folder as the script, or update the INPUT_JSON_PATH variable.")
        return None

    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        print(f"FATAL ERROR: The file '{filepath}' is not a valid JSON file.")
        return None

    df = pd.json_normalize(raw_data)
    print("Preprocessing data...")
    df.rename(columns={
        'userWallet': 'wallet',
        'actionData.amount': 'amount',
        'actionData.assetPriceUSD': 'price_usd',
        'timestamp': 'timestamp_val' # Rename to avoid conflict with DataFrame's timestamp method
    }, inplace=True, errors='ignore')

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['price_usd'] = pd.to_numeric(df['price_usd'], errors='coerce')
    df['timestamp'] = pd.to_datetime(df['timestamp_val'], unit='s')
    df['amount_usd'] = df['amount'] * df['price_usd']
    df.fillna(0, inplace=True)
    
    print(f"--> Loaded and preprocessed {len(df)} transactions.")
    return df

def engineer_features(df):
    #Engineers features for each wallet based on their transaction history.
    
    print("\nEngineering features for each wallet...")
    wallets = df.groupby('wallet')
    features = {}
    
    for wallet_id, wallet_df in wallets:
        first_tx = wallet_df['timestamp'].min()
        last_tx = wallet_df['timestamp'].max()
        wallet_age_days = (last_tx - first_tx).days
        
        deposits_usd = wallet_df[wallet_df['action'] == 'deposit']['amount_usd'].sum()
        borrows_usd = wallet_df[wallet_df['action'] == 'borrow']['amount_usd'].sum()
        repays_usd = wallet_df[wallet_df['action'] == 'repay']['amount_usd'].sum()
        redeems_usd = wallet_df[wallet_df['action'] == 'redeemunderlying']['amount_usd'].sum()
        
        liquidation_count = len(wallet_df[wallet_df['action'] == 'liquidationcall'])
        health_ratio = (deposits_usd + redeems_usd + 1) / (borrows_usd + 1)
        repayment_ratio = repays_usd / (borrows_usd + 1)

        features[wallet_id] = {
            'wallet_age_days': wallet_age_days,
            'transaction_count': len(wallet_df),
            'total_deposit_usd': round(deposits_usd, 3),
            'total_borrowed_usd': round(borrows_usd, 3),
            'total_repaid_usd': round(repays_usd, 3),
            'liquidation_count': liquidation_count,
            'health_ratio': round(health_ratio, 3),
            'repayment_ratio': round(repayment_ratio, 3)
        }
        
    features_df = pd.DataFrame.from_dict(features, orient='index')
    print(f"--> Engineered features for {len(features_df)} unique wallets.")
    return features_df

def calculate_credit_scores(features_df):
    """
    Calculates a credit score for each wallet based on engineered features.
    """
    print("\nCalculating credit scores...")
    weights = {
        'wallet_age_days': 0.10,
        'transaction_count': 0.05,
        'total_deposit_usd': 0.15,
        'health_ratio': 0.30,
        'repayment_ratio': 0.25,
        'liquidation_count': -0.15
    }

    score_data = features_df.copy()
    for col in ['total_deposit_usd']:
        score_data[col] = np.log1p(score_data[col])
        
    score_data['health_ratio'] = np.log1p(score_data['health_ratio'])
    score_data['health_ratio'].clip(upper=score_data['health_ratio'].quantile(0.99), inplace=True)
    score_data['repayment_ratio'].clip(upper=1, inplace=True)
    score_data['liquidation_count'] = -score_data['liquidation_count']

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(score_data[list(weights.keys())])
    scaled_df = pd.DataFrame(scaled_features, index=score_data.index, columns=list(weights.keys()))

    raw_score = (scaled_df * pd.Series(weights)).sum(axis=1)
    
    final_scaler = MinMaxScaler(feature_range=(0, 1000))
    features_df['credit_score'] = final_scaler.fit_transform(raw_score.values.reshape(-1, 1)).astype(int)

    print("--> Credit scores calculated successfully.")
    return features_df.sort_values(by='credit_score', ascending=False)

def generate_analysis_artifacts(scored_df, output_dir):
    #Generates and saves analysis artifacts like score distribution plots and tables.
  
    print(f"\nGenerating analysis artifacts in '{output_dir}'...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_csv_path = os.path.join(output_dir, 'wallet_credit_scores.csv')
    scored_df.to_csv(output_csv_path)
    print(f"--> Successfully saved wallet scores to '{output_csv_path}'")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scores = scored_df['credit_score']
    scores.hist(bins=20, ax=ax, color='#4a90e2', edgecolor='black')
    
    ax.set_title('Credit Score Distribution Across Wallets', fontsize=16, fontweight='bold')
    ax.set_xlabel('Credit Score (0-1000)', fontsize=12)
    ax.set_ylabel('Number of Wallets', fontsize=12)
    ax.axvline(scores.mean(), color='red', linestyle='dashed', linewidth=2)
    ax.text(scores.mean() * 1.1, ax.get_ylim()[1] * 0.9, f'Mean: {scores.mean():.0f}', color='red')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'score_distribution.png')
    plt.savefig(plot_path)
    print(f"--> Successfully saved score distribution plot to '{plot_path}'")
    plt.close()

    print("\n--- Score Distribution by Range ---")
    bins = range(0, 1001, 100)
    labels = [f"{i}-{i+99}" for i in bins[:-1]]
    scored_df['score_range'] = pd.cut(scored_df['credit_score'], bins=bins, labels=labels, right=False, include_lowest=True)
    distribution = scored_df['score_range'].value_counts().sort_index()
    print("\n| Score Range | Number of Wallets |")
    print("|-------------|-------------------|")
    for score_range, count in distribution.items():
        print(f"| {score_range:<11} | {count:<17} |")

#Execute the Pipeline
def run_pipeline():
    #This function executes all the steps in the correct order.
    
    transaction_df = load_and_preprocess_data(INPUT_JSON_PATH)
    
    if transaction_df is not None:
        features_df = engineer_features(transaction_df)
        scored_df = calculate_credit_scores(features_df)
        generate_analysis_artifacts(scored_df, OUTPUT_DIR)
        
        print("\n\n✅ Pipeline finished successfully!")
        print(f"Check the '{OUTPUT_DIR}' folder for the output files.")

if __name__ == "__main__":
    run_pipeline()

