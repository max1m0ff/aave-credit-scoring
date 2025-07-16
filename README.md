# DeFi Wallet Credit Scoring Model

This repository contains my solution for the Aave V2 credit scoring challenge. The project involves analyzing raw transaction data to build a model that assigns a credit score from 0 to 1000 to each wallet, reflecting their on-chain financial health and reliability.

---

## Project Overview

The core challenge was to translate a wallet's complex transaction history into a single, meaningful score. A high score should indicate a reliable, low-risk user, while a low score should flag potentially risky or irresponsible behavior. This required a deep dive into the data to engineer features that could effectively capture these behaviors.

## My Approach: An Interpretable, Rule-Based Model

For this financial application, I concluded that **transparency and interpretability** were just as important as predictive power. Instead of using a "black box" model, I chose to build an **unsupervised, weighted scoring system**.

This approach allowed me to define a clear, logical set of rules based on fundamental financial principles. The final score is a direct, explainable result of these rules, ensuring that any stakeholder can understand why a wallet received a particular score.

The entire process is contained within a single, executable Python script (`credit_scorer.py`).

## Features Engineered

I engineered a set of features for each unique wallet to capture the key dimensions of their on-chain behavior:

* **`wallet_age_days`**: The time between a wallet's first and last transaction. A longer history can indicate a more established and stable user.
* **`transaction_count`**: The total number of interactions with the protocol. This measures the wallet's overall activity level.
* **`total_deposited_usd`**: The total USD value of all assets supplied. This is a strong indicator of the user's capital commitment and trust in the protocol.
* **`total_borrowed_usd` & `total_repaid_usd`**: The absolute amounts borrowed and repaid.
* **`health_ratio`**: A crucial measure of collateralization, calculated as `(Deposits + Redeems) / Borrows`. A high ratio is a primary indicator of financial safety.
* **`repayment_ratio`**: Calculated as `Repays / Borrows`, this measures a user's diligence in paying back their loans. A ratio near 1 indicates responsible borrowing.
* **`liquidation_count`**: The number of times a wallet's position was liquidated. This is the strongest signal of high-risk behavior.

## The Scoring Logic

The credit score is calculated through a weighted sum of the features listed above. I assigned weights based on their conceptual importance to financial health:

| Feature               | Weight | Rationale                                                              |
| --------------------- | :----: | ---------------------------------------------------------------------- |
| **`health_ratio`** | +30%   | The most critical indicator of a wallet's ability to absorb shocks.    |
| **`repayment_ratio`** | +25%   | Directly measures a user's reliability as a borrower.                  |
| **`total_deposited_usd`** | +15%   | Rewards users with more "skin in the game".                            |
| **`wallet_age_days`** | +10%   | Favors established, long-term users.                                   |
| **`transaction_count`** | +5%    | A small reward for being an active protocol participant.               |
| **`liquidation_count`** | -15%   | A strong penalty for the clearest sign of poor risk management.        |

Before applying weights, the features are transformed (e.g., using a log function for monetary values) and scaled to a common range to ensure fair comparison. The final raw score is then scaled to the target range of 0-1000.

## How to Run This Project

1.  **Prerequisites:** Ensure you have Python and the following libraries installed:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
2.  **Setup:** Place the `credit_scorer.py` script and the `user-transactions.json` file in the same folder.
3.  **Execution:** Open a terminal, navigate to the project folder, and run the command:
    ```bash
    python credit_scorer.py
    ```
4.  **Output:** The script will create a new folder named `analysis_results/` containing the final `wallet_credit_scores.csv` file and a `score_distribution.png` graph. It will also print a markdown table in the console for the `analysis.md` file.


