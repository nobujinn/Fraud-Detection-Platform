CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    user_id INT,
    amount FLOAT,
    timestamp TIMESTAMP,
    location VARCHAR(100),
    merchant_type VARCHAR(100),
    is_fraud BOOLEAN
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    transaction_id INT,
    fraud_probability FLOAT,
    decision VARCHAR(50),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);