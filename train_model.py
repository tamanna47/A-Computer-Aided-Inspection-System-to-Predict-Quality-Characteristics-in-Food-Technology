import pandas as pd
import numpy as np

# Define the 26 column names: 5 identifiers/settings + 21 sensors
cols = ['engine_id', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + \
       [f'sensor_{i}' for i in range(1, 22)]

# Load the dataset (you put it in the same directory or under 'data/')
df = pd.read_csv('data/train_FD001.txt', sep='\s+', header=None, names=cols)

# Confirm the shape
print("✅ File loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# Calculate RUL (Remaining Useful Life)
rul_df = df.groupby('engine_id')['cycle'].max().reset_index()
rul_df.columns = ['engine_id', 'max_cycle']
df = df.merge(rul_df, on='engine_id')
df['RUL'] = df['max_cycle'] - df['cycle']

# Label as failure if RUL <= 30
df['failure'] = (df['RUL'] <= 30).astype(int)

# Drop columns with low variance or not useful
drop_columns = ['setting_1', 'setting_2', 'setting_3', 'sensor_1', 'sensor_5',
                'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
df = df.drop(columns=drop_columns)

# Final output check
print("\n✅ Preprocessing complete.")
print("Final shape:", df.shape)
print(df.head())
# Step 3: Feature Selection
X = df.drop(columns=['engine_id', 'cycle', 'RUL', 'failure', 'max_cycle'], errors='ignore')
y = df['failure']

# Step 3.2: Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3.3: Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 3.4: Evaluate model
from sklearn.metrics import classification_report, accuracy_score
y_pred = model.predict(X_test)
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Step 3.5: Save model
import joblib
joblib.dump(model, 'model/predictive_model.pkl')
