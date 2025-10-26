# climate_emissions_predictor.py (Enhanced)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

class ClimateEmissionsPredictor:
    """
    AI Solution for SDG 13: Climate Action
    Predicts CO2 emissions and provides sustainability insights
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.features = [
            'gdp_per_capita', 'population_millions', 'renewable_energy_percent',
            'industrial_output', 'vehicle_per_1000', 'forest_coverage_percent',
            'temperature_change'
        ]
        
    def load_or_generate_data(self, csv_path='sample_data/sample_emissions_data.csv'):
        """
        Load data from CSV if available, otherwise generate sample data
        """
        if os.path.exists(csv_path):
            print(f"📂 Loading data from {csv_path}")
            return pd.read_csv(csv_path)
        else:
            print("📊 Generating sample climate data...")
            return self.generate_sample_data()
    
    def generate_sample_data(self, n_countries=100):
        """
        Generate synthetic climate and emissions data for demonstration
        """
        np.random.seed(42)
        
        regions = ['Europe', 'Asia', 'Africa', 'Americas', 'Oceania']
        data = {
            'country': [f'Country_{i}' for i in range(n_countries)],
            'region': np.random.choice(regions, n_countries),
            'gdp_per_capita': np.random.normal(15000, 8000, n_countries),
            'population_millions': np.random.uniform(1, 150, n_countries),
            'renewable_energy_percent': np.random.uniform(5, 80, n_countries),
            'industrial_output': np.random.normal(50, 20, n_countries),
            'vehicle_per_1000': np.random.normal(300, 150, n_countries),
            'forest_coverage_percent': np.random.uniform(10, 70, n_countries),
            'temperature_change': np.random.normal(1.2, 0.5, n_countries)
        }
        
        # Generate realistic CO2 emissions
        co2_emissions = (
            data['gdp_per_capita'] * 0.0003 +
            data['population_millions'] * 0.4 +
            data['industrial_output'] * 0.5 -
            data['renewable_energy_percent'] * 0.6 +
            data['vehicle_per_1000'] * 0.2 -
            data['forest_coverage_percent'] * 0.3 +
            np.random.normal(0, 2, n_countries)
        )
        
        data['co2_emissions_mt'] = np.maximum(co2_emissions, 1)
        
        df = pd.DataFrame(data)
        
        # Save sample data for future use
        os.makedirs('sample_data', exist_ok=True)
        df.to_csv('sample_data/sample_emissions_data.csv', index=False)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the dataset for machine learning"""
        X = df[self.features]
        y = df['co2_emissions_mt']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2 Score': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def predict_emissions_reduction(self, country_data, intervention_scenarios):
        """
        Predict emissions reduction for different intervention scenarios
        """
        predictions = {}
        current_emission = country_data['co2_emissions_mt']
        country_features = country_data[self.features].values
        
        for scenario, changes in intervention_scenarios.items():
            modified_features = country_features.copy()
            
            # Apply changes based on scenario
            for feature, change in changes.items():
                feature_idx = self.features.index(feature)
                if '%' in str(change):
                    # Percentage change
                    pct_change = float(change.strip('%')) / 100
                    modified_features[feature_idx] *= (1 + pct_change)
                else:
                    # Absolute change
                    modified_features[feature_idx] += change
            
            # Predict new emissions
            scaled_data = self.scaler.transform([modified_features])
            new_emission = self.model.predict(scaled_data)[0]
            predictions[scenario] = new_emission
        
        return predictions, current_emission
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.feature_importance, x='importance', y='feature')
        plt.title('Feature Importance in CO2 Emissions Prediction')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('assets/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_predictions_vs_actual(self, y_test, y_pred):
        """Plot predictions vs actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual CO2 Emissions (MT)')
        plt.ylabel('Predicted CO2 Emissions (MT)')
        plt.title('Actual vs Predicted CO2 Emissions')
        plt.tight_layout()
        plt.savefig('assets/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main demonstration function"""
    print("🌍 AI for Climate Action: CO2 Emissions Predictor")
    print("=" * 50)
    
    # Initialize predictor
    climate_ai = ClimateEmissionsPredictor()
    
    # Load or generate data
    df = climate_ai.load_or_generate_data()
    print(f"Dataset shape: {df.shape}")
    
    # Display sample data
    print("\nSample data:")
    print(df[['country', 'region', 'co2_emissions_mt', 'renewable_energy_percent']].head())
    
    # Preprocess data
    print("\n🔄 Preprocessing data...")
    X_train, X_test, y_train, y_test = climate_ai.preprocess_data(df)
    
    # Train model
    print("🤖 Training Random Forest model...")
    climate_ai.train_model(X_train, y_train)
    
    # Evaluate model
    print("📈 Evaluating model performance...")
    metrics, y_pred = climate_ai.evaluate_model(X_test, y_test)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    print("\n🔍 Feature Importance Analysis:")
    print(climate_ai.feature_importance)
    
    # Create assets directory
    os.makedirs('assets', exist_ok=True)
    
    # Demonstrate intervention scenarios
    print("\n🌱 Sustainability Intervention Analysis")
    print("=" * 40)
    
    # Select a sample country for analysis
    sample_country = df.iloc[0]
    
    # Define intervention scenarios
    intervention_scenarios = {
        'Current Policy': {},
        'Renewable Energy Boost': {'renewable_energy_percent': 20},
        'Sustainable Transport': {'vehicle_per_1000': -50, 'renewable_energy_percent': 10},
        'Green Economy': {
            'renewable_energy_percent': 25,
            'forest_coverage_percent': 15,
            'industrial_output': -10
        }
    }
    
    # Predict emissions for each scenario
    predictions, current_emission = climate_ai.predict_emissions_reduction(
        sample_country, 
        intervention_scenarios
    )
    
    print(f"\nCurrent emissions for {sample_country['country']} ({sample_country['region']}): {current_emission:.2f} MT")
    
    print("\nEmissions under different scenarios:")
    for scenario, emission in predictions.items():
        reduction = ((current_emission - emission) / current_emission) * 100
        print(f"  {scenario}: {emission:.2f} MT ({reduction:+.1f}% reduction)")
    
    # Visualizations
    print("\n📊 Generating visualizations...")
    climate_ai.plot_feature_importance()
    climate_ai.plot_predictions_vs_actual(y_test, y_pred)
    
    # Ethical considerations
    print("\n⚖️ Ethical Considerations:")
    ethical_points = [
        "• Ensure fair representation of developing vs developed countries",
        "• Consider economic impacts of emission reduction policies", 
        "• Address potential bias in training data",
        "• Promote equitable climate action solutions"
    ]
    
    for point in ethical_points:
        print(point)
    
    print("\n🎯 SDG 13 Impact: This AI solution helps policymakers:")
    impact_points = [
        "• Predict CO2 emissions based on key factors",
        "• Test different climate intervention scenarios", 
        "• Identify most impactful sustainability measures",
        "• Support evidence-based climate policy decisions"
    ]
    
    for point in impact_points:
        print(f"   {point}")

if __name__ == "__main__":
    main()