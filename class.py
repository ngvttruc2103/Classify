import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time
import warnings
import os



# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    # Suppress XGBoost warnings
    xgb.set_config(verbosity=0)
    XGBOOST_AVAILABLE = True
    print("XGBoost is available (warnings suppressed)")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available - using GradientBoostingClassifier instead")

class BearingFaultClassifier:
    """
    Complete bearing fault classifier with all four main algorithms
    """
    
    def __init__(self, suppress_warnings=True):
        if suppress_warnings:
            warnings.filterwarnings('ignore')
        
        # Initialize classifiers with optimized parameters
        self.classifiers = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            ),
            'SVM': SVC(
                kernel='rbf', 
                C=1.0,
                gamma='scale',
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='euclidean',
                n_jobs=-1
            ),
        }
        
        if XGBOOST_AVAILABLE:
            self.classifiers['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,  # Suppress output
                n_jobs=-1
            )
        else:
            self.classifiers['Gradient Boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        
        self.results = {}
        self.trained_models = {}
        self.scaler = StandardScaler()
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, class_names=None, cv_folds=5):
        """
        Train and evaluate all classifiers with comprehensive metrics
        """
        print("Training and evaluating classifiers...")
        print("=" * 60)
        
        for name, classifier in self.classifiers.items():
            print(f"\nTraining {name}...")
            
            # Time the training
            start_time = time.time()
            
            # Suppress specific warnings for each classifier
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Make predictions
            start_time = time.time()
            y_pred = classifier.predict(X_test)
            prediction_time = time.time() - start_time
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(classifier, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(classifier, 'feature_importances_'):
                feature_importance = classifier.feature_importances_
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'cv_scores': cv_scores,
                'training_time': training_time,
                'prediction_time': prediction_time,
                'y_pred': y_pred,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'feature_importance': feature_importance
            }
            
            self.trained_models[name] = classifier
            
            # Display results
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  CV Score: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
            print(f"  Training time: {training_time:.3f}s")
            print(f"  Prediction time: {prediction_time:.4f}s")
        
        return self.results
    
    def plot_comprehensive_results(self, class_names=None, figsize=(20, 16)):
        """
        Create comprehensive visualization of results
        """
        if not self.results:
            print("No results to plot. Run train_and_evaluate first.")
            return
        
        fig = plt.figure(figsize=figsize)
        
        # Create a grid of subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        models = list(self.results.keys())
        
        # 1. Performance Metrics Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in models]
            ax1.bar(x + i*width - 1.5*width, values, width, label=metric.title(), alpha=0.8)
        
        ax1.set_xlabel('Classifier')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # 2. Training Time Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        training_times = [self.results[model]['training_time'] for model in models]
        prediction_times = [self.results[model]['prediction_time'] for model in models]
        
        x = np.arange(len(models))
        ax2.bar(x - 0.2, training_times, 0.4, label='Training Time', alpha=0.8)
        ax2.bar(x + 0.2, prediction_times, 0.4, label='Prediction Time', alpha=0.8)
        ax2.set_xlabel('Classifier')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Time Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cross-Validation Scores
        ax3 = fig.add_subplot(gs[0, 2])
        cv_data = []
        cv_labels = []
        for model in models:
            cv_scores = self.results[model]['cv_scores']
            cv_data.extend(cv_scores)
            cv_labels.extend([model] * len(cv_scores))
        
        cv_df = pd.DataFrame({'CV_Score': cv_data, 'Model': cv_labels})
        sns.boxplot(data=cv_df, x='Model', y='CV_Score', ax=ax3)
        ax3.set_title('Cross-Validation Score Distribution')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4-7. Confusion Matrices for each model
        if class_names is None:
            unique_classes = np.unique(np.concatenate([self.results[model]['y_pred'] for model in models]))
            class_names = [f'Class {i}' for i in unique_classes]
        
        for idx, model in enumerate(models):
            if idx < 4:  # Only show first 4 confusion matrices
                row = 1 + idx // 2
                col = idx % 2
                ax = fig.add_subplot(gs[row, col])
                cm = self.results[model]['confusion_matrix']
                
                # Normalize confusion matrix
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                           xticklabels=class_names, yticklabels=class_names)
                ax.set_title(f'Confusion Matrix - {model}')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # 8. Feature Importance (if available)
        tree_models = [model for model in models if self.results[model]['feature_importance'] is not None]
        if tree_models:
            ax8 = fig.add_subplot(gs[2, 2])
            
            # Show feature importance for the best tree-based model
            best_tree_model = max(tree_models, key=lambda x: self.results[x]['accuracy'])
            importance = self.results[best_tree_model]['feature_importance']
            
            # Show top 10 features
            top_indices = np.argsort(importance)[-10:]
            ax8.barh(range(len(top_indices)), importance[top_indices])
            ax8.set_yticks(range(len(top_indices)))
            ax8.set_yticklabels([f'Feature {i}' for i in top_indices])
            ax8.set_xlabel('Importance')
            ax8.set_title(f'Top 10 Feature Importance - {best_tree_model}')
            ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print best model summary
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        print(f"\nBEST PERFORMING MODEL: {best_model}")
        print("=" * 50)
        print(f"Accuracy: {self.results[best_model]['accuracy']:.4f}")
        print(f"Precision: {self.results[best_model]['precision']:.4f}")
        print(f"Recall: {self.results[best_model]['recall']:.4f}")
        print(f"F1-Score: {self.results[best_model]['f1_score']:.4f}")
        print(f"CV Score: {self.results[best_model]['cv_mean']:.4f} +/- {self.results[best_model]['cv_std']:.4f}")
        print(f"Training Time: {self.results[best_model]['training_time']:.3f}s")
    
    def print_detailed_report(self):
        """
        Print a detailed classification report
        """
        print("\nDETAILED CLASSIFICATION REPORT")
        print("=" * 80)
        
        # Create summary table
        df_results = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'Precision': [self.results[model]['precision'] for model in self.results.keys()],
            'Recall': [self.results[model]['recall'] for model in self.results.keys()],
            'F1-Score': [self.results[model]['f1_score'] for model in self.results.keys()],
            'CV Mean': [self.results[model]['cv_mean'] for model in self.results.keys()],
            'CV Std': [self.results[model]['cv_std'] for model in self.results.keys()],
            'Train Time(s)': [self.results[model]['training_time'] for model in self.results.keys()],
            'Pred Time(s)': [self.results[model]['prediction_time'] for model in self.results.keys()]
        })
        
        # Format the dataframe for better display
        pd.set_option('display.precision', 4)
        pd.set_option('display.width', None)
        pd.set_option('display.max_columns', None)
        
        print(df_results.to_string(index=False))
        
        return df_results
    
    def save_results(self, filename='bearing_classification_results.csv'):
        """
        Save results to CSV file
        """
        df = self.print_detailed_report()
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")

def generate_realistic_bearing_data(n_samples=2000, n_features=40, n_classes=10, noise_level=0.1):
    """
    Generate more realistic bearing fault data
    """
    print("Generating realistic bearing fault data...")
    
    np.random.seed(42)
    
    # Create features that simulate bearing characteristics
    X = np.zeros((n_samples, n_features))
    y = np.random.randint(0, n_classes, n_samples)
    
    for i in range(n_classes):
        mask = y == i
        n_class_samples = np.sum(mask)
        
        if i == 0:  # Normal condition
            X[mask] = np.random.randn(n_class_samples, n_features) * 0.5
        else:  # Fault conditions
            # Create distinct patterns for different fault types
            fault_pattern = np.random.randn(n_features) * (i * 0.3)
            
            for j in range(n_class_samples):
                sample = np.random.randn(n_features) * noise_level
                sample += fault_pattern
                
                # Add some frequency-like characteristics
                sample[:10] += np.sin(np.linspace(0, 2*np.pi*i, 10)) * (i * 0.2)
                
                X[mask][j] = sample
    
    # Add overall noise
    X += np.random.randn(n_samples, n_features) * noise_level
    
    # Define realistic class names
    class_names = ['Normal', 'IR007', 'IR014', 'IR021', 'B007', 'B014', 'B021', 'OR007', 'OR014', 'OR021']
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"{n_classes} classes: {class_names[:n_classes]}")
    
    return X, y, class_names[:n_classes]

def main():
    """
    Main function to run the complete bearing fault classification
    """
    print("ADVANCED BEARING FAULT CLASSIFICATION SYSTEM")
    print("=" * 70)
    
    # Generate realistic sample data
    X, y, class_names = generate_realistic_bearing_data(
        n_samples=3000, 
        n_features=40, 
        n_classes=10, 
        noise_level=0.15
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nDataset Information:")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    print(f"  Training set: {X_train_scaled.shape[0]} samples")
    print(f"  Test set: {X_test_scaled.shape[0]} samples")
    
    # Initialize classifier
    classifier = BearingFaultClassifier(suppress_warnings=True)
    
    # Train and evaluate
    print(f"\nAvailable Classifiers: {list(classifier.classifiers.keys())}")
    
    results = classifier.train_and_evaluate(
        X_train_scaled, X_test_scaled, y_train, y_test, 
        class_names=class_names, cv_folds=10
    )
    
    # Plot results
    classifier.plot_comprehensive_results(class_names)
    
    # Print detailed report
    classifier.print_detailed_report()
    
    # Save results
    classifier.save_results()
    
    print("\n" + "="*70)
    print("CLASSIFICATION COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return results, classifier

if __name__ == "__main__":
    results, classifier = main()
    
    # Additional analysis
    print("\nADDITIONAL ANALYSIS:")
    print("-" * 30)
    
    # Find the most balanced model (best F1 score)
    best_f1_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
    print(f"Most balanced model (highest F1): {best_f1_model}")
    print(f"F1 Score: {results[best_f1_model]['f1_score']:.4f}")
    
    # Find the fastest model
    fastest_model = min(results.keys(), key=lambda x: results[x]['training_time'])
    print(f"Fastest training model: {fastest_model}")
    print(f"Training time: {results[fastest_model]['training_time']:.4f}s")
    
    # Find the most consistent model (lowest CV std)
    most_consistent = min(results.keys(), key=lambda x: results[x]['cv_std'])
    print(f"Most consistent model (lowest CV std): {most_consistent}")
    print(f"CV Score: {results[most_consistent]['cv_mean']:.4f} +/- {results[most_consistent]['cv_std']:.4f}")
    
    print("\nClassification pipeline completed!")