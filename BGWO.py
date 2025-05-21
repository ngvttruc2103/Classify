import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

class BGWO:
    """Binary Grey Wolf Optimization for feature selection
    
    This implementation is based on Emary et al. (2016) paper 
    "Binary grey wolf optimization approaches for feature selection".
    
    The algorithm is designed to select an optimal subset of features by:
    1. Reducing from 20 to 9 types per file
    2. Reducing from 40 to 16 types for combined feature sets
    
    Key statistical features that are prioritized:
    - Average (Mean)
    - Root Mean Square (RMS)
    - Form
    - Peak to Peak
    - Variance
    - Mean Square
    - Margin
    - Impulse
    - Median Frequency
    """
    
    def __init__(self, n_wolves=50, max_iter=500):
        """Initialize the BGWO algorithm
        
        Args:
            n_wolves (int): Number of grey wolves in the pack (default: 50)
            max_iter (int): Maximum number of iterations (default: 500)
        """
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.best_wolf_position = None
        self.best_wolf_fitness = float('inf')
        self.convergence_curve = []
        
    def initialize_population(self, n_features):
        """Initialize binary wolf positions randomly
        
        Args:
            n_features (int): Total number of features
            
        Returns:
            np.ndarray: Binary population matrix of shape (n_wolves, n_features)
        """
        return np.random.randint(0, 2, size=(self.n_wolves, n_features))
    
    def calculate_fitness(self, X, y, positions, alpha=0.99, beta=0.01):
        """Calculate fitness using KNN classifier
        
        The fitness function balances classification accuracy and feature reduction:
        Fitness = α * error_rate + β * (|R|/|C|)
        
        Where:
        - error_rate is the classification error (1 - accuracy)
        - |R| is the length of selected feature subset
        - |C| is the total number of features
        - α and β control the importance of accuracy vs. feature count
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            positions (np.ndarray): Binary positions of wolves
            alpha (float): Weight for error rate (default: 0.99)
            beta (float): Weight for feature count (default: 0.01)
            
        Returns:
            np.ndarray: Fitness values for each wolf
        """
        fitness_values = []
        
        for wolf_position in positions:
            # Skip if no features are selected
            if np.sum(wolf_position) == 0:
                fitness_values.append(1.0)  # Worst fitness
                continue
                
            # Select features based on wolf position
            selected_features = X[:, wolf_position == 1]
            
            # Use 5-fold cross-validation to evaluate
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            accuracies = []
            
            for train_idx, val_idx in kf.split(selected_features):
                X_train, X_val = selected_features[train_idx], selected_features[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train KNN classifier
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)
                
                # Calculate accuracy
                accuracy = knn.score(X_val, y_val)
                accuracies.append(accuracy)
            
            mean_accuracy = np.mean(accuracies)
            error_rate = 1 - mean_accuracy
            
            # Calculate feature reduction term (|R|/|C|)
            feature_ratio = np.sum(wolf_position) / len(wolf_position)
            
            # Combined fitness (minimize error and feature count)
            fitness = alpha * error_rate + beta * feature_ratio
            fitness_values.append(fitness)
            
        return np.array(fitness_values)
    
    def update_wolf_position(self, alpha_pos, beta_pos, delta_pos, a):
        """Update wolf position using binary GWO approach
        
        This uses the sigmoid transfer function to convert continuous 
        values to binary (approach 2 from Emary et al. 2016)
        
        Args:
            alpha_pos (np.ndarray): Position of alpha wolf
            beta_pos (np.ndarray): Position of beta wolf
            delta_pos (np.ndarray): Position of delta wolf
            a (float): Parameter controlling exploration vs. exploitation
            
        Returns:
            np.ndarray: Updated binary position
        """
        new_positions = np.zeros_like(alpha_pos)
        
        # Calculate A and C vectors
        r1 = np.random.random()
        r2 = np.random.random()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        
        r1 = np.random.random()
        r2 = np.random.random()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        
        r1 = np.random.random()
        r2 = np.random.random()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        
        # Binary updates using sigmoid function
        for d in range(len(alpha_pos)):
            # Calculate D values (distance from wolves to the current position)
            D_alpha = np.abs(C1 * alpha_pos[d] - alpha_pos[d])
            D_beta = np.abs(C2 * beta_pos[d] - beta_pos[d])
            D_delta = np.abs(C3 * delta_pos[d] - delta_pos[d])
            
            # Calculate steps toward alpha, beta, delta
            X1 = alpha_pos[d] - A1 * D_alpha
            X2 = beta_pos[d] - A2 * D_beta
            X3 = delta_pos[d] - A3 * D_delta
            
            # Average the positions (Eq. 24 in paper)
            X_avg = (X1 + X2 + X3) / 3
            
            # Apply sigmoid function (Eq. 25 in paper)
            sigmoid_val = 1 / (1 + np.exp(-10 * (X_avg - 0.5)))
            
            # Stochastic binary decision
            if np.random.random() < sigmoid_val:
                new_positions[d] = 1
            else:
                new_positions[d] = 0
                
        return new_positions
    
    def fit(self, X, y, feature_names=None, target_features=None):
        """Main BGWO algorithm for feature selection
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            feature_names (list): Names of features (optional)
            target_features (int): Target number of features to select (optional)
            
        Returns:
            tuple: Selected feature indices and convergence curve
        """
        n_features = X.shape[1]
        
        # Set target number of features if not provided
        if target_features is None:
            # Default: reduce to ~9 features for single signals or ~16 for combined
            target_features = 9 if n_features <= 20 else 16
        
        print(f"BGWO starting with {n_features} features, targeting ~{target_features} features")
        
        # Initialize population
        positions = self.initialize_population(n_features)
        
        # Main loop
        for iteration in range(self.max_iter):
            # Calculate fitness for all wolves
            fitness_values = self.calculate_fitness(X, y, positions)
            
            # Sort wolves based on fitness
            sorted_indices = np.argsort(fitness_values)
            sorted_fitness = fitness_values[sorted_indices]
            sorted_positions = positions[sorted_indices]
            
            # Update best wolves
            alpha_pos = sorted_positions[0].copy()
            beta_pos = sorted_positions[1].copy()
            delta_pos = sorted_positions[2].copy()
            
            # Update best solution if found
            if sorted_fitness[0] < self.best_wolf_fitness:
                self.best_wolf_fitness = sorted_fitness[0]
                self.best_wolf_position = alpha_pos.copy()
                
                # Print progress
                n_selected = np.sum(self.best_wolf_position)
                if iteration % 50 == 0 or n_selected == target_features:
                    print(f"Iteration {iteration}: Best fitness = {self.best_wolf_fitness:.6f}, Selected {n_selected} features")
            
            # Save convergence info
            self.convergence_curve.append(self.best_wolf_fitness)
            
            # Update parameter a (linearly decreased from 2 to 0)
            a = 2 - iteration * (2 / self.max_iter)
            
            # Update positions of all wolves
            for i in range(self.n_wolves):
                positions[i] = self.update_wolf_position(alpha_pos, beta_pos, delta_pos, a)
        
        # Get final selected features
        selected_indices = np.where(self.best_wolf_position == 1)[0]
        
        # Print selected features
        n_original = n_features
        n_selected = len(selected_indices)
        reduction_percent = ((n_original - n_selected) / n_original) * 100
        
        print(f"\nBGWO completed: Selected {n_selected} features out of {n_original} ({reduction_percent:.1f}% reduction)")
        
        if feature_names is not None:
            selected_names = [feature_names[i] for i in selected_indices]
            print("\nSelected features:")
            for idx, feature in enumerate(selected_names):
                print(f"{idx+1}. {feature}")
            
            # Count statistical features by type
            if n_original <= 20:  # Single signal features
                # These are the key feature groups we're looking for
                key_features = ["Mean", "RMS", "Form", "Peak_to_Peak", "Variance", 
                               "Mean_Square", "Margin", "Impulse", "Median_Frequency"]
                
                counts = {feature: 0 for feature in key_features}
                for feature in selected_names:
                    for key in key_features:
                        if key in feature:
                            counts[key] += 1
                
                print("\nSelected statistical feature types:")
                for feature, count in counts.items():
                    status = "✓" if count > 0 else "✗"
                    print(f"{status} {feature}: {count}")
        
        return selected_indices, self.convergence_curve



