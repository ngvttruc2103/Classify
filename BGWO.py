import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def sigmoid(x):
    """Sigmoid transfer function for converting to binary"""
    return 1 / (1 + np.exp(-10 * (x - 0.5)))

def binary_grey_wolf_optimization(X, y, n_features=16, 
                                   n_wolves=50, max_iter=500):
    """
    Binary Grey Wolf Optimization for feature selection
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - n_features: Number of features to select
    - n_wolves: Number of wolves in the population
    - max_iter: Maximum number of iterations
    
    Returns:
    - Indices of selected features
    """
    n_samples, total_features = X.shape
    
    # Initialize wolf population
    population = np.random.rand(n_wolves, total_features) > 0.5
    
    def fitness_function(solution):
        """
        Evaluate the fitness of a solution (feature subset)
        """
        # Ensure at least one feature is selected
        if np.sum(solution) == 0:
            return 0
        
        # Select features
        selected_features = X[:, solution]
        
        # Use KNN for classification and cross-validation
        clf = KNeighborsClassifier(n_neighbors=5)
        
        # Perform cross-validation
        cv_scores = cross_val_score(clf, selected_features, y, cv=5)
        
        # Penalize solutions with too many features
        feature_penalty = n_features / np.sum(solution)
        
        # Fitness is average accuracy with feature count consideration
        return np.mean(cv_scores) * feature_penalty
    
    # Track best solution
    best_solution = None
    best_fitness = -np.inf
    
    # GWO optimization process
    for iteration in range(max_iter):
        # Calculate fitness for each wolf
        fitness_scores = np.array([fitness_function(wolf) for wolf in population])
        
        # Sort wolves by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        
        # Update alpha, beta, delta wolves
        alpha_idx = sorted_indices[0]
        beta_idx = sorted_indices[1]
        delta_idx = sorted_indices[2]
        
        alpha_wolf = population[alpha_idx]
        beta_wolf = population[beta_idx]
        delta_wolf = population[delta_idx]
        
        # Update best solution
        if fitness_scores[alpha_idx] > best_fitness:
            best_fitness = fitness_scores[alpha_idx]
            best_solution = alpha_wolf
        
        # Update wolf positions 
        a = 2 * (1 - iteration / max_iter)  # Decreasing parameter
        
        for i in range(n_wolves):
            # Distance calculations
            D_alpha = np.abs(population[i] - alpha_wolf)
            D_beta = np.abs(population[i] - beta_wolf)
            D_delta = np.abs(population[i] - delta_wolf)
            
            # Position update
            C1 = 2 * np.random.rand(total_features)
            C2 = 2 * np.random.rand(total_features)
            C3 = 2 * np.random.rand(total_features)
            
            A1 = 2 * a * np.random.rand(total_features) - a
            A2 = 2 * a * np.random.rand(total_features) - a
            A3 = 2 * a * np.random.rand(total_features) - a
            
            D1 = np.abs(C1 * alpha_wolf - population[i])
            D2 = np.abs(C2 * beta_wolf - population[i])
            D3 = np.abs(C3 * delta_wolf - population[i])
            
            X1 = alpha_wolf - A1 * D1
            X2 = beta_wolf - A2 * D2
            X3 = delta_wolf - A3 * D3
            
            new_position = (X1 + X2 + X3) / 3
            
            # Convert to binary
            new_position = sigmoid(new_position)
            population[i]