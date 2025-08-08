"""
Causal Bayesian Optimization with Unknown Graphs (CBO-U)
========================================================

This implementation is based on the algorithm presented in the paper:
"Causal Bayesian Optimization" (Zhang et al.)

Your experiment must be carefully designed to ensure:

You fully specify all potentially relevant direct causes of your target.

You have a setup where direct interventions (experimentally manipulating independent variables directly) are feasible.

You explicitly exclude scenarios where hidden factors might independently influence multiple observed variables simultaneously.

Core Algorithmic Assumptions:
----------------------------
1. Causal Structure: The algorithm assumes there is an underlying causal graph where
   a target variable Y is influenced by some parent nodes. The graph structure is unknown
   and must be learned.

2. Interventional vs. Observational Data: The algorithm distinguishes between:
   - Observational data: Passively collected data from the system
   - Interventional data: Data collected by actively setting values of variables

3. Experimental Prior: The algorithm uses a uniform prior over potential causal structures
   initially, and updates this prior as interventions are performed. No strong prior knowledge
   about the causal structure is assumed.

4. Parent Discovery: The algorithm uses bootstrap sampling and a simplified correlation-based
   approach to estimate potential parent sets of the target variable.

5. GP Models: The algorithm fits Gaussian Process models for different potential parent sets
   to predict the target variable's value under interventions.

6. Acquisition Function: The algorithm uses a Lower Confidence Bound (LCB) acquisition function
   that accounts for uncertainty in both:
   - The causal structure (epistemic uncertainty)
   - The prediction given a causal structure (aleatoric uncertainty)

7. No Confounders: A key assumption is the absence of unmeasured confounders affecting the target
   variable Y. The algorithm assumes causal sufficiency - that all common causes of variables in
   the system are included in the model. This allows for consistent estimation of causal effects
   without needing to correct for confounding bias.

8. Hard Interventions: The algorithm assumes that hard interventions (setting a variable to a
   fixed value and removing all its causal dependencies) are optimal for causal discovery and
   optimization. This assumption is particularly justified when no spouse nodes (nodes sharing
   a common child with the target node) exist in the graph, thereby excluding bidirected edges
   or shared hidden common causes between intervention targets and the outcome variable.

Implementation Details:
---------------------
- Synthetic Data: This implementation generates synthetic causal graphs using an Erdos-Renyi
  random graph model with probability p_edge for each potential edge.

- True Parents: For experimental evaluation, the true parent set is known and used to generate
  the synthetic data, but the algorithm does not have access to this information.

- Evaluation: The algorithm is evaluated based on:
  1. Ability to identify the true parents of the target variable
  2. Ability to find interventions that minimize the target variable

Experimental Parameters:
----------------------
- n_nodes: Number of nodes in the causal graph
- p_edge: Probability of an edge in the Erdos-Renyi graph
- n_bootstrap: Number of bootstrap samples for parent estimation
- n_interventions: Number of interventions to perform
- use_seed: Whether to use a fixed random seed (1) for reproducibility

Algorithm Workflow:
-----------------
1. Generate observational data from the true causal graph
2. Estimate potential parent sets using bootstrap sampling
3. Fit GP models for each potential parent set
4. For T interventions:
   a. Compute acquisition function accounting for structural uncertainty
   b. Select and perform optimal intervention
   c. Update dataset with new observation
   d. Periodically update parent set probabilities and refit models

Output:
------
The algorithm produces visualizations showing:
- The true causal graph
- The learned causal structure
- Parent probabilities and their evolution over time
- Intervention results
- Evaluation metrics for causal discovery
"""

import numpy as np
import networkx as nx
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.stats import norm
from collections import Counter
import random
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Suppress certain convergence warnings
warnings.filterwarnings("ignore", category=Warning, 
                       message="The optimal value found for dimension 0 of parameter*")
warnings.filterwarnings("ignore", category=Warning, 
                       message="lbfgs failed to converge*")

class CausalBayesianOptimization:
    """
    Implementation of Algorithm 1: CBO-U (Causal Bayesian Optimization with Unknown Causal Graphs)
    """
    def __init__(self, n_nodes=50, p_edge=0.1, n_bootstrap=100, n_interventions=20):
        """
        Initialize the CBO-U algorithm.
        
        Args:
            n_nodes: Number of nodes in the graph
            p_edge: Probability of edge in the Erdos-Renyi graph
            n_bootstrap: Number of bootstrap samples for parent estimation
            n_interventions: Number of interventions to perform (T in Algorithm 1)
        """
        self.n_nodes = n_nodes
        self.p_edge = p_edge
        self.n_bootstrap = n_bootstrap
        self.n_interventions = n_interventions  # T in Algorithm 1
        
        # Create a more meaningful synthetic dataset with actual causal relationships
        # Generate true causal graph
        self.true_G = nx.DiGraph()
        self.true_G.add_nodes_from(range(n_nodes))
        
        # Add random edges with probability p_edge
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and random.random() < p_edge:
                    self.true_G.add_edge(i, j)
        
        # Randomly choose a target node Y
        self.Y = np.random.choice(n_nodes)
        self.manipulative_vars = [i for i in range(n_nodes) if i != self.Y]
        
        # Find true parents of Y
        self.true_parents = list(self.true_G.predecessors(self.Y))
        print(f"True parents of node {self.Y}: {self.true_parents}")
        
        # Generate synthetic data with actual causal relationships
        self.D_obs = self.generate_synthetic_data(200)
        
        # Initialize interventional dataset D_int
        self.D_int = self.generate_synthetic_data(2)
        
        # Initialize GP kernel - create a fresh kernel for each model instead of cloning
        self.base_kernel = {
            'constant_value': 1.0,
            'constant_bounds': (1e-2, 1e2),
            'length_scale': 1.0,
            'length_scale_bounds': (1e-2, 1e2),
            'nu': 2.5,
            'noise_level': 1e-5,
            'noise_bounds': (1e-10, 1e-1)
        }
        
        # Dictionary to store multiple GP models (one for each potential parent set)
        self.gp_models = {}
        
        # Scalers for data normalization
        self.X_scaler = StandardScaler()
        self.Y_scaler = StandardScaler()
        
        # Initial uniform prior over all nodes
        self.node_probabilities = np.ones(n_nodes) / n_nodes
        
        # Track all estimated parent sets from bootstrap samples
        self.all_parent_sets = []
        
        # Store potential parent sets with their probabilities
        # This is P(g) in the paper - the probability of each parent set configuration
        self.parent_sets_with_probs = []
        
        # Keep track of the best intervention found
        self.best_intervention = None
        self.best_value = float('inf')
        
        # Keep track of parent probability history for visualization
        self.parent_probability_history = []
        self.intervention_history = []
        self.y_value_history = []
    
    def create_kernel(self):
        """Create a fresh kernel with the base parameters"""
        return (ConstantKernel(self.base_kernel['constant_value'], 
                              constant_value_bounds=self.base_kernel['constant_bounds']) * 
                Matern(length_scale=self.base_kernel['length_scale'], 
                      length_scale_bounds=self.base_kernel['length_scale_bounds'], 
                      nu=self.base_kernel['nu']) +
                WhiteKernel(noise_level=self.base_kernel['noise_level'], 
                           noise_level_bounds=self.base_kernel['noise_bounds']))
    
    def generate_synthetic_data(self, n_samples):
        """
        Generate synthetic data where Y actually depends on its parents
        """
        data = np.random.randn(n_samples, self.n_nodes)
        
        # Make Y depend on its true parents with some noise
        for i in range(n_samples):
            # Y = weighted sum of parents + noise
            parent_contribution = 0
            for parent in self.true_parents:
                # Each parent has a random weight between 0.5 and 1.5
                weight = 0.5 + random.random()
                parent_contribution += weight * data[i, parent]
            
            # Add noise to Y
            data[i, self.Y] = parent_contribution + 0.3 * np.random.randn()
        
        return data

    def bootstrap_samples(self):
        """
        Step 3 in Algorithm 1: Draw N bootstrap samples from D_obs
        """
        # Create bootstrap samples (sample with replacement)
        bootstrap_indices = [np.random.choice(len(self.D_obs), len(self.D_obs), replace=True) 
                            for _ in range(self.n_bootstrap)]
        bootstrap_samples = [self.D_obs[indices] for indices in bootstrap_indices]
        
        # Run doubly robust estimation on each bootstrap sample
        return [self.doubly_robust_estimation(sample) for sample in bootstrap_samples]

    def doubly_robust_estimation(self, data=None):
        """
        Step 4 in Algorithm 1: Run doubly robust to estimate set of parents a_i
        
        This implementation uses correlation as a simple approximation of the 
        doubly robust method mentioned in the paper.
        """
        if data is None:
            data = self.D_obs
            
        # Calculate correlation between each manipulative variable and Y
        corr = np.array([
            np.abs(np.corrcoef(data[:, i], data[:, self.Y])[0, 1]) 
            for i in self.manipulative_vars
        ])
        
        # Add some randomness to parent selection to increase diversity of sets
        corr += 0.1 * np.random.randn(len(corr))
        
        # Return top k correlated manipulative variables as potential parents
        # Vary k between 2 to 6 to increase diversity of parent sets
        k = np.random.randint(2, 7)
        top_k_indices = np.argsort(corr)[-k:]
        
        # Convert back to original node indices
        return np.array([self.manipulative_vars[i] for i in top_k_indices])

    def compute_parent_probabilities(self, bootstrap_results):
        """
        Steps 5-7 in Algorithm 1: Calculate P(g_i) for each potential parent set
        
        Args:
            bootstrap_results: List of potential parent sets from bootstrap samples
        """
        # Store all parent sets from bootstrap
        self.all_parent_sets = bootstrap_results
        
        # Count frequencies of each unique parent set
        parent_set_counter = Counter()
        for parent_set in bootstrap_results:
            # Convert to sorted tuple for hashable key
            parent_set_key = tuple(sorted(parent_set))
            parent_set_counter[parent_set_key] += 1
        
        # Calculate probabilities for each unique parent set
        total_samples = len(bootstrap_results)
        
        # Store unique parent sets with their probabilities
        self.parent_sets_with_probs = []
        for parent_set, count in parent_set_counter.items():
            # Convert tuple back to array for consistency
            parent_set_array = np.array(parent_set)
            probability = count / total_samples
            self.parent_sets_with_probs.append((parent_set_array, probability))
        
        # Sort by probability (highest first)
        self.parent_sets_with_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Update node-level probabilities for reporting purposes
        self.node_probabilities = np.zeros(self.n_nodes)
        
        # Count how many times each node appears across all parent sets
        node_counter = Counter()
        for parent_set, probability in self.parent_sets_with_probs:
            for node in parent_set:
                node_counter[node] += probability
        
        # Convert to node probabilities
        for node, prob in node_counter.items():
            self.node_probabilities[node] = prob
        
        # Normalize node probabilities
        if np.sum(self.node_probabilities) > 0:
            self.node_probabilities /= np.sum(self.node_probabilities)
        
        print(f"\nIdentified {len(self.parent_sets_with_probs)} unique potential parent sets")
        
        if len(self.parent_sets_with_probs) > 0:
            print("Top 3 parent sets with their probabilities:")
            for i, (parent_set, prob) in enumerate(self.parent_sets_with_probs[:3]):
                print(f"  {i+1}. {parent_set} with probability {prob:.4f}")

    def fit_gp_models(self):
        """
        Steps 8-9 in Algorithm 1: Update P(g_i) and fit GP models for each parent set
        """
        # Clear existing models
        self.gp_models = {}
        
        # Combined data for fitting
        combined_data = np.vstack((self.D_obs, self.D_int))
        
        # Fit a separate GP model for each potential parent set
        parent_sets_to_fit = self.parent_sets_with_probs
        
        # If there are too many parent sets, only fit the most probable ones
        max_parent_sets = min(20, len(parent_sets_to_fit))
        if len(parent_sets_to_fit) > max_parent_sets:
            parent_sets_to_fit = parent_sets_to_fit[:max_parent_sets]
        
        # Fit GP models for each parent set
        for idx, (parent_set, probability) in enumerate(parent_sets_to_fit):
            if probability > 0.01:  # Only fit models for reasonably likely parent sets
                try:
                    # Ensure parent_set is not empty
                    if len(parent_set) == 0:
                        continue
                    
                    # Extract features for this parent set
                    X = combined_data[:, parent_set]
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)
                    
                    # Create scalers for this specific model
                    x_scaler = StandardScaler()
                    y_scaler = StandardScaler()
                    
                    # Normalize features and target
                    X_scaled = x_scaler.fit_transform(X)
                    y_scaled = y_scaler.fit_transform(
                        combined_data[:, self.Y].reshape(-1, 1)
                    ).ravel()
                    
                    # Create a fresh kernel for this model
                    kernel = self.create_kernel()
                    
                    # Create GP model
                    gp = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-6,
                        n_restarts_optimizer=10,
                        normalize_y=True,
                        random_state=42,
                        copy_X_train=False
                    )
                    
                    # Fit the GP model
                    gp.fit(X_scaled, y_scaled)
                    
                    # Store the fitted model along with its scaler
                    parent_key = tuple(sorted(parent_set))
                    self.gp_models[parent_key] = {
                        'gp': gp,
                        'X_scaler': x_scaler,
                        'Y_scaler': y_scaler,
                        'probability': probability,
                        'parent_set': parent_set
                    }
                    
                except Exception as e:
                    print(f"Error fitting GP for parent set {parent_set}: {e}")
        
        print(f"Successfully fit {len(self.gp_models)} GP models")

    def compute_acquisition_function(self, X_candidates):
        """
        Steps 10-11 in Algorithm 1: Compute acquisition function
        
        This implements the expectation in equations (10-12) from the paper, averaging
        over uncertainty in the causal structure.
        
        Args:
            X_candidates: Candidate intervention points to evaluate
        
        Returns:
            Acquisition values for each candidate point
            Mean values for each candidate
            Standard deviation values for each candidate
        """
        if len(self.gp_models) == 0:
            # If no models, return random values
            random_values = np.random.randn(len(X_candidates))
            return random_values, np.zeros(len(X_candidates)), np.ones(len(X_candidates))
        
        # Initialize acquisition values for each candidate
        n_candidates = X_candidates.shape[0]
        acquisition_values = np.zeros(n_candidates)
        mean_values = np.zeros(n_candidates)
        std_values = np.zeros(n_candidates)
        
        # Compute acquisition value for each candidate by taking expectations over parent sets
        for candidate_idx in range(n_candidates):
            expected_mean = 0
            expected_variance = 0
            squared_means = 0  # For calculating V_P(g)[E[Y|ξ,C,g]]
            total_probability = 0
            
            # Compute expectations over all parent sets as in equations (10-12)
            for parent_key, model_data in self.gp_models.items():
                gp = model_data['gp']
                parent_set = model_data['parent_set']
                prob = model_data['probability']
                x_scaler = model_data['X_scaler']
                y_scaler = model_data['Y_scaler']
                
                # Extract features for this candidate using this parent set
                x = X_candidates[candidate_idx, parent_set].reshape(1, -1)
                if x.shape[1] == 0:  # Skip if no parents in this set
                    continue
                
                # Scale the input
                x_scaled = x_scaler.transform(x)
                
                try:
                    # Get prediction from this model
                    mean, std = gp.predict(x_scaled, return_std=True)
                    
                    # Transform back to original scale
                    mean = y_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()[0]
                    std = std[0] * y_scaler.scale_[0]  # Adjust std for scaling
                    variance = std**2
                    
                    # Accumulate weighted predictions for E_P(g)[E[Y|ξ,C,g]]
                    expected_mean += prob * mean
                    
                    # For later calculating V_P(g)[E[Y|ξ,C,g]]
                    squared_means += prob * (mean ** 2)
                    
                    # E_P(g)[V[Y|ξ,C,g]] component
                    expected_variance += prob * variance
                    
                    total_probability += prob
                    
                except Exception as e:
                    print(f"Error in prediction for parent set {parent_set}: {e}")
            
            if total_probability > 0:
                # Normalize by total probability used
                expected_mean /= total_probability
                squared_means /= total_probability
                expected_variance /= total_probability
                
                # Calculate V_P(g)[E[Y|ξ,C,g]] component = E[mean²] - E[mean]²
                variance_of_means = squared_means - expected_mean**2
                
                # Total variance = V_P(g)[E[Y|ξ,C,g]] + E_P(g)[V[Y|ξ,C,g]]
                total_variance = variance_of_means + expected_variance
                total_std = np.sqrt(total_variance)
                
                # Store for reporting
                mean_values[candidate_idx] = expected_mean
                std_values[candidate_idx] = total_std
                
                # Compute Lower Confidence Bound (LCB) acquisition function
                # Can be adjusted based on exploration vs. exploitation trade-off
                beta = 2.0  # Controls exploration-exploitation trade-off
                acquisition_values[candidate_idx] = expected_mean - beta * total_std
            else:
                # Fallback to a random value if no predictions were made
                acquisition_values[candidate_idx] = np.random.randn()
                mean_values[candidate_idx] = 0
                std_values[candidate_idx] = 1
        
        return acquisition_values, mean_values, std_values

    def obtain_optimal_intervention(self):
        """
        Step 12 in Algorithm 1: Obtain optimal set value ξ_t
        
        Returns:
            Dictionary mapping parent nodes to their optimal intervention values
        """
        # Generate candidate interventions (grid of possible interventions)
        # For simplicity, use points from observational data
        X_candidates = self.D_obs
        
        # Compute acquisition function values for all candidates
        acq_values, mean_values, std_values = self.compute_acquisition_function(X_candidates)
        
        # Find the candidate with the minimum acquisition value
        optimal_idx = np.argmin(acq_values)
        optimal_candidate = X_candidates[optimal_idx]
        
        # Get the actual top parent nodes by probability
        top_nodes = np.argsort(self.node_probabilities)[-5:]
        top_nodes = [node for node in top_nodes if self.node_probabilities[node] > 0.01]
        
        # Create intervention dictionary focusing on likely parents
        intervention_dict = {}
        
        # Add intervention values for all likely parent nodes
        for node in top_nodes:
            if node != self.Y:  # Don't intervene on Y
                intervention_dict[node] = optimal_candidate[node]
        
        # Add some other nodes to intervene on
        for node in self.manipulative_vars:
            if node not in intervention_dict and random.random() < 0.3:  # 30% chance to include other nodes
                intervention_dict[node] = optimal_candidate[node]
        
        return intervention_dict, acq_values[optimal_idx], mean_values[optimal_idx], std_values[optimal_idx]

    def intervene_and_obtain_data(self, intervention_dict):
        """
        Step 13 in Algorithm 1: Intervene and obtain D_int
        
        Args:
            intervention_dict: Dictionary mapping nodes to intervention values
            
        Returns:
            New data point resulting from the intervention
        """
        # Create a new data point starting with random values
        new_intervention = np.random.randn(1, self.n_nodes)
        
        # Set the intervention values
        for node, value in intervention_dict.items():
            new_intervention[0, node] = value
        
        # Calculate the resulting Y based on its true parents
        y_value = 0
        for parent in self.true_parents:
            if parent in intervention_dict:  # If parent was intervened on
                weight = 0.5 + random.random()
                y_value += weight * intervention_dict[parent]
            else:  # If parent wasn't intervened on, use random value
                weight = 0.5 + random.random()
                y_value += weight * new_intervention[0, parent]
        
        # Add noise to Y
        new_intervention[0, self.Y] = y_value + 0.3 * np.random.randn()
        
        return new_intervention

    def update_interventional_dataset(self, new_data):
        """
        Step 14 in Algorithm 1: Update D_int
        
        Args:
            new_data: New data point to add to the interventional dataset
        """
        self.D_int = np.vstack((self.D_int, new_data))
        
        # Track best intervention found so far
        current_value = new_data[0, self.Y]
        if current_value < self.best_value:
            self.best_value = current_value
            self.best_intervention = {
                node: new_data[0, node] for node in self.manipulative_vars
            }

    def run_optimization(self):
        """
        Main method implementing Algorithm 1 (CBO-U)
        """
        print("Starting Causal Bayesian Optimization with Unknown Graph (CBO-U)")
        
        # Step 1: Start with initial D_obs and P_init (done in __init__)
        print(f"Initialized with {len(self.D_obs)} observational data points")
        print(f"Target variable Y: Node {self.Y}")
        print(f"True parents of Y: {self.true_parents}")
        
        # Visualize the true graph before starting
        # self.visualize_true_graph()
        
        # Steps 2-7: Bootstrap sampling and parent probability estimation
        print("\nEstimating parent sets using bootstrap sampling...")
        bootstrap_results = self.bootstrap_samples()
        self.compute_parent_probabilities(bootstrap_results)
        
        # Store initial parent probabilities for visualization
        self.parent_probability_history.append(self.node_probabilities.copy())
        
        # Steps 8-9: Fit initial GP models
        print("\nFitting GP models for potential parent sets...")
        self.fit_gp_models()
        
        # If no GP models could be fit, use simple correlation for interventions
        if len(self.gp_models) == 0:
            print("\nWarning: No GP models could be fit. Using correlation-based approach instead.")
            # Use correlation-based method for finding parents
            corr = np.array([
                np.abs(np.corrcoef(self.D_obs[:, i], self.D_obs[:, self.Y])[0, 1]) 
                for i in self.manipulative_vars
            ])
            
            # Get top correlated nodes
            top_correlated = [self.manipulative_vars[i] for i in np.argsort(corr)[-5:]]
            for node in top_correlated:
                self.node_probabilities[node] = 0.2  # Assign equal probabilities
        
        # Steps 10-17: Main optimization loop for T iterations
        print(f"\nStarting {self.n_interventions} intervention iterations:")
        for t in range(self.n_interventions):
            print(f"\nIntervention {t+1}/{self.n_interventions}")
            
            # Steps 10-12: Compute acquisition function and find optimal intervention
            intervention_dict, acq_value, mean_value, std_value = self.obtain_optimal_intervention()
            
            # Store intervention for history
            self.intervention_history.append(intervention_dict.copy())
            
            # Print the nodes being intervened on
            print(f"Intervening on {len(intervention_dict)} nodes with acquisition value: {acq_value:.4f}")
            print(f"Expected mean: {mean_value:.4f}, expected std: {std_value:.4f}")
            print(f"Intervening on nodes: {list(intervention_dict.keys())}")
            
            # Step 13: Perform intervention
            new_data = self.intervene_and_obtain_data(intervention_dict)
            print(f"Resulting Y value: {new_data[0, self.Y]:.4f}")
            
            # Store Y value for history
            self.y_value_history.append(new_data[0, self.Y])
            
            # Step 14: Update interventional dataset
            self.update_interventional_dataset(new_data)
            
            # Steps 15-16: Update parent probabilities and refit GP models
            if (t + 1) % 5 == 0 or t == self.n_interventions - 1:
                print("Updating parent set probabilities and refitting GP models...")
                bootstrap_results = self.bootstrap_samples()
                self.compute_parent_probabilities(bootstrap_results)
                self.fit_gp_models()
                
                # Store updated parent probabilities for visualization
                self.parent_probability_history.append(self.node_probabilities.copy())
                
                # Print current node probabilities for top nodes
                print("\nCurrent node probabilities for likely parents:")
                top_indices = np.argsort(self.node_probabilities)[-5:]
                for idx in top_indices[::-1]:
                    if self.node_probabilities[idx] > 0.01:
                        in_true_parents = "✓" if idx in self.true_parents else "✗"
                        print(f"  Node {idx}: {self.node_probabilities[idx]:.4f} {in_true_parents}")
        
        # Step 18: Return best intervention found
        print("\nOptimization complete!")
        print(f"Best intervention found (Y = {self.best_value:.4f}):")
        
        # Identify which nodes are true parents in the best intervention
        for node, value in self.best_intervention.items():
            if node in np.argsort(self.node_probabilities)[-5:] and self.node_probabilities[node] > 0.01:
                is_true_parent = "True parent" if node in self.true_parents else "Not a true parent"
                print(f"  Node {node} = {value:.4f} (prob = {self.node_probabilities[node]:.4f}) - {is_true_parent}")
        
        # Evaluate the effectiveness of causal structure learning
        true_positive = sum(1 for node in self.true_parents if self.node_probabilities[node] > 0.05)
        false_negative = len(self.true_parents) - true_positive
        
        # Get nodes identified as parents but aren't true parents
        identified_parents = [node for node in range(self.n_nodes) 
                            if self.node_probabilities[node] > 0.05 and node != self.Y]
        false_positive = sum(1 for node in identified_parents if node not in self.true_parents)
        
        print("\nCausal structure learning performance:")
        print(f"True positive (correctly identified parents): {true_positive}/{len(self.true_parents)}")
        print(f"False negative (missed parents): {false_negative}")
        print(f"False positive (incorrectly identified as parents): {false_positive}")
        
        # Run comprehensive analysis and visualization
        self.analyze_and_visualize()
        
        return self.best_intervention, self.best_value

    def visualize_true_graph(self):
        """
        Visualize the true causal graph with target node highlighted
        """
        plt.figure(figsize=(10, 8))
        
        # Specify positions for better visualization
        pos = nx.spring_layout(self.true_G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.true_G, pos, node_size=300, node_color='lightblue')
        
        # Highlight the target node Y
        nx.draw_networkx_nodes(self.true_G, pos, nodelist=[self.Y], node_size=500, 
                              node_color='red', node_shape='s')
        
        # Highlight the true parents of Y
        nx.draw_networkx_nodes(self.true_G, pos, nodelist=self.true_parents, node_size=400, 
                              node_color='green')
        
        # Draw edges
        nx.draw_networkx_edges(self.true_G, pos, width=1.0, alpha=0.7)
        
        # Highlight edges from parents to Y
        parent_edges = [(p, self.Y) for p in self.true_parents]
        nx.draw_networkx_edges(self.true_G, pos, edgelist=parent_edges, width=3.0, 
                              edge_color='green', alpha=0.8)
        
        # Add labels
        nx.draw_networkx_labels(self.true_G, pos)
        
        plt.title(f'True Causal Graph (Target: Node {self.Y}, {len(self.true_parents)} parents)')
        plt.axis('off')
        
        # Add legend
        plt.plot([], [], 'rs', markersize=10, label='Target Node Y')
        plt.plot([], [], 'go', markersize=10, label='True Parents')
        plt.plot([], [], 'o', color='lightblue', markersize=10, label='Other Nodes')
        plt.legend(loc='upper right')
        
        plt.savefig("true_causal_graph.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_learned_structure(self):
        """
        Visualize the learned causal structure based on node probabilities
        """
        plt.figure(figsize=(10, 8))
        
        # Create a graph representation of the learned structure
        learned_G = nx.DiGraph()
        learned_G.add_nodes_from(range(self.n_nodes))
        
        # Add edges based on node probabilities
        threshold = 0.05  # Probability threshold for including an edge
        for node in range(self.n_nodes):
            if node != self.Y and self.node_probabilities[node] > threshold:
                learned_G.add_edge(node, self.Y)
        
        # Specify positions for better visualization
        pos = nx.spring_layout(learned_G, seed=42)
        
        # Create a colormap for edges based on probabilities
        edge_colors = [self.node_probabilities[u] for u, v in learned_G.edges()]
        
        # Draw nodes
        nx.draw_networkx_nodes(learned_G, pos, node_size=300, node_color='lightblue')
        
        # Highlight the target node Y
        nx.draw_networkx_nodes(learned_G, pos, nodelist=[self.Y], node_size=500, 
                              node_color='red', node_shape='s')
        
        # Draw edges if there are any
        if edge_colors:
            edges = nx.draw_networkx_edges(learned_G, pos, width=2.0, alpha=0.7,
                                        edge_color=edge_colors, edge_cmap=plt.cm.Blues)
            
            # Draw edge labels with probabilities
            edge_labels = {(u, v): f"{self.node_probabilities[u]:.2f}" for u, v in learned_G.edges()}
            nx.draw_networkx_edge_labels(learned_G, pos, edge_labels=edge_labels, font_size=8)
            
            # Add colorbar for edge probabilities only if there are edges
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=max(edge_colors)))
            sm.set_array([])
            plt.colorbar(sm, label='Edge Probability', ax=plt.gca())
        else:
            print("No edges in learned structure (no nodes above threshold)")
        
        # Draw node labels
        nx.draw_networkx_labels(learned_G, pos)
        
        plt.title(f'Learned Causal Structure (Target: Node {self.Y})')
        plt.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Target Node Y'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Potential Parent Nodes'),
        ]
        
        # Add true parents to legend
        for p in self.true_parents:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                         markeredgecolor='green', markeredgewidth=2, markersize=10, 
                         label=f'True Parent: Node {p}')
            )
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig("learned_causal_structure.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_parent_probabilities(self):
        """
        Visualize the probabilities of each node being a parent of Y
        """
        plt.figure(figsize=(12, 6))
        
        # Get indices sorted by probability
        sorted_indices = np.argsort(self.node_probabilities)
        
        # Get node numbers and probabilities
        nodes = [i for i in sorted_indices if i != self.Y and self.node_probabilities[i] > 0.01]
        probs = [self.node_probabilities[i] for i in nodes]
        
        # Create bar colors - green for true parents, blue for others
        colors = ['green' if node in self.true_parents else 'skyblue' for node in nodes]
        
        # Create the bar chart
        bars = plt.bar(range(len(nodes)), probs, color=colors)
        
        plt.title('Node Parent Probabilities for Target Y')
        plt.xlabel('Node Index')
        plt.ylabel('Probability')
        plt.xticks(range(len(nodes)), nodes)
        plt.ylim(0, max(probs) * 1.1)
        
        # Add horizontal line for threshold
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Threshold (0.05)')
        
        # Add a legend
        plt.legend(['Threshold (0.05)', 'True Parent', 'Not a Parent'])
        
        plt.tight_layout()
        plt.savefig("parent_probabilities.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_probability_evolution(self):
        """
        Visualize how parent probabilities evolved over interventions
        """
        if not self.parent_probability_history:
            print("No probability history available.")
            return
        
        # Convert history to array
        history_array = np.array(self.parent_probability_history)
        
        # Get only true parents and top non-parents
        top_nodes = list(self.true_parents)
        
        # Add some top non-parent nodes
        non_parents = [i for i in range(self.n_nodes) if i != self.Y and i not in self.true_parents]
        if non_parents:  # Check if there are any non-parent nodes
            final_probs = self.node_probabilities[non_parents]
            # Check if there are at least 3 non-parent nodes
            n_to_include = min(3, len(non_parents))
            top_indices = np.argsort(final_probs)[-n_to_include:]
            top_non_parents = [non_parents[i] for i in top_indices]
        else:
            top_non_parents = []
        
        # Combine nodes to plot
        plot_nodes = top_nodes + top_non_parents
        
        if not plot_nodes:
            print("No nodes to plot in probability evolution.")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot each node's probability over time
        for node in plot_nodes:
            if node < self.n_nodes:  # Ensure node is valid
                label = f"Node {node} (True Parent)" if node in self.true_parents else f"Node {node}"
                linestyle = '-' if node in self.true_parents else '--'
                plt.plot(range(len(history_array)), history_array[:, node], 
                        label=label, marker='o', linestyle=linestyle, markersize=4)
        
        plt.title('Evolution of Parent Probabilities During Optimization')
        plt.xlabel('Intervention Round')
        plt.ylabel('Parent Probability')
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("probability_evolution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_intervention_results(self):
        """
        Visualize the Y values obtained from interventions
        """
        if not self.y_value_history:
            print("No intervention history available.")
            return
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(range(len(self.y_value_history)), self.y_value_history, 
                marker='o', linestyle='-', color='blue')
        
        # Mark the best Y value
        best_idx = np.argmin(self.y_value_history)
        plt.plot(best_idx, self.y_value_history[best_idx], 'r*', markersize=15, 
                label=f'Best Value: {self.y_value_history[best_idx]:.4f}')
        
        plt.title('Y Values from Interventions')
        plt.xlabel('Intervention Number')
        plt.ylabel('Y Value (lower is better)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("intervention_results.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_confusion_matrix(self):
        """
        Visualize the confusion matrix for parent identification
        """
        # Get predictions based on probability threshold
        threshold = 0.05
        predicted_parents = [i for i in range(self.n_nodes) 
                           if i != self.Y and self.node_probabilities[i] > threshold]
        
        # Get confusion matrix values
        tp = sum(1 for node in predicted_parents if node in self.true_parents)
        fp = len(predicted_parents) - tp
        fn = len(self.true_parents) - tp
        tn = self.n_nodes - 1 - tp - fp - fn  # All nodes except Y, minus tp, fp, fn
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create confusion matrix as a 2x2 array
        cm = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Not Parent', 'Parent'],
                   yticklabels=['Not Parent', 'Parent'])
        
        plt.title('Confusion Matrix for Parent Identification')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        
        # Add metrics text
        plt.figtext(0.6, 0.25, f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}",
                  bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_side_by_side_comparison(self):
        """
        Visualize the true causal graph and learned structure side by side
        """
        plt.figure(figsize=(18, 8))
        
        # First subplot for true graph
        plt.subplot(1, 2, 1)
        
        # Specify positions for better visualization
        true_pos = nx.spring_layout(self.true_G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.true_G, true_pos, node_size=300, node_color='lightblue')
        
        # Highlight the target node Y
        nx.draw_networkx_nodes(self.true_G, true_pos, nodelist=[self.Y], node_size=500, 
                              node_color='red', node_shape='s')
        
        # Highlight the true parents of Y
        nx.draw_networkx_nodes(self.true_G, true_pos, nodelist=self.true_parents, node_size=400, 
                              node_color='green')
        
        # Draw edges
        nx.draw_networkx_edges(self.true_G, true_pos, width=1.0, alpha=0.7)
        
        # Highlight edges from parents to Y
        parent_edges = [(p, self.Y) for p in self.true_parents]
        nx.draw_networkx_edges(self.true_G, true_pos, edgelist=parent_edges, width=3.0, 
                              edge_color='green', alpha=0.8)
        
        # Add labels
        nx.draw_networkx_labels(self.true_G, true_pos)
        
        plt.title(f'True Causal Graph (Target: Node {self.Y})')
        plt.axis('off')
        
        # Second subplot for learned structure
        plt.subplot(1, 2, 2)
        
        # Create a graph representation of the learned structure
        learned_G = nx.DiGraph()
        learned_G.add_nodes_from(range(self.n_nodes))
        
        # Add edges based on node probabilities
        threshold = 0.05  # Probability threshold for including an edge
        for node in range(self.n_nodes):
            if node != self.Y and self.node_probabilities[node] > threshold:
                learned_G.add_edge(node, self.Y)
        
        # Use the same positions as true graph for better comparison
        # This helps visualize corresponding nodes in the same positions
        learned_pos = true_pos
        
        # Draw nodes
        nx.draw_networkx_nodes(learned_G, learned_pos, node_size=300, node_color='lightblue')
        
        # Highlight the target node Y
        nx.draw_networkx_nodes(learned_G, learned_pos, nodelist=[self.Y], node_size=500, 
                              node_color='red', node_shape='s')
        
        # Draw edges if there are any
        edge_colors = [self.node_probabilities[u] for u, v in learned_G.edges()]
        if edge_colors:
            edges = nx.draw_networkx_edges(learned_G, learned_pos, width=2.0, alpha=0.7,
                                       edge_color=edge_colors, edge_cmap=plt.cm.Blues)
            
            # Add edge labels with probabilities for learned structure
            edge_labels = {(u, v): f"{self.node_probabilities[u]:.2f}" for u, v in learned_G.edges()}
            nx.draw_networkx_edge_labels(learned_G, learned_pos, edge_labels=edge_labels, font_size=8)
        else:
            print("No edges in learned structure (no nodes above threshold)")
        
        # Add node labels
        nx.draw_networkx_labels(learned_G, learned_pos)
        
        plt.title(f'Learned Causal Structure (Target: Node {self.Y})')
        plt.axis('off')
        
        # Add a shared legend
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Target Node Y'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Other Nodes'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Parent'),
        ]
        
        if edge_colors:
            # Add legend for edge probability if there are edges
            plt.figtext(0.5, 0.01, "Edge color intensity indicates probability of being a parent", 
                      ha='center', fontsize=10)
        
        plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), 
                     ncol=3, bbox_transform=plt.gcf().transFigure)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig("causal_graph_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print validation statistics
        self.print_validation_statistics()
    
    def print_validation_statistics(self):
        """
        Print detailed validation statistics for the causal structure learning
        """
        print("\n=== Validation Statistics ===")
        
        # Get predicted parents based on threshold
        threshold = 0.05
        predicted_parents = [i for i in range(self.n_nodes) 
                           if i != self.Y and self.node_probabilities[i] > threshold]
        
        # Calculate performance metrics
        tp = sum(1 for node in predicted_parents if node in self.true_parents)
        fp = len(predicted_parents) - tp
        fn = len(self.true_parents) - tp
        tn = self.n_nodes - 1 - tp - fp - fn  # All nodes except Y, minus tp, fp, fn
        
        # Calculate additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (self.n_nodes - 1) if (self.n_nodes > 1) else 0
        
        # Calculate the structural Hamming distance (SHD)
        # SHD counts the number of edge operations (addition, deletion, reversal) to transform one graph to another
        # In our case, it's the number of false positives + false negatives (since we're only looking at direct parents)
        shd = fp + fn
        
        # Print statistics
        print(f"Number of true parents: {len(self.true_parents)}")
        print(f"Number of predicted parents: {len(predicted_parents)}")
        print("\nDetailed True Parents:")
        for parent in self.true_parents:
            is_discovered = parent in predicted_parents
            status = "✓ Discovered" if is_discovered else "✗ Missed"
            prob = self.node_probabilities[parent]
            print(f"  Node {parent}: {status} (probability: {prob:.4f})")
        
        print("\nFalse Positives:")
        for parent in predicted_parents:
            if parent not in self.true_parents:
                prob = self.node_probabilities[parent]
                print(f"  Node {parent}: Incorrectly identified as parent (probability: {prob:.4f})")
        
        print("\nConfusion Matrix:")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  True Negatives: {tn}")
        print(f"  False Negatives: {fn}")
        
        print("\nPerformance Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall/Sensitivity: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Structural Hamming Distance (SHD): {shd}")
        
        # Calculate node-specific probabilities
        print("\nNode Probability Analysis:")
        print("  Node  |  True Parent  |  Probability  |  Status")
        print("  --------------------------------------------------")
        for node in range(self.n_nodes):
            if node != self.Y:
                is_true = node in self.true_parents
                prob = self.node_probabilities[node]
                predicted = prob > threshold
                status = "True Positive" if is_true and predicted else \
                         "False Negative" if is_true and not predicted else \
                         "False Positive" if not is_true and predicted else "True Negative"
                true_str = "Yes" if is_true else "No"
                print(f"  {node:4d}  |  {true_str:^11s}  |  {prob:^11.4f}  |  {status}")
        
        # Return the calculated metrics for potential further use
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'shd': shd
        }
    
    def analyze_and_visualize(self):
        """
        Comprehensive analysis and visualization of results
        """
        print("\n=== Causal Structure Analysis ===")
        
        # # Wrap all visualizations in try/except to prevent errors from stopping the analysis
        # try:
        #     # 1. Visualize the true causal graph
        #     self.visualize_true_graph()
        # except Exception as e:
        #     print(f"Error visualizing true graph: {e}")
        
        # try:
        #     # 2. Visualize the learned causal structure
        #     self.visualize_learned_structure()
        # except Exception as e:
        #     print(f"Error visualizing learned structure: {e}")
        
        try:
            # 3. Side-by-side comparison of true and learned structures
            self.visualize_side_by_side_comparison()
        except Exception as e:
            print(f"Error creating side-by-side comparison: {e}")
        
        try:
            # 4. Visualize parent probabilities
            self.visualize_parent_probabilities()
        except Exception as e:
            print(f"Error visualizing parent probabilities: {e}")
        
        try:
            # 5. Visualize probability evolution if available
            if self.parent_probability_history:
                self.visualize_probability_evolution()
        except Exception as e:
            print(f"Error visualizing probability evolution: {e}")
        
        try:
            # 6. Visualize intervention results if available
            if self.y_value_history:
                self.visualize_intervention_results()
        except Exception as e:
            print(f"Error visualizing intervention results: {e}")
        
        try:
            # 7. Visualize confusion matrix
            self.visualize_confusion_matrix()
        except Exception as e:
            print(f"Error visualizing confusion matrix: {e}")
        
        # 8. Print validation statistics (even if visualizations fail)
        self.print_validation_statistics()
            
        print("\nAnalysis complete. Check the saved PNG files for visualizations.")

# Run optimization
if __name__ == "__main__":
    # Parameters for the experiment
    n_nodes = 50           # Number of nodes in the graph
    p_edge = 0.15           # Probability of edge in Erdos-Renyi graph
    n_bootstrap = 50       # Number of bootstrap samples for parent estimation
    n_interventions = 20   # Number of interventions to perform
    use_seed = False       # Whether to use a fixed random seed
    
    # Set random seed if specified
    if use_seed:
        print("Using fixed random seed: 1")
        np.random.seed(1)
        random.seed(1)
    
    print("=== Causal Bayesian Optimization Experiment ===")
    print(f"Number of nodes: {n_nodes}")
    print(f"Edge probability: {p_edge}")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Interventions: {n_interventions}")
    print(f"Fixed random seed: {use_seed}")
    print("="*50)
    
    # Create and run the optimization
    cbo = CausalBayesianOptimization(
        n_nodes=n_nodes, 
        p_edge=p_edge, 
        n_bootstrap=n_bootstrap, 
        n_interventions=n_interventions
    )
    
    best_intervention, best_value = cbo.run_optimization()
    
    print("\n=== Experiment Complete ===")
    # print(f"Best Y value achieved: {best_value:.4f}")
    print(f"Results saved as PNG files in the current directory.")