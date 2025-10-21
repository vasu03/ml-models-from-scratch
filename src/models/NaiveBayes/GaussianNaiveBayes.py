"""
- File:     src/model/GaussianNaiveBayes.py
- Desc:     Implementation of Gaussian Naive Bayes Classifier from scratch using minimal dependencies. 
- License:  MIT License
- Author:   Vasu Makadia
"""

# Import core libraries
import numpy as np

class GaussianNaiveBayes:
    """
    Brief:
        Implementation of a Gaussian Naive Bayes (GNB) classifier from scratch.
    Description:
        This model assumes that features follow a Gaussian (normal) distribution 
        and are conditionally independent given the class label.
    Attributes:
        classes_ (np.ndarray):   Unique class labels in the training set.
        means_ (dict):           Mean of each feature for each class.
        variances_ (dict):       Variance of each feature for each class.
        priors_ (dict):          Prior probability of each class.
    """

    def __init__ (self):
        """
        Brief:
            Initialize the Gaussian Naive Bayes model attributes.
        """
        self.classes_   =   None
        self.means_     =   None
        self.variances_ =   None
        self.priors_    =   None
    
    def fit (self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Brief:
            Public method to train the model for learning parameters for each class through the given data
        Parameters:
            X (np.ndarray):     Feature Matrix (independent variables) of shape (n_samples, n_features)
            y (np.ndarray):     Target Labels (class or dependent variables) of shape (n_samples, )
        Returns:
            None
        """
        # Convert inputs to np arrays type
        X = np.array(X)
        y = np.array(y)

        # Identify unique class labels in target vector
        self.classes_ = np.unique(y)

        # Initialize dictionary to store class-wise stats
        self.means_ = {}
        self.variances_ = {}
        self.priors_ = {}

        # Calculate class wise mean, variance, prior prob
        for cls in self.classes_:
            # extract subset of samples belonging to class 'cls'
            X_cls = X[y == cls]
            # Compute mean and variance for class 'cls'
            self.means_[cls] = X_cls.mean(axis=0)
            self.variances_[cls] = X_cls.var(axis=0) + 1e-9   # added epsilon to avoid div-by-zero error
            # compute prior probablity for class 'cls'
            # P (class) = samples_in_class / total_samples
            self.priors_[cls] = X_cls.shape[0] / X.shape[0]

    def _pdf (self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """
        Brief:
            Private Method to calculate the Gaussian Probability Density Function using formula
        Formula:
            P(x|class) = (1 / sqrt(2πσ²)) * exp(- (x - μ)² / (2σ²))
        Parameters:
            x (np.ndarray):     Feature matrix of shape (n_samples, n_features)
            mean (np.ndarray):  Array of Means of given features
            var (np.ndarray):   Array of Variances of given feature
        Returns:
            pdf (np.ndarray):   Probability density function value for given feature
        """
        # Ensure no zero or negative variance
        var = np.clip(var, 1e-9, None)

        # Compute numerator and denominator separately
        numerator = np.exp(-0.5 * ((x - mean) ** 2) / var)
        denominator = np.sqrt(2 * np.pi * var)
        pdf = numerator / denominator

        # Prevent log(0) or underflow when taking log later
        return np.clip(pdf, 1e-12, None)

    def _log_likelihood (self, X: np.ndarray) -> np.ndarray:
        """
        Brief:
            Private method to calculate the log-posterior probabilities for all classes,
            Using log-space to prevent underflow of small probabilities using formula;]
        Formula:
            log(P(class|X)) ∝ log(P(class)) + Σ log(P(x_i|class))
        Parameters:
            X (np.ndarray) :    Feature matrix of shape (n_samples, n_features)
        Returns:
            log_probs (np.ndarray): A matrix of log-posterior probabilities of shape (n_samples, n_features)
        """
        # create a matrix to store log-posterior probabilities
        log_probs = np.zeros((X.shape[0], len(self.classes_)))

        for idx, cls in enumerate (self.classes_):
            # extract parameters for this class
            mean = self.means_[cls]
            var = self.variances_[cls]
            prior = np.log(self.priors_[cls])
            pdf_vals = self._pdf(X, mean, var)
            log_likelihood = np.sum(np.log(pdf_vals), axis=1)
            log_probs[:, idx] = prior + log_likelihood
        
        return log_probs
    
    def predict (self, X: np.ndarray) -> np.ndarray:
        """
        Brief:
            Public method to make predictions (or calculate class variables) for the given test features by;
            1) Computing log-posterior probabilities for all classes
            2) Choose the class with maximum posterior for each sample value 
        Parameters:
            X (np.ndarray): Test Feature Matrix of shape (n_samples, n_features)
        Returns:
            y (np.ndarray): Predicted class variables for given X feature matrix of shape (n_samples, )
        """
        X = np.array(X)
        log_probs = self._log_likelihood(X)
        return self.classes_[np.argmax(log_probs, axis=1)]
    
    def predict_probability (self, X: np.ndarray):
        """
        Brief:
            Public method to predict the posterior probabilities for each class of given samples
        Parameters:
            X (np.ndarray): Test Feature Matrix of shape (n_samples, n_features)
        Retruns:
            probs (np.ndarray): Predicted class probabilities with shape (n_samples, n_classes), where each row sums to 1.
        """
        X = np.array(X)
        log_probs = self._log_likelihood(X)
        # Convert from log space back to probability space
        probs = np.exp(log_probs)
        # Normalize each row to make valid probabilities (sum = 1)
        return (probs / probs.sum(axis=1, keepdims=True))