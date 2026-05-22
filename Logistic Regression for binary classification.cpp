#include <vector>
#include <cmath> //Need to use to calcualte e^-z in the sigmoid function
#include <iostream>

namespace sklearn_cpp {
namespace linear_model {

class LogisticRegression {
private:
    std::vector<double> weights; // Stores our learnable weights (w1, w2, ..., wn)
    double bias;                 // Stores the bias term (b)
    
    // Hyperparameters
    double learning_rate;        // How big of a step we take during gradient descent (alpha)
    int max_iter;                // How many times we loop over the data to learn
    double lambda;               // Regularisation to prevent overfitting

   
    // Maps any real number (z) into a probability between 0 and 1
    double sigmoid(double z) const {
        return 1.0 / (1.0 + std::exp(-z));
    }

public:
    // Constructor: Sets up our model with default parameters (just incase user forgets to input parameters) and list initialiser
    LogisticRegression(double lr = 0.01, int iters = 1000, double reg_lambda = 0.01)
        : learning_rate(lr), max_iter(iters), lambda(reg_lambda), bias(0.0) {}

    // The core training function where the math happens. X is the training data, y is the true target (0 or 1) for each sample
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        if (X.empty() || y.empty()) return; // Will exit if either x or y is empty. Used as a safety check.

        int num_samples = X.size(); //The number of data rows we have
        int num_features = X[0].size(); //The number of information columns per row

        //Resizes weight vector to exactly hwo many features we have and sets every single starting weight to zero. Also reset baseline to zero
        weights.assign(num_features, 0.0); 
        bias = 0.0;

        // Gradient Descent Training Loop 
        for (int k = 0; k < max_iter; ++k) {
            
            
            
            // Temporary variables to hold the accumulated gradients
            std::vector<double> dw(num_features, 0.0); //dw; derivative of weights
            double db = 0.0; //db: derivative of bias
            // Step 1: Forward Pass & Gradient Accumulation
            for (int i = 0; i < num_samples; ++i) {
                // Calculate the combination: z = (w * x) + b
                double z = bias;
                for (int j = 0; j < num_features; ++j) {
                    z += weights[j] * X[i][j];
                }

                // Pass z through sigmoid to get our predicted probability 
                double y_pred = sigmoid(z); 
                
                // Calculate the error (Prediction - Actual) 
                double error = y_pred - y[i];

                // Accumulate the gradients for weights and bias based on the error
                for (int j = 0; j < num_features; ++j) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            // Step 2: Average the gradients and apply L2 Regularisation 
            for (int j = 0; j < num_features; ++j) {
                // Average the gradient and add the derivative of the L2 penalty: 2 * lambda * weight
                dw[j] = (dw[j] / num_samples) + (2.0 * lambda * weights[j]);
                
                // Step 3: Update the actual weights (move in the opposite direction of the gradient)
                weights[j] -= learning_rate * dw[j];
            }

            // Update the bias in the exact same way
            db = (db / num_samples) + (2.0 * lambda * bias);
            bias -= learning_rate * db;
        }
    }

    // The prediction function for new, unseen data 
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const { //stores our final outputs 0's or 1's.
        std::vector<int> predictions;
        //Loops through every sample provided and just like in training calculates the raw score z using learned weights and bias, then
        // converts to probability between 0 and 1 
        for (const auto& sample : X) {
            // First, calculate z for the new sample
            double z = bias;
            for (size_t j = 0; j < weights.size(); ++j) {
                z += weights[j] * sample[j];
            }
            
            // Get the probability 
            double probability = sigmoid(z);
            
            // Apply our 0.5 threshold to make a binary decision (0 or 1) 
            if (probability >= 0.5) {
                predictions.push_back(1);
            } else {
                predictions.push_back(0);
            }
        }
        
        return predictions;
    }
};

} // namespace linear_model
} // namespace sklearn_cpp