#pragma once
#include <vector>
#include <cmath> // Need to use to calculate e^-z in the sigmoid function
#include <algorithm> // For std::max

namespace sklearn_cpp {
namespace linear_model {

class LogisticRegression {
private:
    std::vector<double> weights_binary; // Stores our learnable weights (w1, w2, ..., wn)
    std::vector<std::vector<double>> weights_multi; // For multi-class 
    double bias_binary = 0.0;                 // Stores the bias term (b)
    std::vector<double> bias_multi;
    
    // Hyperparameters
    double learning_rate;        // How big of a step we take during gradient descent (alpha)
    int max_iter;                // How many times we loop over the data to learn
    double lambda;               // Regularisation to prevent overfitting

    // Mode = 0 then Binary, if mode = 1 then multi-class
    int mode = 0;
   
    // Maps any real number (z) into a probability between 0 and 1
    double sigmoid(double z) const {
        return 1.0 / (1.0 + std::exp(-z));
    }

    std::vector<int> one_hot_encode(int answer_int) {
        std::vector<int> answer_vector(weights_multi.size(), 0); // vector to store predicted answers
        if (answer_int >= 0 && answer_int < weights_multi.size()) { // check if the predicted answer is within the valid range
            answer_vector[answer_int] = 1; // set the predicted answer to 1
        }
        return answer_vector; // return the vector of predicted answers
    }
        
    std::vector<double> softmax_calculation(const std::vector<double>& pixel_data) {
        // vector to store the probabilities of classes dynamically
        std::vector<double> probility_vector(weights_multi.size(), 0.0); 
        double sum_exp = 0.0; // zk
        
        for (int i = 0; i < probility_vector.size(); i++) { // loop through each pixel
            probility_vector[i] = std::exp(pixel_data[i]); // calculate the exponential of the linear combination of weights and pixel data
            sum_exp += probility_vector[i]; 
        }
        
        for (int i = 0; i < probility_vector.size(); i++) { // loop through classes
            probility_vector[i] /= sum_exp; // normalize the probabilities
        }
        return probility_vector; // return the vector of probabilities
    }

public:
    // Constructor: Sets up our model with default parameters (just incase user forgets to input parameters) and list initialiser
    LogisticRegression(double lr = 0.01, int iters = 1000, double reg_lambda = 0.01)
        : learning_rate(lr), max_iter(iters), lambda(reg_lambda), bias_binary(0.0) {}

    // The core training function where the math happens. X is the training data, y is the true target (0 or 1) for each sample
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        if (X.empty() || y.empty()) return; // Will exit if either x or y is empty. Used as a safety check.

        int num_samples = X.size(); // The number of data rows we have
        int num_features = X[0].size(); // The number of information columns per row

        // Detect mode
        int max_label = 0;
        for (double v : y) max_label = std::max(max_label, (int)v);
        mode = (max_label <= 1 ? 0 : 1); // if labels are less than or = 1 then binary mode, else multi-class mode
      
        // BINARY MODE
        if (mode == 0) {
            // Resizes weight vector to exactly hwo many features we have and sets every single starting weight to zero. Also reset baseline to zero
            weights_binary.assign(num_features, 0.0); 
            bias_binary = 0.0;

            // Gradient Descent Training Loop 
            for (int k = 0; k < max_iter; ++k) {
                // Temporary variables to hold the accumulated gradients
                std::vector<double> dw(num_features, 0.0); // dw; derivative of weights
                double db = 0.0; // db: derivative of bias
                
                // Step 1: Forward Pass & Gradient Accumulation
                for (int i = 0; i < num_samples; ++i) {
                    // Calculate the combination: z = (w * x) + b
                    double z = bias_binary;
                    for (int j = 0; j < num_features; ++j) {
                        z += weights_binary[j] * X[i][j];
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
                    dw[j] = (dw[j] / num_samples) + (2.0 * lambda * weights_binary[j]);
                    
                    // Step 3: Update the actual weights (move in the opposite direction of the gradient)
                    weights_binary[j] -= learning_rate * dw[j];
                }

                // Update the bias in the exact same way
                db = (db / num_samples) + (2.0 * lambda * bias_binary);
                bias_binary -= learning_rate * db;
            }
        }  
        // MULTI-CLASS LOGISTIC REGRESSION
        else {
            int num_classes = max_label + 1; // Calculate classes
            bias_multi.assign(num_classes, 0.0);
            weights_multi.assign(num_classes, std::vector<double>(num_features, 0.0));
                
            for (int image = 0; image < X.size(); image++) { // loop through each image in the dataset
                std::vector<double> initial_score = bias_multi; // initial score
                
                for (int k = 0; k < weights_multi.size(); k++) { // loop through classes
                    for (int p = 0; p < weights_multi[k].size(); p++) { // loop through each pixel
                        initial_score[k] += weights_multi[k][p] * X[image][p]; // WX + b
                    }
                }
                
                std::vector<double> probility_vector = softmax_calculation(initial_score); // calculate the probabilities for each class using softmax
                std::vector<double> error_vector(num_classes, 0.0);
                std::vector<int> int_answer_vector = one_hot_encode(static_cast<int>(y[image])); // convert the true answer to a vector of 0s and 1s
                
                for (int i = 0; i < error_vector.size(); i++) { // loop through classes
                    error_vector[i] = probility_vector[i] - int_answer_vector[i];
                }

                for (int k = 0; k < weights_multi.size(); k++) { // loop through classes
                    for (int p = 0; p < weights_multi[k].size(); p++) { // loop through each pixel
                        weights_multi[k][p] -= learning_rate * error_vector[k] * X[image][p]; // update weights
                    }
                    bias_multi[k] -= learning_rate * error_vector[k]; // update bias
                }
            } 
        }
    }

    // BINARY: The prediction function for new, unseen data 
    std::vector<int> predict_binary(const std::vector<std::vector<double>>& X) const { // stores our final outputs 0's or 1's.
        std::vector<int> predictions;
        // Loops through every sample provided and just like in training calculates the raw score z using learned weights and bias, then
        // converts to probability between 0 and 1 
        for (const auto& sample : X) {
            // First, calculate z for the new sample
            double z = bias_binary;
            for (size_t j = 0; j < weights_binary.size(); ++j) {
                z += weights_binary[j] * sample[j];
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

    // MULTI-CLASS PREDICT FUNCTION
    int predict_multi(const std::vector<double>& pixel_data) {
        std::vector<double> initial_score = bias_multi; // initial score
        for (int k = 0; k < weights_multi.size(); k++) { // loop through classes
            for (int p = 0; p < weights_multi[k].size(); p++) { // loop through each pixel
                initial_score[k] += weights_multi[k][p] * pixel_data[p]; // WX + b
            }
        }
        
        std::vector<double> probility_vector = softmax_calculation(initial_score); // calculate the probabilities for each class using softmax

        int guess = 0;
        double max_probability = probility_vector[0];
        for (int i = 1; i < probility_vector.size(); i++) { // loop through classes to find the class with the highest probability
            if (probility_vector[i] > max_probability) {
                max_probability = probility_vector[i];
                guess = i;
            }
        }
        return guess; // return the predicted label
    }

    // UNified predict that calls the correct one
    int predict(const std::vector<double>& x) {
        if (mode == 0) {
            return predict_binary({x})[0];
        }
        return predict_multi(x);
    }
};
/* use Gemini and VS built in AI to help parts of the code */
} // namespace linear_model
} // namespace sklearn_cpp


