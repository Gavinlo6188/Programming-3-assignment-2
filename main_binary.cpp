#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "sklearn_cpp/linear_model/LogisticRegression.hpp"

/**
 * load_data: A helper function to read comma-separated values from a file.
 * X: A 2D vector that will store all feature data (all columns except the last).
 * y: A 1D vector that will store the target labels (the last column).
 */
bool load_data(const std::string& filename, std::vector<std::vector<double>>& X, std::vector<double>& y) {
    // Attempt to open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Alert: Could not open the file at " << filename << ". Check your path." << std::endl;
        return false;
    }

    std::string line;
    // Read the file one line at a time
    while (std::getline(file, line)) {
        // Skip empty lines to prevent errors during processing
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;

        // Split the line by commas
        while (std::getline(ss, val, ',')) {
            try { 
                // Convert string to double. If it's a header word, the catch block skips it.
                row.push_back(std::stod(val)); 
            } catch (...) { break; } 
        }

        // If we found data, store the last value as our label (y) and the rest as features (X)
        if (!row.empty()) {
            y.push_back(row.back()); // Store target label
            row.pop_back();          // Remove label from row
            X.push_back(row);        // Store remaining features
        }
    }
    file.close();
    return true;
}

int main() {
    // Storage for training data
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    //Load data
    // Specify the file to load. Change this to "concrete.csv" or "mnist.csv" as needed.
    std::string data_file = "ecg.csv";
    std::cout << "Loading data from " << data_file << "..." << std::endl;

    if (!load_data(data_file, X, y)) {
        return 1; // Exit if loading failed
    }
    std::cout << "Successfully loaded " << X.size() << " samples." << std::endl;

    // Train Model
    // Hyperparameters: Learning Rate (0.01), Iterations (1000), Regularisation (0.01)
    sklearn_cpp::linear_model::LogisticRegression model(0.01, 1000, 0.01);
    
    
    model.fit(X, y); // The model detects mode (binary/multi) automatically
    

    // Evaluate Performace
    std::cout << "\n--- Checking Predictions on First 5 Samples ---" << std::endl;
    int correct_guesses = 0;

    for (int i = 0; i < 5 && i < (int)X.size(); ++i) {
        int prediction = model.predict(X[i]);
        bool is_correct = (prediction == static_cast<int>(y[i]));
        
        if (is_correct) correct_guesses++;

        std::cout << "Sample " << i << " | Actual: " << y[i] 
                  << " | Model Guess: " << prediction 
                  << (is_correct ? " [PASS]" : " [FAIL]") 
                  << std::endl;
    }

    std::cout << "\nBatch accuracy for these 5 samples: " << (correct_guesses / 5.0) * 100 << "%" << std::endl;
    

    return 0; // Everything ran successfully
}
