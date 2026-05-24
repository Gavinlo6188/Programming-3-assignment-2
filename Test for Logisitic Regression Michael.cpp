#include <iostream>
#include <vector>

// This pulls in the header file 
#include "Logistic_Regression.hpp"

int main() {
    // Create a tiny mock dataset
    // Matrix X: Column 0 = Study Hours, Column 1 = Attendance %
    std::vector<std::vector<double>> X_train = {
        {4.5, 90.0},  // Student 1 (Studied hard, good attendance) -> Pass
        {1.0, 30.0},  // Student 2 (Barely studied, skipped class) -> Fail
        {5.0, 95.0},  // Student 3 (Excellent stats)               -> Pass
        {1.5, 45.0}   // Student 4 (Low stats)                     -> Fail
    };

    // Vector y: Target binary labels (1 = Pass, 0 = Fail)
    std::vector<double> y_train = {1, 0, 1, 0};

    // Instantiate model from custom namespace
    sklearn_cpp::linear_model::LogisticRegression model(0.1, 2000, 0.001);

    model.fit(X_train, y_train);
    
    // Khiem you can input the students stats here that you want to test. Left term is hours spent studying and right is percentage attendance
    // Make sure the datatype is double though
    double hours = 4.0;         // Set the study hours here
    double attendance = 85.0;   // Set the attendance percentage here
    

    // Package the student's data into a 2D matrix layout for the model
    std::vector<std::vector<double>> X_test = {{hours, attendance}};

    // Generates the prediction using the function from the header file
    std::vector<int> predictions = model.predict(X_test);
    int result = predictions[0];

    
    std::cout << "\n================ RESULTS FOR LOGISTIC REGRESSION MODEL ================" << std::endl;
    std::cout << "Testing Stats: " << hours << " Hours Studied, " << attendance << "% Attendance." << std::endl;
    std::cout << "Predicted Outcome: " << (result == 1 ? "PASS (1)" : "FAIL (0)") << std::endl;
    

    return 0;
}