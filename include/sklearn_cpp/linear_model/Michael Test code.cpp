#include <iostream>
#include <vector>

#include "logistic_Regression_New.hpp" //Pulls in header file

int main() {
    

    //Creating a tiny mock datatset
    std::vector<std::vector<double>> X_train = {
        {4.5, 90.0},  // Pass
        {1.0, 30.0},  // Fail
        {5.0, 95.0},  // Pass
        {1.5, 45.0}   // Fail
    };
    //Target Binary Variables
    std::vector<double> y_train = {1, 0, 1, 0};

    // Instantiate the model from the custom namespace
    sklearn_cpp::linear_model::LogisticRegression model(0.1, 2000, 0.001);

    
    model.fit(X_train, y_train);
    

    // Khiem you can input the students stats here that you want to test. Left term is hours spent studying and right is percentage attendance
    // Make sure the datatype is double though
    
    double hours = 4.0;         
    double attendance = 85.0;   

    // Packaged student data as a 1D Vector 
    std::vector<double> single_student = {hours, attendance};

    // Generates the prediction using the function from the header file
    int result = model.predict(single_student);

    std::cout << "\n================ RESULTS FOR LOGISTIC REGRESSION MODEL ================" << std::endl;
    std::cout << "Testing Stats: " << hours << " Hours Studied, " << attendance << "% Attendance." << std::endl;
    std::cout << "Predicted Outcome: " << (result == 1 ? "PASS (1)" : "FAIL (0)") << std::endl;
    

    return 0;
}