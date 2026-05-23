#pragma once

#include <iostream>
#include <cmath>
#include <vector>

namespace math{

    class LogisticRegression{
        private:
        std::vector<std::vector<double>> weight; //weights for the logistic regression model
        double weight_value;
        double bias;

        std::vector<int> answer_predicted_vector(int answer_predicted_int) {
            std::vector<int> answer_predicted_vector(weight.size(), 0); //vector to store predicted answers
            if (answer_predicted_int >= 0 && answer_predicted_int < weight.size()) { //check if the predicted answer is within the valid range
                answer_predicted_vector[answer_predicted_int] = 1; //set the predicted answer to 1
            }
            return answer_predicted_vector; //return the vector of predicted answers
        };
        
        std::vector<double> softmax_calculation(const std::vector<double>& pixel_data){
            std::vector<double> probility_vector(1.0 , 0.0); //vector to store the probabilities for each class
            double sum_exp = 0.0; //variable to store the sum of exponentials
            for (int i = 0; i < weight.size() + 1; i++) { //loop through each class
                
                probility_vector[i] = std::exp(pixel_data[i]); //calculate the exponential of the linear combination of weights and pixel data
                sum_exp += probility_vector[i]; //add the exponential to the sum
            }
            for (int i = 0; i < probility_vector.size(); i++) { //loop through each class
                probility_vector[i] /= sum_exp; //normalize the probabilities
            }
            return probility_vector; //return the vector of probabilities
        }

        public:
        LogisticRegression(const std::vector<std::vector<double>>& W, double b): weight(W), bias(b){} //constructor to initialize weights
        
        void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) { //X = data, y = answers
            //fit the logistic regression model to the training data
        };

    };

}