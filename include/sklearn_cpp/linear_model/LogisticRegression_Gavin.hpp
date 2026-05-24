#pragma once
#include <cmath>
#include <vector>

namespace sklearn_cpp{
    class LogisticRegression{
        private:
        std::vector<std::vector<double>> weight; //weights for the logistic regression model
        double weight_value;
        std::vector<double> bias;
        double learning_rate;

        std::vector<int> one_hot_encode(int answer_int) {
            std::vector<int> answer_vector(weight.size(), 0); //vector to store predicted answers
            if (answer_int >= 0 && answer_int < weight.size()) { //check if the predicted answer is within the valid range
                answer_vector[answer_int] = 1; //set the predicted answer to 1
            }
            return answer_vector; //return the vector of predicted answers
        };
        
        std::vector<double> softmax_calculation(const std::vector<double>& pixel_data){
            std::vector<double> probility_vector(10 , 0.0); //vector to store the probabilities of 0~9
            double sum_exp = 0.0; //zk
            for (int i = 0; i < weight.size(); i++) { //loop through each pixel
                
                probility_vector[i] = std::exp(pixel_data[i]); //calculate the exponential of the linear combination of weights and pixel data
                sum_exp += probility_vector[i]; 
            }
            for (int i = 0; i < probility_vector.size(); i++) { //loop through 0~9
                probility_vector[i] /= sum_exp; //normalize the probabilities
            }
            return probility_vector; //return the vector of probabilities
        }

        public:
        LogisticRegression(double lr){
            learning_rate = lr;

            bias.assign(10, 0.0); //initial condition
            weight.assign(10, std::vector<double>(784, 0.0)); //initial condition
        }
        
        void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) { //X = data, y = answers
            
            for (int image = 0; image < X.size(); image++) { //loop through each image in the dataset
                std::vector<double> initial_score = bias; //initial score
                for (int k = 0; k < weight.size(); k++) { //loop through 0~9
                    for (int p = 0; p < weight[k].size(); p++) { //loop through each pixel
                        initial_score[k] += weight[k][p] * X[image][p]; //WX + b
                    }
                    
                }
                std::vector<double> probility_vector = softmax_calculation(initial_score); //calculate the probabilities for each class using softmax

                std::vector<double> error_vector(10, 0.0);

                std::vector<int> int_answer_vector = one_hot_encode(static_cast<int>(y[image])); //convert the true answer to a vector of 0s and 1s
                for (int i = 0; i < error_vector.size(); i++) { //loop through 0~9
                    error_vector[i] = probility_vector[i] - int_answer_vector[i];
                }

                for (int k = 0; k < weight.size(); k++) { //loop through 0~9
                    for (int p = 0; p < weight[k].size(); p++) { //loop through each pixel
                        weight[k][p] -= learning_rate * error_vector[k] * X[image][p]; //update weights
                    }
                    bias[k] -= learning_rate * error_vector[k]; //update bias
                }
                

            }
        };

        int predict(const std::vector<double>& pixel_data) {
            std::vector<double> initial_score = bias; //initial score
            for (int k = 0; k < weight.size(); k++) { //loop through 0~9
                for (int p = 0; p < weight[k].size(); p++) { //loop through each pixel
                    initial_score[k] += weight[k][p] * pixel_data[p]; //WX + b
                }
                
            }
            std::vector<double> probility_vector = softmax_calculation(initial_score); //calculate the probabilities for each class using softmax

            int guess = 0;
            double max_probability = probility_vector[0];
            for (int i = 1; i < probility_vector.size(); i++) { //loop through 0~9 to find the class with the highest probability
                if (probility_vector[i] > max_probability) {
                    max_probability = probility_vector[i];
                    guess = i;
                }
            }
            return guess; //return the predicted label
        }

    };
}
/* use Gemni and VS built in AI to help parts of the code*/
