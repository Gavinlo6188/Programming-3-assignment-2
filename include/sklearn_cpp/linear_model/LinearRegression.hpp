#include <vector>

namespace sklearn_cpp {
    namespace linear_model {

        class LinearRegression{
        private:
        // Private member variables will be defined here
        double bias;               // starting point 
        double learning_rate;      // controls how big the steps are
        int maximum_iterations;    // decides how long the training will run
        std::vector<double> weights; // Vector to store weights for their respective features 

        public: 
        // Constructor to set learning rate and iteration number
        LinearRegression(double lr = 0.02, int max_itr = 1000) : 
        learning_rate{lr}, maximum_iterations{max_itr}, bias{0.0} {
        // Initially, weightss are empty, until we fit the model into the data
        }   
                                
        // Helper function to compute the dot product of the predicted weights and input features
        double compute_dot_product(const std::vector<double> &feature) const {
            double dot_product{0.0};
            for (int i=0; i < weights.size(); i++){
            dot_product += weights[i] * feature[i];
            }
            return dot_product;
        }

        double predict_function(const std::vector<double> &feature) const {
            // Add the bias
            double predicted_value{compute_dot_product(feature) + bias};
            return predicted_value;
        }
 
        // This function will take the responsibility of aplying th learning algorithm to the data 
        void fit_function(const std::vector<std::vector<double>> &samples, const std::vector<double> &targets) {    // Update the weights and bias accordingly
            // Compute predictions, x is the vector of vector features, & y is the vector of outputs
            double feature_value{0.0};
            int number_of_features{samples[0].size()}; // number of features in one sample = no. of weights
            weights.resize(number_of_features, 0.0); // Set weight vector to no. of features & initialise them all to 0.0;
            std::vector<double> gradient_weight(number_of_features, 0.0);
            double gradient_bias{0.0};
            //double gradient_weight_sum{0.0};
            //double gradient_bias_sum{0.0};     
            //double weight_old {0.0};
            //double bias_old{0.0};
     
            for (int iterations=0; iterations < 1000; iterations++) {
                for (int i=0; i < samples.size(); i++) {
                    // Compute errors
                    double error = predict_function(samples[i]) - targets[i]; 
                    // Store the value of the j-th feature in the i-th sample (i.e. the value inside the row)
                    for (int j = 0; j < number_of_features; j++){
                        feature_value = samples[i][j];
                        // Accumulate the gradients for weights and biasses 
                        gradient_weight[j] += feature_value * error;
                        gradient_bias += error; 
                    }        
                }
                gradient_bias /= samples.size();
                for (int j = 0; j < number_of_features; j++){
                    // Update weights and bias using the calculated errors and also the learning rate
                    gradient_weight[j] /= samples.size();
                    weights[j] = weights[j] - learning_rate * (gradient_weight[j]);
                }
                bias = bias - learning_rate * (gradient_bias);  
                
                // Reset the sum for the next iteration
                gradient_weight.assign(gradient_weight.size(), 0.0); 
                gradient_bias=0.0; 
            }
        }
        };
    }
}
