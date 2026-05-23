#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include "math/multiclass_logistic_regression.hpp"

void load_read_mnist(const std::string& filepath, std::vector<std::vector<double>>& pixels, std::vector<double>& answers){ //load MNIST from Data/mnist.csv
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl; //error showing
        return;
    }else {
        std::cout << "Successfully opened file: " << filepath << std::endl; //success showing
    }
    std::string line;
    while (std::getline(file, line)) { //read line by line
        std::stringstream pixel(line); //set pixel to data in line
        std::vector<double> pixel_values; //vector to store pixel values
        std::string value; //string to store pixel value

        while (std::getline(pixel, value, ',')) { //read pixel by pixel until column is 0
            pixel_values.push_back(std::stod(value)); //store value to double
        }
        answers.push_back(pixel_values.back()); //store the answer (the last value in the pixel vector)
        pixel_values.pop_back(); //remove the answer from the pixel vector
        pixels.push_back(pixel_values); //store pixel values for the current image
    }


    file.close(); //close after reading
}

int main(){
    std::vector<std::vector<double>> pixels; //vector to store pixel values
    std::vector<double> answers; //vector to store answers
    load_read_mnist("Data/mnist_full.csv", pixels, answers); //load and read MNIST dataset from csv file

    std::cout << "Number of images: " << pixels.size() << std::endl; //print number of images
    std::cout << "Number of labels: " << answers.size() << std::endl; //print number of labels
    std::cout << "Data for the first image: " << pixels[0].size() << std::endl; //print data for the first image
    std::cout << "Answer for the first image: " << answers[0] << std::endl; //print answer for the first image

    std::cout << "Training Logistic Regression model..." << std::endl; //traning start
    sklearn_cpp::LogisticRegression model(0.01); //load .hpp
    model.fit(pixels, answers); //train the model with the loaded data
    std::cout << "Model training completed." << std::endl; //training completed

    load_read_mnist("Data/mnist_test.csv", pixels, answers); //load and read MNIST test dataset from csv file
    int guess = model.predict(pixels[0]); //predict the answer for the first image using the trained model
    std::cout << "Predicted answer for the first image: " << guess << std::endl;
    std::cout << "Actual answer for the first image: " << answers[0] << std::endl;

    return 0;
}
/* uses Gemni and VS built in AI to help parts of the code
for the dataset, due to the file size is too big, i deleted them for git, if download -> put into Data folder
change file just edit line 37 (traning), 49 (actual predeict)
*/