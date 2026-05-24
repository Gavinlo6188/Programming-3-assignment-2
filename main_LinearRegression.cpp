//To run the code use:  "g++ -Iinclude main_LinearRegression.cpp"
#include <iostream>
#include <vector>
// Add the header file for linear regression
#include "sklearn_cpp/linear_model/LinearRegression.hpp"

// Hi Khiem, it is me Ammar! I made this with your favorite anime character in mind ;)
int main (){
    std::cout << std::endl << "ANIME POWER LEVELS LINEAR REGRESSION MODEL" << std::endl;
    
    // 1. ANIME TRAINING DATASET (POWER LEVELS)
    std::vector<std::vector<double>> training_level = {
        {1.0},             // Goku
        {2.0},             // Vegeta
        {3.0},             // Saitama (One Punch Man)
        {4.0},             // Luffy
        {5.0},             // Tatsumaki (One Punch Man)
        {6.0},             // Naruto
        };

    std::vector<double> anime_power_levels = {
        1000.0,               // Goku
        2000.0,             // Vegeta
        3500.0,            // Saitama (One Punch Man)
        4000.0,           // Luffy
        4200.0,         // Tatsumaki (One Punch Man) // also 42 ur favourite number :D
        6900.0,     // Naruto
    };          
      
    // 2. MAKE LINEAR REGRESSION MODEL
    sklearn_cpp::linear_model::LinearRegression model(0.001, 1042); // Best values for learning rate and iteration number that I found after testing
    std::cout << std::endl << "Ninja Training the model on Anime Power Levels" << std::endl;

    // 3. TRAIN THE ANIME MODEL
    model.fit(training_level, anime_power_levels);
    std::cout << std::endl << "MODEL TRAINING COMPLETE KHIEM!!!" << std::endl;

    // 4. Choose Your Anime Character to Test (Tatsumaki ofcourse ;))
    std::vector<double> SUPER_SAIYAN_TEST = {2.0};  // Khiem, please change this value to test other characters!
    auto result = model.predict(SUPER_SAIYAN_TEST);

    // 5. Print Congrats + Result
    std::cout << std::endl << "Predicted Power Level:   " << result << std::endl;
    std::cout << std::endl << "CONGRATS GENIUS KHIEM! YOU PREDICTED THEIR POWER LEVEL CORRECTLY!" << std::endl;
    return 0;
}
