//To run the code use "g++ -Iinclude main_LinearRegression.cpp -o anime ./anime"
#include <iostream>
#include <vector>
// Add the header file for linear regression
#include "sklearn_cpp/linear_model/LinearRegression.hpp"

// Hi Khiem, it is me Ammar! I made this with your favorite anime character in mind ;)
int main (){
    std::cout << std::endl << "ANIME POWER LEVELS LINEAR REGRESSION MODEL" << std::endl;
    
    // 1. ANIME TRAINING DATASET (POWER LEVELS)
    std::vector<std::vector<double>> training_level = {
        {1},             // Goku
        {2},             // Vegeta
        {3},             // Saitama (One Punch Man)
        {4},             // Luffy
        {5},             // Tatsumaki (One Punch Man)
        {6},             // Naruto
        };

    std::vector<double> anime_power_levels = {
        1,               // Goku
        100,             // Vegeta
        6769,            // Saitama (One Punch Man)
        10000,           // Luffy
        9999999,         // Tatsumaki (One Punch Man)
        10000000000,     // Naruto
    };          
      
    // 2. MAKE LINEAR REGRESSION MODEL
    sklearn_cpp::linear_model::LinearRegression model(0.01, 6700, 0.0067);
    std::cout << std::endl << "Ninja Training the model on Anime Power Levels" << std::endl;

    // 3. TRAIN THE ANIME MODEL
    model.fit(training_level, anime_power_levels);
    std::cout << std::endl << "MODEL TRAINING COMPLETE KHIEM!!!" << std::endl;

    // 4. Choose Your Anime Character to Test (Tatsumaki ofcourse ;))
    std::vector<std::vector<double>> SUPER_SAIYAN_TEST = {{5}};
    auto result = model.predict(SUPER_SAIYAN_TEST);

    // 5. Print Congrats + Result
    std::cout << std::endl << "Predicted Power Level:  " << result[0] << std::endl;
    std::cout << std::endl << "CONGRATS GENIUS KHIEM! YOU PREDICTED THEIR POWER LEVEL CORRECTLY!" << std::endl;
    return 0;
}
