#include <iostream>
#include <vector>
#include <string>
#include <fstream>  // Required for reading files
#include <sstream>  // Required for splitting strings

// Function to load data from a CSV file.
// It takes the filename, and two empty vectors (X for features, y for labels) passed by reference (&) so it can fill them.
bool load_csv_data(const std::string& filename, std::vector<std::vector<double>>& X, std::vector<double>& y) {
    
    // Open the file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'. Check if the path is correct!" << std::endl;
        return false;
    }

    std::string line;
    
    // Read the file line by line until we reach the end
    while (std::getline(file, line)) {
        
        // Skip completely empty lines
        if (line.empty()) {
            continue; 
        }

        //Set up a string stream to break down the current line
        std::stringstream ss(line);
        std::string value_str;
        std::vector<double> row_features;

        //Extract every comma-separated value in the row
        while (std::getline(ss, value_str, ',')) {
            try {
                // Convert the text string into a double (decimal number) and add it to our row
                row_features.push_back(std::stod(value_str));
            } catch (const std::invalid_argument& e) {
                // If it hits a column header (like the word "cement" in concrete.csv), it skips the row
                break; 
            }
        }

        // Separate the features from the label
        // If the row successfully populated with numbers
        if (!row_features.empty()) {
            // The last item in the row is our target classification label (y)
            y.push_back(row_features.back());
            
            // Remove the label from our feature list so X only contains data attributes
            row_features.pop_back();
            
            // Add the cleaned feature row to our main dataset X
            X.push_back(row_features);
        }
    }

    // Close the file to free up system memory
    file.close();
    return true;
}
