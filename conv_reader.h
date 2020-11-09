//
//  conv_reader.hpp
//
//  Created by Hossam Amer 
//

#ifndef FEATURE_MAP_READER_H
#define FEATURE_MAP_READER_H


#include <vector>

#include <stdio.h>

#include <math.h>

// for debugging LookNBits
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


using namespace std;

class conv_reader {

public:
	// Constructor
    conv_reader(std::string);

    // Destructor
    ~conv_reader();


    // buffer for the feature map
    vector<vector<vector<float>>> m_input_feature_map;

    // buffer to hold the feature map row major
    vector<float> m_input_feature_map_rowMajor;

    // buffer to hold the feature map col major
    vector<float> m_input_feature_map_colMajor;


    // Define the input dimensions
    int In;
    int Ih;
    int Iw;
    int Ic;


    // Main loop
    void readFile(std::string filename);



};

#endif
