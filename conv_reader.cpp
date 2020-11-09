//
//  conv_reader.cpp
//
//  Created by Hossam Amer 
//


#include "conv_reader.h"
#include <fstream>
#include <assert.h>

conv_reader::conv_reader(std::string filename) {

	// Create the dimension vector
	std::ifstream infile(filename.c_str());

	// Read the dimensions
	infile >> In;
	infile >> Ih;
	infile >> Iw;
	infile >> Ic;

	// Create the 3D Vector
	m_input_feature_map.resize(Ih);
    for(int i = 0; i < Ih; i++)
    {
        m_input_feature_map[i].resize(Iw);
        for(int j = 0; j < Iw; j++)
            m_input_feature_map[i][j].resize(Ic);
    }

    // Create the row and col major vectors:
    m_input_feature_map_rowMajor.resize(In*Ih*Iw*Ic);
    m_input_feature_map_colMajor.resize(In*Ih*Iw*Ic);
}


// Define the destructor
conv_reader::~conv_reader(){}

void conv_reader::readFile(std::string filename)
{

	// Create the dimension vector
	std::ifstream infile(filename.c_str());

	// Read the dimensions
	int N, H, W, C;
	infile >> N;
	infile >> H;
	infile >> W;
	infile >> C;

	// Read the data
	float featureElement;
	int count = 0;
	int channels = 0;

	for(int c = 0; c < Ic; ++c)
	{
		for(int h = 0; h < Ih; ++h)
		{
			for(int w = 0; w < Iw; ++w)
			{
				float a;
				infile >> a;

				int row_major_index = w + h*Iw + c*(Ih*Iw);
				int col_major_index = h + w*Iw + c*(Ih*Iw);

				m_input_feature_map[h][w][c]                  = a; 
				m_input_feature_map_rowMajor[row_major_index] = a;
				m_input_feature_map_colMajor[col_major_index] = a;
				count++;
			}
		}

	}
	

	// Make sure that you read the same number of elements
	assert(count == N*H*W*C);
}

int main(int argc, const char * argv[]) 
{

	std::string fileName = "/Users/hossam.amer/7aS7aS_Works/work/my_Tools/tf_conv/data/IV3_dataset_IV3_ImgID_7_Conv_82.input";

	// Define the reader
	conv_reader featureMap_reader(fileName);
	featureMap_reader.readFile(fileName);

	return 0;
}