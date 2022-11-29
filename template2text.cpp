#include<algorithm>
#include<torch/torch.h>
#include<torch/script.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core/utility.hpp>
#include<iostream>
#include<memory>
#include<string>
#include<fstream>
#include<vector>
#include<list>
#include<cmath>
#include<ctime>
#include<algorithm>
#include<numeric>



#define SUCCESS 0
#define FAILURE 1


using namespace std;
using namespace cv;


int main(int argc, const char* argv[])
{
  string enroll_template_path = "D:/C++/todaytemplates/validation_ours/odd/*.template";
	vector<string>fn2;
	int nextnum = 2;
	cv::glob(enroll_template_path, fn2, true); // template nicche fn2 er moddhe
	string filepath = "D:/C++/todaytemplates/val_ours_txt/"; // je file e rakhbo tar path



	for (size_t k = 0; k < fn2.size(); k++)
	{
		string m = fn2[k];
		std::ifstream file(m, std::ios::binary);;
		streampos filesize;
		file.seekg(0, ios::end);
		filesize = file.tellg();
		file.seekg(0, ios::beg);
		std::cout << "filesize:" << filesize << endl;
		std::vector<byte> filedata(filesize);
		file.read((char*)&filedata[0], filesize);
		
		
		stringstream ss;
		ss << std::setw(4) << std::setfill('0') << nextnum;

		std::string s = ss.str();
		std::cout << s << endl;

		//string data;
		//creating file in the path directory
		std::ofstream filename;
		//opening file
		filename.open(filepath + s + ".txt");

		for (unsigned i = 0; i < 511; i++)
		{
			std::vector<float>x2;
			x2 = ToFloats(filedata);
			std::cout << x2[i] << endl;
			filename << x2[i];
			file.close();
		}


		nextnum = nextnum+2;

	}
	
}
