//A reference code of template creation through face detction and validation./////
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




const static Scalar colors[] =
{
	Scalar(255,0,0),
	Scalar(255,128,0),
	Scalar(255,255,0),
	Scalar(0,255,0),
	Scalar(0,128,255),
	Scalar(0,255,255),
	Scalar(0,0,255),
	Scalar(255,0,255)
};

auto detectAndCrop(Mat frame, CascadeClassifier  cascade)
{
    double t = 0;
    std::vector<Rect> faces;

    Mat gray;
    Mat crop;
    Mat frame_gray;
    Mat res;
    Mat res_gray;

    // change in gray color
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    //equalizeHist(frame_gray, frame_gray);

    // Detect faces
    t = (double)getTickCount();
    cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));


    

    //Set Region of Interest

    cv::Rect roi_b;
    cv::Rect roi_c;

    size_t ic = 0;  // ic is index of current element
    int ac = 0; // ac is area of current element


    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element

    for (ic = 0; ic < faces.size(); ic++) //Itarate through all current elements (detected faces)
    {

        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);


        ac = roi_c.width * roi_c.height; //Get the area of current element (detected face)
        //std::cout << "ac:"<< ac << endl;

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of the biggest element, at beginning it is same as "current" element
        //std::cout << "ab:" << ab << endl;
        
        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }

        crop = frame(roi_b);
        //std::cout << "crop:" << crop.size[1] << endl;
        //std::cout << "crop:" << crop.size[0] << endl;

        resize(crop, res, Size(160, 160), 0, 0, INTER_LINEAR); // Resize the image to have the same dimension for all the pictures
        //std::cout << "res:" << res.size[0] << endl; // Convert cropped image to Grayscale


        Point pt1(faces[ic].x, faces[ic].x);
        Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y +  faces[ic].width));
        rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
        
    }

    return res;
    
}

auto ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze = false, int unsqueeze_dim = 0)
{
    //std::cout << "image shape: " << img.size() << std::endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        //std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }

    if (show_output)
    {
        //std::cout << tensor_image  << std::endl;
    }
    //std::cout << "tenor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}

void split(char sourceFile[], char destFile1[], char destFile2[]) {
    int chars = 0;
    ifstream sFile;
    sFile.open(sourceFile);
    ofstream file1;
    file1.open(destFile1);
    ofstream file2;
    file2.open(destFile2);

    while (!sFile.eof()) {
        sFile.read(sourceFile + chars, 1);

        cout << sourceFile[chars];
        if (chars % 2 == 0) {
            file1 << sourceFile[chars];
        }
        else {
            file2 << sourceFile[chars];
        }
        chars++;


    }
}


auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 2, 0, 1 })
{
    tensor = tensor.permute(dims);
    return tensor;
}

enum class TemplateRole {
    Enrollment_11 = 0,
    Verification_11,

};

enum class ReturnCode {
    /** Success */
    Success = 0,
    /** Catch-all error */
    UnknownError,
    /** Error reading configuration files */
    ConfigError,
    /** Elective refusal to process the input */
    RefuseInput,
    /** Involuntary failure to process the image */
    ExtractError,
    /** Cannot parse the input data */
    ParseError,
    /** Elective refusal to produce a template */
    TemplateCreationError,
    /** Either or both of the input templates were result of failed
     * feature extraction */
     VerifTemplateError,
     /** Unable to detect a face in the image */
     FaceDetectionError,
     /** The implementation cannot support the number of input images */
     NumDataError,
     /** Template file is an incorrect format or defective */
     TemplateFormatError,
     /**
      * An operation on the enrollment directory
      * failed (e.g. permission, space)
      */
      EnrollDirError,
      /** Cannot locate the input data - the input files or names seem incorrect */
      InputLocationError,
      /** Memory allocation failed (e.g. out of memory) */
      MemoryError,
      /** Error occurred during the 1:1 match operation */
      MatchError,
      /** Failure to generate a quality score on the input image */
      QualityAssessmentError,
      /** Function is not implemented */
      NotImplemented,
      /** Vendor-defined failure */
      VendorError
};

typedef struct EyePair
{
    /** If the left eye coordinates have been computed and
     * assigned successfully, this value should be set to true,
     * otherwise false. */
    bool isLeftAssigned;
    /** If the right eye coordinates have been computed and
     * assigned successfully, this value should be set to true,
     * otherwise false. */
    bool isRightAssigned;
    /** X and Y coordinate of the center of the subject's left eye.  If the
     * eye coordinate is out of range (e.g. x < 0 or x >= width), isLeftAssigned
     * should be set to false, and the left eye coordinates will be ignored. */
    uint16_t xleft;
    uint16_t yleft;
    /** X and Y coordinate of the center of the subject's right eye.  If the
     * eye coordinate is out of range (e.g. x < 0 or x >= width), isRightAssigned
     * should be set to false, and the right eye coordinates will be ignored. */
    uint16_t xright;
    uint16_t yright;

    EyePair() :
        isLeftAssigned{ false },
        isRightAssigned{ false },
        xleft{ 0 },
        yleft{ 0 },
        xright{ 0 },
        yright{ 0 }
    {}

    EyePair(
        bool isLeftAssigned,
        bool isRightAssigned,
        uint16_t xleft,
        uint16_t yleft,
        uint16_t xright,
        uint16_t yright
    ) :
        isLeftAssigned{ isLeftAssigned },
        isRightAssigned{ isRightAssigned },
        xleft{ xleft },
        yleft{ yleft },
        xright{ xright },
        yright{ yright }
    {}
} EyePair;


int
readTemplateFromFile(const string& filename,vector<uint8_t>& templ)
{
    streampos fileSize;
    ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        cerr << "[ERROR] Failed to open stream for " << filename << "." << endl;
        return FAILURE;
    }
    file.seekg(0, ios::end);
    fileSize = file.tellg();
    file.seekg(0, ios::beg);

    templ.resize(fileSize);
    file.read((char*)&templ[0], fileSize);
    return SUCCESS;
}



//int createTemplate(std::shared_ptr<Interface> &implPtr,)



int main(int argc, const char* argv[])
{
    CascadeClassifier cascade;
    cascade.load("D:\\Software\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
    //std::cout << "load detection model successfully \n";
    std::string inputName = "sample image";
    std:ifstream f("D:/C++/Input/input.txt");
    std::string line;
    std::vector<std::string> mylist;
    while (std::getline(f, line))
    {
        mylist.push_back(line);
        //std::cout << mylist.back() << std::endl;
    }

    
    //o << fn << std::endl;
    std::vector<std::string> mylist2;
    std::vector<float> finallist;
    std::string r18_model_path = "D:\\C++\\demo1\\recognition.pt";
    //std::cout << "loading recognition model successfully \n";
    torch::jit::script::Module r18;
    r18 = torch::jit::load(r18_model_path);
    //std::cout << "loading r18 successfully \n";
    int nextnum = 000001;
    for (size_t k = 0; k < mylist.size(); ++k)
    {
        cv::Mat image = cv::imread(mylist[k], IMREAD_COLOR);
        if (image.empty())
        {
            cout << "Could not read" << inputName << endl;
            return 1;
        }
        else
        {   
            //loading Input image
            Mat frame1 = image.clone();
            //cv::imshow("Input",frame1);
            
            //clock_t detect_start, detect_end;
            //detect_start = clock();
            // Applying detection and crop function on the image 
            cv::Mat frame2 = detectAndCrop(frame1, cascade);
            
            //detect_end = clock() - detect_start;
            //std::cout << "detection & cropping time per image:" << detect_end/(double)CLOCKS_PER_SEC << endl;
            // Convert cropped image to tensor
            at::Tensor tensor_cropped = ToTensor(frame2);
            //std::cout << "Cropped Tensor Size:" << tensor_cropped.sizes() << endl;
            //tensor_cropped = tensor_cropped.toType(c10::kFloat).div(1);
            
            // Transpose cropped tensor 
            tensor_cropped = transpose(tensor_cropped, { (2), (0), (1) });
            


            // Unsqueeze the cropped tensor to expand dimension 
            auto tensor_cropped1 = tensor_cropped.unsqueeze(0);
            //std::cout << "t_Cropped:" << tensor_cropped1 << endl;

            //Normalizing the cropped tensor 
            tensor_cropped = tensor_cropped1.toType(c10::kFloat).div(255);
            
            // converting the tensor into torch jit Ivalue for loading in the torch model
            auto inputs1 = std::vector<torch::jit::IValue>{ tensor_cropped};
            
         
            try
            {     
                //clock_t recog_start, recog_end;
                //recog_start = clock();

                //loading tensor into the recognition model, output would be a (1,512) vector
                at::Tensor cropped_tensor = r18.forward(inputs1).toTensor();
                //recog_end = clock() - recog_start;
                //std::cout << "recognition time per image:" << recog_end / (double)CLOCKS_PER_SEC << endl;
               // std::cout << "The output of the recognition model:" <<cropped_tensor.sizes()<< endl;

                //converting tensor to vector
                cropped_tensor = cropped_tensor.contiguous();
                std::vector<float> v(cropped_tensor.data_ptr<float>(), cropped_tensor.data_ptr<float>() + cropped_tensor.numel());
                std::cout << "Vector:" << v.size() << endl;
                
                
                
                
                //torch::save(v, "./my_tensor_vec.template");
                //std::ostringstream stream;
                //torch::save(v, stream);
                
                string data = to_string(nextnum);
                
                ofstream file;
                string filepath = "D:/C++/Template/";
                file.open(filepath + data  + ".template");
                file << v;
                file.close();

                
                //std::string line;
                
               
                 mylist2.push_back(filepath + data + ".template\n");
                    //std::cout << mylist.back() << std::endl;
                
                 std::cout << "mylist:" <<mylist2.size()<< endl;

                
                nextnum = nextnum++;
                

                

            }

            
            
            catch (const c10::Error& e)
            {
                std::cerr << "error loading the model\n" << e.msg();
                return -1;
            }//catch done
            //std::cout << "Cropped Tensor Size:" << cropped_tensor << endl;

        }
        








        //string t2 = "D:/C++/Input/log.txt/2.template";

      
    }

    std::ofstream z;
    z.open("D:/C++/Input/log.txt");
    z << mylist2;
    z.close();

    





    //////////////// Validation ////////////////////////

    
    String path("D:/C++/Template/*.template"); //select only jpg
    vector<String> fn;
    cv::glob(path, fn, true); // recurse
    for (size_t x = 0; x < fn.size(); x=x+2)
    {
        std::ifstream o(fn[x]);

        std::vector<float> t1;
        double element;
        while (o >> element)
        {
            t1.push_back(element);

        }
        //std::cout << "element:" << element << endl;
        // std::cout << "t1: " << t1 << std::endl;
        o.close();

        std::ifstream p(fn[x+1]);
        std::vector<float> t2;
        double element1;
        while (p >> element1)
        {
            t2.push_back(element1);

        }
        //std::cout << "t2: " << t2 << std::endl;
        p.close();

        std::sort(t1.begin(), t1.end());
        std::sort(t2.begin(), t2.end());

        std::vector<float>difference;
        std::set_difference(
            t1.begin(), t1.end(),
            t2.begin(), t2.end(),
            std::back_inserter(difference)
        );

        
        transform(difference.begin(), difference.end(), difference.begin(), [](float x) {return x * x * 1000; });
        float sum_of_elements = std::accumulate(difference.begin(), difference.end(), 0.0f);

        std::cout << "similarity:" << sum_of_elements / 512 << std::endl;
        finallist.push_back(sum_of_elements/512);
    }

    std::ofstream z1;
    z1.open("D:/C++/Input/match.txt");
    z1 << finallist;
    z1.close();


    std::cout << "ok\n";
    std::system("pause");
    return 0;
}


