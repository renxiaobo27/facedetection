
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <fstream>



using namespace dlib;
using namespace std;
//
//// A 5x5 conv layer that does 2x downsampling
//template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
//// A 3x3 conv layer that doesn't do any downsampling
//template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;
//
//// Now we can define the 8x downsampling block in terms of conv5d blocks.  We
//// also use relu and batch normalization in the standard way.
//template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;
//
//// The rest of the network will be 3x3 conv layers with batch normalization and
//// relu.  So we define the 3x3 block we will use here.
//template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;
//
//// Finally, we define the entire network.   The special input_rgb_image_pyramid
//// layer causes the network to operate over a spatial pyramid, making the detector
//// scale invariant.
//using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;



template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler5  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler5<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

void evaluation(char** argv)
{
	//string file = "/home/tcl-admin/Downloads/FDDB-folds/FDDB-fold-05.txt";
	//string out_file = "/home/tcl-admin/face/myface/evaluation/fold-05-out.txt";

	string file(argv[1]);
	string out_file(argv[2]);

	ifstream infile(file);
	ofstream outfile(out_file);

	string line;
	int cnt = 0;
	net_type net;
	deserialize("model/mmod_network-40x40-xiaobo-complex.dat") >> net;
	string imageRoot = "/home/tcl-admin/Downloads/";
	//cv::Mat img;
	matrix<rgb_pixel> img;
    image_window win;


	while(getline(infile,line))
	{
		cout<<line<<endl;
		cnt++;
		outfile<<line<<endl;

		string imagePath = imageRoot + line+ ".jpg";

		load_image(img,imagePath);

		win.set_image(img);
		std::vector<dlib::mmod_rect> dets = net(img);
		outfile<<dets.size()<<endl;
		for (auto& d : dets)
		{
			cout<<d.rect.width()<<" "<<d.rect.height()<<" "<<d.detection_confidence<<endl;
			//win.add_overlay(d,rgb_pixel(128,128,128));
			outfile<<d.rect.left()<<" "<<d.rect.top()<<" "<<d.rect.width()<<" "<<d.rect.height()<<" "<<d.detection_confidence<<endl;
		}


		//cin.get();
		//win.clear_overlay();
	}
	cout<<cnt;
	infile.close();
	outfile.close();

}
// ----------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//evaluation(argv);

	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("model/shape_predictor_68_face_landmarks.dat") >> pose_model;

	net_type net;
	deserialize("model/mmod_network-40x40-xiaobo-complex.dat") >> net;
	try
	{
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            cap >> temp;
            //cout<<temp.rows<<" "<<temp.cols<<endl;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces
            std::vector<rectangle> faces = detector(cimg);
            matrix<rgb_pixel> image;
            assign_image(image,cimg);
            std::vector<dlib::mmod_rect> dets = net(image);
            // Find the pose of each face.
//            std::vector<full_object_detection> shapes;
//            for (unsigned long i = 0; i < faces.size(); ++i)
//                shapes.push_back(pose_model(cimg, faces[i]));

            // Display it all on the screen
            win.clear_overlay();
            win.set_image(image);
            for(auto &&face:faces)
            	win.add_overlay(face,rgb_pixel(0,255,0));

            //win.add_overlay(faces,rgb_pixel(0,255,0));
            //win.add_overlay(dets,rgb_pixel(255,0,0));
            for (auto& d : dets)
            {
            	cout<<d.rect.width()<<" "<<d.rect.height()<<" "<<d.detection_confidence<<endl;
            	win.add_overlay(d,rgb_pixel(255,0,0));
            }

        }
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }

}
