/*

CMP3110M Parallel Computing Assignment 1
By George Markham
MAR15561551

*/


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

/* Vector library to enable vector use*/
#include <vector>

/* Libraries to read in data from files and print to terminal, needed as I'll be loading in the temperatures from a .txt file and kernel from a .cl file */
#include <iostream>
#include <fstream>

/* Load in string library*/
#include <string>

/* Use ctime to deal with the time so it can be stored correctly in one vector reducing the amount of memory used */
#include <ctime>

/* Lets me easily do some maths */
#include <math.h>

/*OpenCL Library (Only need windows as this will only be running on windows)*/
#include <CL/cl.hpp>

using namespace std;


float sum(cl::Context context, cl::CommandQueue queue, cl::Kernel sum_kernel, vector<float> input) {
	vector<float> output(input.size());

	//Sizes so memory can be allocated correctly
	size_t input_size = input.size() * sizeof(float);
	size_t output_size = output.size() * sizeof(float);
	size_t local_size = 10;

	//Initialise sum value to 0 in order to hold the output, this also means that if anything fails the function will always return a value
	float sum_val = 0;

	//Create the buffers to pass memory between host and kernel
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, output_size);

	//Write input into the input buffer
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, input.data());

	//Fill the output buffer with 0 values, to be filled with data by the kernel
	queue.enqueueFillBuffer(output_buffer, 0, 0, output_size);

	//Set the buffers as arguments
	sum_kernel.setArg(0, input_buffer);
	sum_kernel.setArg(1, output_buffer);
	//Allocate local memory and pass it to the kernel
	sum_kernel.setArg(2, cl::Local(local_size * sizeof(float)));

	//Run the kernel
	queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(local_size));
	
	//Read the kernel's output into the correct vector
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size, &output[0]);

	//Get the sum from the output vector
	sum_val = output.front();

	//Print sum and mean to terminal
	printf("sum = %f\n", sum_val);
	printf("mean = %f\n\n\n", sum_val / input.size());

	//Return the sum
	return sum_val;
}

float find_max(cl::Context context, cl::CommandQueue queue, cl::Kernel max_kernel, vector<float> input) {
	vector<float> output(input.size());

	vector<float> input_two;

	//Sizes so memory can be allocated correctly
	size_t input_size = input.size() * sizeof(float);
	size_t output_size = output.size() * sizeof(float);
	size_t local_size = 10;

	//Initialise max value to 0 in order to hold the output, this also means that if anything fails the function will always return a value
	float max_val = 0;

	//Create the buffers to pass memory between host and kernel
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, output_size);

	//Write input into the input buffer
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, input.data());

	//Fill the output buffer with 0 values, to be filled with data by the kernel
	queue.enqueueFillBuffer(output_buffer, 0, 0, output_size);

	//Set the buffers as arguments
	max_kernel.setArg(0, input_buffer);
	max_kernel.setArg(1, output_buffer);
	//Allocate local memory and pass it to the kernel
	max_kernel.setArg(2, cl::Local(local_size * sizeof(float)));

	//Run the kernel
	queue.enqueueNDRangeKernel(max_kernel, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(local_size));
	
	//Read the kernel's output into the correct vector
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size, &output[0]);

	//Get the maximum value from the output of the max kernel
	max_val = output.front();

	//Print maximum to terminal
	printf("max = %f\n", max_val);

	//Return maximum value
	return max_val;
}

float find_min(cl::Context context, cl::CommandQueue queue, cl::Kernel min_kernel, vector<float> input) {
	vector<float> output(input.size());

	vector<float> input_two;

	//Sizes so memory can be allocated correctly
	size_t input_size = input.size() * sizeof(float);
	size_t output_size = output.size() * sizeof(float);
	size_t local_size = 10;

	//Initialise min value to 0 in order to hold the output, this also means that if anything fails the function will always return a value
	float min_val = 0;

	//Create the buffers to pass memory between host and kernel
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, output_size);

	//Write input into the input buffer
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, input.data());

	//Fill the output buffer with 0 values, to be filled with data by the kernel
	queue.enqueueFillBuffer(output_buffer, 0, 0, output_size);

	//Set the buffers as arguments
	min_kernel.setArg(0, input_buffer);
	min_kernel.setArg(1, output_buffer);
	//Allocate local memory and pass it to the kernel
	min_kernel.setArg(2, cl::Local(local_size * sizeof(float)));

	//Run the kernel
	queue.enqueueNDRangeKernel(min_kernel, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(local_size));

	//Read the kernel's output into the correct vector
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size, &output[0]);

	//Get the mimimum value from the output of the min kernel
	min_val = output.front();

	//Print minimum to terminal
	printf("min = %f\n\n\n", min_val);

	//Return minimum value
	return min_val;
}


float std_dev(cl::Context context, cl::CommandQueue queue, cl::Kernel var_kernel, vector<float> input, float mean) {
	vector<float> output(input.size());

	vector<float> input_two;

	//Sizes so memory can be allocated correctly
	size_t input_size = input.size() * sizeof(float);
	size_t output_size = output.size() * sizeof(float);
	size_t local_size = 10;

	//Initialise standard deviation variable to 0
	float std_deviation = 0;

	//Create the buffers to pass memory between host and kernel
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer output_buffer(context, CL_MEM_READ_WRITE, output_size);

	//Write input into the input buffer
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, input.data());

	//Fill the output buffer with 0 values, to be filled with data by the kernel
	queue.enqueueFillBuffer(output_buffer, 0, 0, output_size);

	//Set the buffers as arguments
	var_kernel.setArg(0, input_buffer);
	var_kernel.setArg(1, output_buffer);
	//Allocate local memory and pass it to the kernel
	var_kernel.setArg(2, cl::Local(local_size * sizeof(float)));
	//Pass scalar value needed to run the kernel
	var_kernel.setArg(3, sizeof(int), &mean);

	//Run the kernel
	queue.enqueueNDRangeKernel(var_kernel, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(local_size));

	//Read the kernel's output into the correct vector
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, output_size, &output[0]);

	//Do the final standard deviation calculation on the output of the variance kernel
	std_deviation = sqrt(output.front()/input.size());

	//Print the standard deviation to the terminal
	printf("std_deviation = %f\n\n\n", std_deviation);

	//Return the standard deviation
	return std_deviation;
}

void print_histogram(vector<int> histogram, int min_val, int max_val, int bin_size) {
	//Initialise bin id to 0
	int bin_id = 0;

	//Start the histogram output
	printf("\n\nHISTOGRAM\n--------------------------------------------------\n\n");

	//Loop through histogram
	for each (int bin in histogram)
	{
		//Print some info about the bin (formatted for correct width)
		printf("Bin %-3d contains %-7d values:", bin_id, bin);
		//Loop through the bin and print out a *, each additional * represents 2000 values
		for (int i = 0; i <= bin; i+=2000) {
			printf("*");
		}
		printf("\n");
		bin_id++;
	}
	//finish printing the histogram
	printf("\n--------------------------------------------------\n");
}

void histogram(cl::Context context, cl::CommandQueue queue, cl::Kernel histogram_kernel, vector<float> input, float min_val, float max_val, float std_dev) {
	
	int nr_of_bins = 1 + (3.322 * log10(input.size())); //Sturge's rule to find histogram bin size
	int bin_size = (max_val - min_val) / nr_of_bins; //Divide the range of values by the bin size to find number of bins

	//Print out the number of bins and the bin size
	printf("\nNumber of bins = %d\n", nr_of_bins);
	printf("bin_size = %d\n", bin_size);

	//Initialise histogram vector
	vector<int> hist(nr_of_bins);

	//Sizes so memory can be allocated correctly
	size_t input_size = input.size() * sizeof(float);
	size_t hist_size = hist.size() * sizeof(float);
	size_t local_size = 10;

	//Create the buffers to pass memory between host and kernel
	cl::Buffer input_buffer(context, CL_MEM_READ_WRITE, input_size);
	cl::Buffer hist_buffer(context, CL_MEM_READ_WRITE, hist_size);

	//Write input into the input buffer
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, input_size, input.data());

	//Fill the output buffer with 0 values, to be filled with data by the kernel
	queue.enqueueFillBuffer(hist_buffer, 0, 0, hist_size);

	//Set the buffers as arguments
	histogram_kernel.setArg(0, input_buffer);
	histogram_kernel.setArg(1, hist_buffer);
	//Allocate local memory and pass it to the kernel
	histogram_kernel.setArg(2, cl::Local(hist.size() * sizeof(float)));
	//Pass scalar values needed to run the kernel
	histogram_kernel.setArg(3, sizeof(int), &bin_size);
	histogram_kernel.setArg(4, sizeof(float), &min_val);
	histogram_kernel.setArg(5, sizeof(float), &max_val);

	//Run the kernel
	queue.enqueueNDRangeKernel(histogram_kernel, cl::NullRange, cl::NDRange(input.size()), cl::NDRange(local_size));
	
	//Read the kernel's output into the correct vector
	queue.enqueueReadBuffer(hist_buffer, CL_TRUE, 0, hist_size, &hist[0]);

	//Print the generated histogram to the terminal
	print_histogram(hist, min_val, max_val, bin_size);
}

int main(int argc, char **argv) {
	/*GET PLATFORMS*/
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		cout << "No platforms found!" << endl;
		exit(1);
	}
	cl::Platform default_platform;

	int num_platforms = platforms.size();

	cout << "Found: " << num_platforms << " platforms" << endl;

	cout << "Select a platform from the list below" << endl;

	//Print the platfomrs avaliable and allow the user to select which one they want to use
	for (int i = 0; i < num_platforms; i++)
	{
		cout << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << endl;
	}

	int selected_platform_number;

	cin >> selected_platform_number;

	default_platform = platforms[selected_platform_number];
	cout << "Using: " << default_platform.getInfo<CL_PLATFORM_NAME>() << endl;

	/* -------------------------------------------------- */

	/* GET DEVICES*/
	vector<cl::Device> devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size() == 0) {
		cerr << "Can't find any devices" << endl;
		exit(1);
	}

	int num_devices = devices.size();

	cout << "Found: " << num_devices << " devices" << endl;

	cout << "Select a device from the list below" << endl;

	//Print the devices avaliable and allow the user to select which one they want to use
	for (int i = 0; i < num_devices; i++)
	{
		cout << i << ": " << devices[i].getInfo<CL_DEVICE_NAME>() << endl;
	}

	int selected_device_number;

	cin >> selected_device_number;

	cl::Device default_device = devices[selected_device_number];
	cout << "Using: " << default_device.getInfo<CL_DEVICE_NAME>() << endl;

	/* -------------------------------------------------- */
	//Start timing of the program execution
	clock_t start_of_execution = clock();


	/*LOAD IN DATA*/

	/* Create vector to hold the file's temperature data*/

	vector<float> temperatures;
	
	//Creat FILE object to load in the data file
	FILE *data_file = fopen("temp_lincolnshire.txt", "r");

	if (!data_file) {
		cerr << "can't open the file";
		exit(1);
	}
	else {
		while (!feof(data_file)) {
			/*Get the data from the line, split so it can be placed into the correct vector*/

			//create the temporary temperature holder
			float temp;

			//Scanf saves about a minute of execution time over ifstream
			fscanf(data_file, "%*s %*s %*s %*s %*s %f", &temp); //Ignore everything but the last value as we're only interested in the temperature (save memory)
			
			//Add the temperature to the temperatures vector
			temperatures.push_back(temp);
		}
		fclose(data_file);
	}

	/* -------------------------------------------------- */

	/* Run the kernel functions to do the data processing */

	try {
		//GET CONTEXT FROM DEFAULT DEVICE
		cl::Context context{ { default_device } };


		//CREATE COMMAND QUEUE
		cl::CommandQueue command_queue(context);

		//Add kernel.cl as source
		ifstream kernelFile("kernel.cl");
		string kernelAsString(istreambuf_iterator<char>(kernelFile), (istreambuf_iterator<char>()));


		//Make the program
		cl::Program::Sources programSource(1, make_pair(kernelAsString.c_str(), kernelAsString.length() + 1));

		cl::Program program(context, programSource);

		//Build imported kernel to devices
		try {
			program.build(devices);
		}
		catch (const cl::Error& err) {
			//Catch any errors
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Get relevant sizes
		size_t sizeOfInput = temperatures.size() * sizeof(float);
		size_t local_size = 10;


		//Pad the vector to ensure it can be used with the defined local memory size
		size_t padding_size = temperatures.size() % local_size;
		if (padding_size) {
			//create an extra vector with 0 values
			std::vector<float> input_ext(local_size - padding_size, 0.0);
			//append that extra vector to our input
			temperatures.insert(temperatures.end(), input_ext.begin(), input_ext.end());
		}


		//Create queue and copy vectors to device memory
		cl::CommandQueue queue(context);
		
		/*Create the needed kernels*/
		cl::Kernel sum_kernel(program, "find_sum");
		cl::Kernel max_kernel(program, "find_max");
		cl::Kernel min_kernel(program, "find_min");
		cl::Kernel var_kernel(program, "variance");
		cl::Kernel histogram_kernel(program, "histogram");

		//run each function, passing it the context, queue, correct kernel and dataset
		float sum_val = sum(context, queue, sum_kernel, temperatures);
		float mean = sum_val / temperatures.size();

		float max_val = find_max(context, queue, max_kernel, temperatures);
		float min_val = find_min(context, queue, min_kernel, temperatures);

		float std_deviation = std_dev(context, queue, var_kernel, temperatures, mean);
		
		//Generate and print histogram 
		histogram(context, queue, histogram_kernel, temperatures, min_val, max_val, std_deviation);

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", Error code:" << err.err() << endl;
	}

	/* PROGRAM EXIT */
	//Print out the time
	printf("\nTime taken to execute code: %d clicks, %f seconds\n\n", clock() - start_of_execution, ((float)clock() - start_of_execution) / CLOCKS_PER_SEC);

	cout << "\n\nType 'q' and press enter to exit" << endl;

	//Wait for the qut command to be entered
	char wait_for_q;
	cin >> wait_for_q;

	while (wait_for_q != 'q') {
		//do nothing
	}

	return 0;
}