#include <torch/script.h> // One-stop header.

#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>
#include <cmath> //is this old way for pow?
using namespace std::chrono; 

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: test_model <path-to-exported-script-module> (optional)<batch_size\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "loaded\n testing with ones . . .\n";
  
  int batch_size = 1;
  /*
  if (argc>2) {
    std::stringstream convert{ argv[2] };
    convert >> batch_size; //no error checking yet
    //std::cout << "batch_size: " << batch_size << '\n';
  }
  */

  //**Now run several batch sizes (powers of 2), and 
  //  for each batch size run the forward model several
  //  times, in order to get timing statistics
  float duration = 0.0;
  float mean = 0.0; //mean time to run
  float stdev = 0.0; //standard deviation of time to run
  int N = 100;
  int Nskip = 5;
  for (int i=0; i<15; i++) {
      batch_size = pow(2,i); //2^i
      // Create a vector of inputs.
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(torch::ones({batch_size, 2, 32, 32}));
      
      duration = 0.0;
      mean = 0.0;
      stdev = 0.0;
      // measure 100 times for statistics
      for (int j=0; j<(N+Nskip); j++) {
          // Execute the model and turn its output into a tensor.
          auto start = high_resolution_clock::now();
          at::Tensor output = module.forward(inputs).toTensor();
          auto stop = high_resolution_clock::now();
          duration = duration_cast<milliseconds>(stop - start).count();
          if (j>=Nskip) { //skip some, since there might be a startup cost
              mean += duration;
              stdev += pow(duration,2);
          }
      }
      mean = mean/N;
      stdev = stdev/N - pow(mean,2);
      std::cout << "batch_size: " << batch_size << ", Average Time: " << mean << " ms "<< ", Average Time/example: " << mean/batch_size << " ms/example" << ", Std: " << stdev << std::endl;
  }
  //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
