/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <iomanip>
#define FLT_MAX 10000
float *d_min, *d_max;
unsigned int *d_histo;

//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_min));
  checkCudaErrors(cudaFree(d_max));
  checkCudaErrors(cudaFree(d_histo));
}

__global__
void FindMin(const float* const d_logLuminance, float *d_fout, int numRows, int numCols)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];
  unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int tid  = threadIdx.x;
  
  
  // load shared mem from global mem
  if(myId >= numRows*numCols)
      return;
  
  sdata[tid] = d_logLuminance[myId];
  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      if(sdata[tid]>sdata[tid+s])
      {
      	sdata[tid] = sdata[tid+s];
      }
    }
    __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
    d_fout[blockIdx.x] = sdata[0];
  }
   
}

__global__
void FindMax(const float* const d_logLuminance, float *d_fout, int numRows, int numCols)
{
  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
  extern __shared__ float sdata[];
  unsigned int myId = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int tid  = threadIdx.x;
  
  // load shared mem from global mem
 if(myId >= numRows*numCols)
      return;
  
  sdata[tid] = d_logLuminance[myId];
  __syncthreads();            // make sure entire block is loaded!

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s && myId < numRows*numCols)
    {
      if(sdata[tid]<sdata[tid+s])
      {
      	sdata[tid] = sdata[tid+s];
      }
    }
    __syncthreads();        // make sure all adds at one stage are done!
  }

  // only thread 0 writes result for this block back to global mem
  if (tid == 0)
  {
    d_fout[blockIdx.x] = sdata[0];
  }
   
}

__global__
void GenerateHisto(const float* const d_logLuminance, unsigned int *d_histo, const float range, int numRows, int numCols, int numBins, float min_value)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(myId >= numRows*numCols)
		return;
	
    float myItem = d_logLuminance[myId];
    int tempBin = static_cast<int>((myItem - min_value) / range * numBins);
	int myBin = (tempBin <(numBins - 1))?tempBin:(numBins - 1);
	atomicAdd(&(d_histo[myBin]), 1);
}

__global__
void GeneratecCDF(unsigned int *d_histo, unsigned int *d_cdf, int numBins)
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
    
	if(myId >= numBins)
		return;

	for(size_t i=1;i<myId;i++)
	{
		//atomicAdd(&d_cdf[myId], d_histo[i]);
		d_cdf[myId] += d_histo[i];
	}
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	   
  unsigned int blockSize=1024;
  unsigned int gridSize=ceil((numCols*numRows+1024-1)/1024);
  
  //step 1, find the min value
  float *h_min;
  h_min = (float *)malloc(sizeof(float)*blockSize);
  checkCudaErrors(cudaMalloc((void **)&d_min,   sizeof(float) * blockSize));
  checkCudaErrors(cudaMemset(d_min, FLT_MAX, sizeof(float)*blockSize));
  FindMin<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_logLuminance, d_min, numRows, numCols);
  checkCudaErrors(cudaMemcpy(h_min, d_min, blockSize * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  min_logLum = h_min[0];
  for (size_t i = 0; i < blockSize; ++i) {
    //std::cout<<std::setprecision(4)<<h_min[i]<<std::endl;
    min_logLum = std::min(h_min[i], min_logLum);
  }

  //step 1, find the max value
  float *h_max;
  h_max = (float *)malloc(sizeof(float)*blockSize);
  checkCudaErrors(cudaMalloc((void **)&d_max,   sizeof(float) * blockSize));
  checkCudaErrors(cudaMemset(d_max, -FLT_MAX, sizeof(float)*blockSize));
  FindMax<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_logLuminance, d_max, numRows, numCols);
  checkCudaErrors(cudaMemcpy(h_max, d_max, blockSize * sizeof(float), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  max_logLum = h_max[0];
  for (size_t i = 0; i < blockSize; ++i) {
    //std::cout<<std::setprecision(4)<<h_max[i]<<std::endl;
    max_logLum = std::max(h_max[i], max_logLum);
  }
  
  //step 2, find the range
  float range = max_logLum - min_logLum;
  std::cout<<"max:"<<std::setprecision(4)<<max_logLum<<",min:"<<std::setprecision(4)<<min_logLum<<",range:"<<std::setprecision(4)<<range<<std::endl;
  
  //step 3, generate histogram
  checkCudaErrors(cudaMalloc((void **)&d_histo,   sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int)*numBins));
  GenerateHisto<<<gridSize,blockSize>>>(d_logLuminance,d_histo,range,numRows,numCols,numBins,min_logLum);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  //step 4,get cumulative distribution of luminance
  //int *temp_hist;
  //temp_hist = (int *)malloc(sizeof(int)*numBins);
  GeneratecCDF<<<1,blockSize>>>(d_histo,d_cdf,numBins);
  //checkCudaErrors(cudaMemcpy(temp_hist, d_cdf, sizeof(int)*numBins, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  /*for(size_t i=0;i<numBins;i++)
  {
	std::cout<<"origin:"<<temp_hist[i]<<std::endl;
  }*/
  free(h_min);
  free(h_max);
  cleanup();
}


