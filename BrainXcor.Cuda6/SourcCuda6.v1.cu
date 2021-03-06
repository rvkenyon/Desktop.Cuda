
#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>
#include <cmath>
//#include "stdio.h"
#include <stdio.h>
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
//#include <helper_functions.h>  

#define THREADS 512
//#define Gy 32 //grid y dimension
//#define Gx 1920 //grid x dimension
//#define Xt 1 //thread x dimension
//#define Yt 64 //thread y dimension
//#define Fx 2000 //number of frames
typedef struct{
	int x;
	int y;
} twoD;

typedef struct{
	int x;
	int y;
	int z;
} threeD
	typedef struct{
		unsigned int win;
		float sDev;
} pixelLoc;

typedef struct{
	unsigned int loc_Wind1;
	unsigned int loc_Wind2;
	float loc_corrCoef;
} PixelxCor;

//const int Gy = 32; //grid y dimension
//const int Gx = 4; //grid x dimension
const int Xt = 64; //thread x dimension
//int const Gx = 1920; //grid x dimension
//int const Xt = 1; //thread x dimension
const int Yt = 4; //32//thread y dimension
const int Fx = 2000; //number of frames
const int h_Wsize = 50;


using namespace std;

//this where each thread takes a different window to Xcorr
//__global__ void XcrossCUDA(int* d_Pixels, pixelLoc* d_PL, float* d_Cor, int N, int corCount, int Wsize)
//{
//	int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
//	//	int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
//
//	float x1, x2, SumPt2, Sum_X1X2, window[120],sdev1,sdev2;
//	int winStart, window1,window2, Index,temp2,temp3; //change yIdx and xIdx
//	//	float xcorrValues[16000];
//
//	// find local point only for xcorr with window
//	if(xIdx < N-1 && d_PL[xIdx].win > 0)//|| window1 > 0) //not at end of file
//	{
//		winStart = xIdx;
//		temp3 = xIdx; //index from "points to correlate" array d_PL
//		window1 = d_PL[winStart].win;
//		//load window for xcorr
//		sdev1 = d_PL[winStart].sDev;
//
//		//temp2 = ((temp3) * (temp3-1))/2;
//		//Index = (N * temp3) - temp2; //this needs to be checked
//		//calculate the offset to write the output data file [N * I - (N(N-1)/2)]
//		//if(xIdx == 0)
//		//	Index = 0;
//		//else 
//		//	{
//		//	temp2 = ;
//		Index = corCount - ((N-xIdx) * (N-xIdx - 1))/2; //this needs to be checked
//		//	}
//
//		//get pixel values for correlation's Master window
//		for(int n = 0; n < Wsize; n++) 
//			window[n] = d_Pixels[window1 + n];
//
//		//now get windows for other points in the correlation
//		for(int i = 0; i < N - xIdx; i++)
//		{
//			window2 = d_PL[winStart+i].win;
//			sdev2 = d_PL[winStart+i].sDev;
//			//			if(sdev2 == 0 || sdev1 == 0) 
//			//			continue;
//			//if(window2 < 0)
//			//	break;
//			//find data start point for windows in silo
//
//			//if point is valid then begin correlations
//			x1 = x2 = Sum_X1X2 = 0.;
//
//			// do the actual cross correlation now
//			for (int l = 0; l < Wsize; l++)
//			{
//				SumPt2 = d_Pixels[window2 + l];
//				x1 += window[l];
//				x2 += SumPt2;
//				Sum_X1X2 += window[l] * SumPt2;
//			}
//			d_Cor[i + Index] = ((Sum_X1X2 - x1 * x2/Wsize)/(Wsize - 1)/sdev2/sdev1);	
//		} //end of correlation calculation
//	} // end of finding windows inside pixel silo
//}

static void HandleError( cudaError_t err, const char *file, int line ) {
	if (err != cudaSuccess) 
	{
		cerr<<cudaGetErrorString( err )<<" in "<<file<<" at line "<<line<<endl;
		//		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}


__global__ void print_kernel() {
	printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}



//this where thread acts on the same window to Xcorr
__global__ void XcrossCUDA_same(int* Pixels, pixelLoc* PL, PixelxCor* Cor, int X, int corCount, int Wsize)
{
	extern __shared__ int window[];
	//here d_Cor is on Host not Device
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x;

	float x1, x2, SumPt2, Sum_X1X2,sdev1,sdev2;
	unsigned long int winStart, window1,window2, Index, j = xIdx; //change yIdx and xIdx

	// find local point only for xcorr with window
	if(xIdx < X-1)
	{
		for(int i = 0; xIdx < X-1 - i;j = xIdx, i++) //increment through all PL data points
		{
			__syncthreads();  //need this so wndow not changed while still in use.
			Index = corCount - ((X-i) * (X-i - 1))/2; //this needs to be checked
			winStart = i; //index of the window
			window1 = PL[winStart].win;
			sdev1 = PL[winStart].sDev;
			//get pixel values for correlation's Master window
			//MUST use threadIdx not xIdx window must exists in each block.
			if(threadIdx.x == 0)
			{
				for(int ii = 0; ii < Wsize; ii++)
				{
					window[ii] = Pixels[window1 + ii]; // check this...
				}
			}
			__syncthreads();

			//roll through all the data for this window
			while(j < X-1-i)
			{
				window2 = PL[winStart+j].win;
				sdev2 = PL[winStart+j].sDev;

				//if point is valid then begin correlations
				x1 = x2 = Sum_X1X2 = 0.;

				// do the actual cross correlation now
				for (int l = 0; l < Wsize; l++)
				{
					SumPt2 = Pixels[window2 + l];
					x1 += window[l];
					x2 += SumPt2;
					Sum_X1X2 += window[l] * SumPt2;
				}
				//if(x14 > 1)
				//	x14 = x14;
				//				((Sum_X1X2 - x1 * x2/Wsize)/(Wsize - 1)/sdev1/sdev2);	
				Cor[j + Index].loc_corrCoef = ((Sum_X1X2 - x1 * x2/Wsize)/(Wsize - 1)/sdev1/sdev2);
				Cor[j + Index].loc_Wind1 = window1;	
				Cor[j + Index].loc_Wind2 = window2;	

				j += gridDim.x * blockDim.x;
			}
		}
	}
}


__global__ void StdDev(int* Pixels, pixelLoc* PL,  int Wsize, int frames,  int outTotal, twoD numProcThds, int devThres)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

	if(xIdx < numProcThds.x && yIdx < numProcThds.y)
	{
		float temp, x1=0.f, x2=0.f;
		unsigned int xyStart; //where to start reading the window
		unsigned int outStart;  //output file index not to exceed outTotal
		xyStart = xIdx + frames * yIdx;
		outStart = xIdx + numProcThds.x * yIdx;
		if(outStart > outTotal)
			printf("exceed array size: Index=%d;X=%d; Y=%d\n",outStart,xIdx,yIdx);

		for(int i = 0; i < Wsize; i++)
		{
			temp = (float)Pixels[xyStart + i];
			x1 += temp;
			x2 += temp * temp; 
		}
		temp = sqrtf((x2 - x1*x1/Wsize)/(Wsize-1));
		PL[outStart].win = xyStart;
		if(temp > devThres)
			PL[outStart].sDev = temp;
		else
			PL[outStart].sDev = 0.0f;
	}
}

__global__ void transposeNaive(int *odata, int* idata, int width, int height, twoD numProcThds)
{
	int xIndex = blockIdx.x*blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y*blockDim.y + threadIdx.y;

	if(xIndex < numProcThds.x && yIndex < numProcThds.y)
	{
		int index_in = xIndex + width * yIndex;
		int index_out = yIndex + height * xIndex;

		odata[index_out] = idata[index_in];
	}
}

__global__ void CorCuda(int Pixels, pixelLoc PL, PixelxCor Cor, int Wsize, threeD processors)
{
	unsigned int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIdx = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int zIdx = blockIdx.z * blockDim.z + threadIdx.z;

	//check processor in images array
	if(xIdx < processors.x && yIdx < processors.y && zIdx < processors.z)
	{
		//check if stdev is above threshold to do correlation
		if(PL[xIdx*yIdx][zIdx].sDev > 0)
		{
			//store current point window data
		for(int ii = 0; ii < Wsize; ii++)
				window[ii] = Pixels[xIdx*yIdx][zIdx + ii]; // check this...
		//now correlate with all subsequent points in the frame
			for(int i = 0; i + yIdx < processors.y; i++)
			{
				for(int j = 0; xIdx + j < processors.x; j++)
					window2[i] = Pixels[xIdx+j*yIdx][zIdx];

			}

			for(int i=0; i < xxxx; i++)
			{
				// do the actual cross correlation now
				for (int l = 0; l < Wsize; l++)
				{
					SumPt2 = Pixels[window2 + l];
					x1 += window[l];
					x2 += SumPt2;
					Sum_X1X2 += window[l] * SumPt2;
				}
				window[ii] = Pixels[xIdx][yIdx][zIdx + ii]; // check this...
			}
		}

	}
}


int main()
{
	int frames = Fx;

	const int imageX = 172;				//size of Image used ... columns
	const int imageY = 130;				//size of Image used ... rows
	const int totalPixs = imageX * imageY; //total pixel number for image
	const int readSize = frames * totalPixs; //total memory size of all data
	int Xloc1, Yloc1, Floc1;			//used for X,Y,Frame for Point 1
	int Xloc2, Yloc2, Floc2;			//used for X,Y,Frame for Point 2
	int deviceCount;
	int thredMax;
	int devThres = 35;
	int N;
	//	int size_file=0;
	int dev = 0;
	cudaError_t  code;

	twoD numProcThds;
	numProcThds.x = frames - h_Wsize;	//used in Stdev kernel for total number threads X direction
	numProcThds.y = totalPixs;			//used in Stdev kernel for total number threads Y direction
	int procsrTot = numProcThds.x*numProcThds.y; //size of output file elements for STDv

	int *Pixels;						//used to hold pixel values
	pixelLoc *PL;						//used to hold Stdev values
	PixelxCor *Cor;						//used to hold Corr values

	//this MUST be here; flags must be set before any
	//Cuda calls made; if Host Memory use by Device is used!!
	//cudaSetDeviceFlags(cudaDeviceMapHost);

	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) 
	{
		fprintf(stderr, "error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}
	cudaSetDevice(dev);
	cudaDeviceProp devProps;
	if (cudaGetDeviceProperties(&devProps, dev) == 0)
	{
		printf("Using device %d:\n", dev);
		printf("%s; global mem: %uB; compute v%d.%d; clock: %d kHz\n",
			devProps.name, (long)devProps.totalGlobalMem, 
			(int)devProps.major, (int)devProps.minor, 
			(int)devProps.clockRate);
	}

	//	print_kernel<<<10, 10>>>();
	//    cudaDeviceSynchronize();

	//const int gridLimit = devProps.maxGridSize[0];
	//thredMax = devProps.maxThreadsPerBlock; //devProps.maxThreadsPerBlock;

	const dim3 blockSize(Xt, Yt, 1);  //TODO
	const dim3 gridSize((frames/Xt)+1,(totalPixs/Yt)+1, 1);  //TODO
	//int Tot_NumThreads;
	//int BlockWidth;
	//first = Tot_NumThreads/BlockWidth;
	//second = BlockWidth;//threads per block

	//using new Unified Memory in Cuda 6.0 with compute 3.0
	HANDLE_ERROR(cudaMallocManaged(&Pixels, sizeof(int) * readSize)); 
	HANDLE_ERROR(cudaMallocManaged(&PL, sizeof(pixelLoc) * procsrTot));


	std::ifstream fin("d:/data/file_.bin", std::ios::binary);
	fin.read(reinterpret_cast<char*>(Pixels), sizeof(int) * readSize);
	fin.close();

	//FILE* file;
	//file = fopen("d:/data/file_name50.txt", "r");
	//if(file == 0)
	//{
	//	printf("bad file name\n");
	//	exit(0);
	//}
	//float temp;
	////	while (!feof (file))
	//for(int i = 0; i < readSize; i++)
	//{  
	//	fscanf(file, "%E", &temp);
	//	h_Pixels[i] = int(temp);
	//	size_file++;
	//}

	cout<<"Prior to addition: "<<endl;
	for(int i = 0; i < 10; i++)
		cout<<Pixels[i]<<endl;

	//allocate memory space and copy data to device
	//HANDLE_ERROR(cudaMalloc((void**) &d_Pixels, sizeof(int) * readSize));
	//HANDLE_ERROR(cudaMalloc((void**) &d_PL, sizeof(pixelLoc) * readSize));
	//HANDLE_ERROR(cudaMemcpy((void*) d_Pixels, h_Pixels, sizeof(int) * readSize, cudaMemcpyHostToDevice));
	//	HANDLE_ERROR(cudaMemset((void*) d_PL, 0, sizeof(pixelLoc) * readSize));

	//run kernel for finding Standard Deviation of data
	StdDev<<<gridSize, blockSize>>>(Pixels, PL, h_Wsize, frames, procsrTot, numProcThds, devThres);

	//wait for all to finish and copy data to host
	cudaDeviceSynchronize(); 
	HANDLE_ERROR(cudaGetLastError());

	//compress list of points, removing points below threshold 
	int compPLtot = 0;
	for(int i = 0; i < readSize; i++)
	{
		//if(h_PL[i].sDev < 1 && h_PL[i].win != -1)
		//	{
		//	cout<<"std = "<<h_PL[i].sDev<<"   "<<h_PL[i].win<<endl;
		//	}
		if(PL[i].sDev > 0)
			PL[compPLtot++] = PL[i];
	}

	N = compPLtot;
	//cudaFree(d_PL);
	//cudaFree(d_Pixels);
	//HANDLE_ERROR(cudaMalloc((void**) &d_PL, sizeof(pixelLoc) * N));
	//HANDLE_ERROR(cudaMalloc((void**) &d_Pixels, sizeof(int) * readSize));

	int *pixTrans = new int [50];
	twoD numPixElmts = {frames, totalPixs};
	const dim3 blockSize2(Xt, Yt, 1);  //TODO
	const dim3 gridSize2((imageX/Xt)+1,(imageY/Yt)+1, 1);  //TODO
	//int Tot_NumThreads;
	//int BlockWidth;
	//first = Tot_NumThreads/BlockWidth;
	//second = BlockWidth;//threads per block

	transposeNaive<<<gridSize2, blockSize2>>>(Pixels, pixTrans, frames, totalPixs, numPixElmts);

	const int N1 = N +1;
	unsigned int const corSize = N1*(N1-1)/2;
	//unsigned int trmp = corSize*sizeof(PixelxCor);
	//double tp = pow(2.,32.);
	//if(trmp > tp)
	//	fprintf(stderr, "h_Cor is larger than 4GB!!\n");

	//do the regular stuff for passing arrays to Kernel
	//HANDLE_ERROR(cudaMemcpy((void*) d_Pixels, h_Pixels, sizeof(int) * readSize, cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaMemcpy((void*) d_PL, h_PL, sizeof(pixelLoc) * N, cudaMemcpyHostToDevice));

	//use memory on Host for Kernel not Device due to Size of Array
	//HANDLE_ERROR(cudaHostAlloc((void**)&h_Cor, sizeof(PixelxCor) * corSize, cudaHostAllocMapped));
	HANDLE_ERROR(cudaMallocManaged((void**)&Cor, sizeof(PixelxCor) * corSize));

	//get the address for Kernel write to output array
	//HANDLE_ERROR(cudaHostGetDevicePointer(&d_Cor, h_Cor, 0));

	//int *Indexing = new int[300000];
	//for(int idx = 0; idx < N; idx++)
	//	Indexing[idx] = corSize - ((N1-idx) * (N1-idx - 1))/2; //this needs to be checked

	//now do xcorrelation
	const dim3 blockSize3(Xt, Yt, 1);  //TODO
	const dim3 gridSize3((imageX/Xt)+1,(imageY/Yt)+1, frames);  //TODO
	//int Tot_NumThreads;
	//int BlockWidth;
	//first = Tot_NumThreads/BlockWidth;
	//second = BlockWidth;//threads per block
	threeD processors = {imageX, imageY, frames};

	CorCuda(Pixels,PL,Cor,h_Wsize,processors);

	//transpose pixel array


	thredMax /= 2;
	int  blocks = (N+thredMax-1)/thredMax;
	//if(blocks > gridLimit) blocks = gridLimit;

	XcrossCUDA_same<<<blocks, thredMax, h_Wsize * sizeof(int)>>>(Pixels, PL,  Cor, N1, corSize, h_Wsize);

	cudaDeviceSynchronize(); 
	HANDLE_ERROR(cudaGetLastError());

	//	HANDLE_ERROR(cudaMemcpy(h_Cor, d_Cor, sizeof(float) * corSize, cudaMemcpyDeviceToHost));

	//	int ja = 0;
	//	float *pp = new float[300000];
	//	for(int i = 0, temp = 0; i < corSize; i++)
	//	{
	//		temp = 1000.0*h_Cor[i];
	//		if(abs((int) temp) > 998)
	//		{
	//			pp[ja++] = i;
	////			ja++;
	//		}
	//	}
	//	cout<<"After addition:"<<ja<<endl;
	////create file for Lovain analysis: P1 X Pn....; P2 X Pn-1....
	//for(int i = 0; i < corSize; i++)
	//{
	//	h_final[i].addrss = h_PL[i].win;
	//	h_final[i].Xaddrss = h_PL[i + j].win;
	//	h_final[i].XcorVal = h_Cor[j];
	//}
	//	cout<<ja<<endl;

	//for(int i = 0; i < corSize; i++)
	//	if(int(100 * h_Cor[i].loc_corrCoef) < 1)
	//		continue;
	//write out the data to a file

	FILE *fpw;
	char filew[512];
	sprintf(filew,"d:/data/%s.pair.txt","cor_weights");
	if ((fpw = fopen(filew,"w"))==NULL)
	{
		printf("cannot open file\n");
	}
	//		printf("\tPt1\t\t\t\Pt2\t\tXcorr\nX\tY\tF\tX\tY\tF\t\n");
	//		fprintf(fpw, "\tPt1\t\t\t\tPt2\t\tXcorr\nX\tY\tF\tX\tY\tF\t\n");
	//		printf("\tPt\#1\tFrm#\t\t\Pt\#2\t\Frm#\tXcorr\n");
	//	fprintf(fpw, "\tPt\#1\tFrm#\t\t\Pt\#2\t\Frm#\tXcorr\n");
	//	fprintf(fpw, "Pt#1\tFrm#\t\Pt#2\t\Frm#\tXcorr\n");
	fprintf(fpw, "Frm#\tPt#1\tPt#2\tXcorr\n");
	for(int i = 0; i < corSize; i++)
	{
		Floc1 = Cor[i].loc_Wind1 % frames;
		Floc2 = Cor[i].loc_Wind2 % frames;
		Yloc1 = (Cor[i].loc_Wind1-Floc1)/frames;
		Yloc2 = (Cor[i].loc_Wind2-Floc2)/frames;
		//Xloc1 = Yloc1%imageX;
		//Yloc1 = (Yloc1 - Xloc1)/imageX;
		//Xloc2 = Yloc2%imageX;
		//Yloc2 = (Yloc2 - Xloc2)/imageX;
		//Xloc1 += 1;
		//Xloc2 += 1;
		Yloc1 += 1;
		Yloc2 += 1;
		//if (~Yloc1)
		//	Yloc1=imageX;
		//Xloc2 = floor((h_Cor[i].loc_Wind2-Floc2)/imageY);
		//Yloc2 = (h_Cor[i].loc_Wind2-Floc2) - (Xloc2*imageY);
		if(Floc1 == Floc2)
			fprintf(fpw, "%d\t%d\t%d\t%f\n",Floc1, Yloc1, Yloc2,  Cor[i].loc_corrCoef);
		//		fprintf(fpw, "%d\t%d\t%d\t%d\t%f\n",Yloc1, Floc1,Yloc2, Floc2, h_Cor[i].loc_corrCoef);
		//if (~Yloc2)
		//	Yloc2=imageX;
		//		fprintf(fpw, "%d\t%d\t%d\t%d\t%f\n",Yloc1, Floc1,Yloc2, Floc2, h_Cor[i].loc_corrCoef);
		//		printf("%d\t%d\t%d\t%d\t%f\n",Yloc1, Floc1,Yloc2, Floc2, h_Cor[i].loc_corrCoef);
		//		fprintf(fpw, "%d\t%d\t%d\t%d\t%d\t%d\t%f\n",Xloc1, Yloc1, Floc1, Xloc2, Yloc2, Floc2, h_Cor[i].loc_corrCoef);
		//		fprintf(fpw, "Pt1(x,y,f) = %d,%d,%d Pt2(x,y,f) = %d,%d,%d Xcorr = %f\n",Xloc1, Yloc1, Floc1, Xloc2, Yloc2, Floc2, h_Cor[i].loc_corrCoef);
	}

	fclose(fpw);
	cudaFree(Cor);
	cudaFree(PL);
	cudaFree(Pixels);

	return 0;
}
