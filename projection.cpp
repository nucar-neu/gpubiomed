#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "projection.h"

#define MAX_ERR_VAL 64

#define FP 1
#define BP 2

cl_program clProgram;
cl_context clGPUContext;
cl_command_queue clCommandQueue;
cl_kernel forwardProjection;
cl_kernel backwardProjection;

int workSize_x;
int workSize_y;

cl_ulong timerStart, timerEnd;
int numFP = 0;
int numBP = 0;
double cumTimeFP = 0.0;
double cumTimeBP = 0.0;
double cumTime = 0;

cl_mem d_det;
cl_mem d_yrayTmp;
cl_mem d_roi;

cl_mem d_delta_det_col;
cl_mem d_delta_z;
cl_mem d_debug;

void cl_timingInfo() {
  printf("Total FP time: %f   Num of FP: %d   Ave. FP time: %f\n", cumTimeFP, numFP, cumTimeFP/(float)numFP);
  printf("Total BP time: %f   Num of BP: %d   Ave. BP time: %f\n", cumTimeBP, numBP, cumTimeBP/(float)numBP);
  printf("Total cumulative time of functions: %f\n", cumTime);
}

char *cl_errs[MAX_ERR_VAL] = {
   "CL_SUCCESS",                       //0                            
   "CL_DEVICE_NOT_FOUND",              //-1                         
   "CL_DEVICE_NOT_AVAILABLE",          //-2                    
   "CL_COMPILER_NOT_AVAILABLE",        //-3                 
   "CL_MEM_OBJECT_ALLOCATION_FAILURE", //-4            
   "CL_OUT_OF_RESOURCES",              //-5                         
   "CL_OUT_OF_HOST_MEMORY",            //-6                      
   "CL_PROFILING_INFO_NOT_AVAILABLE",  //-7            
   "CL_MEM_COPY_OVERLAP",              //-8                        
   "CL_IMAGE_FORMAT_MISMATCH",         //-9                   
   "CL_IMAGE_FORMAT_NOT_SUPPORTED",    //-10
   "CL_BUILD_PROGRAM_FAILURE",         //-11           
   "CL_MAP_FAILURE",                   //-12
   "",               //-13
   "",               //-14
   "",               //-15
   "",               //-16
   "",               //-17
   "",               //-18
   "",               //-19
   "",               //-20
   "",               //-21
   "",               //-22
   "",               //-23
   "",               //-24
   "",               //-25
   "",               //-26
   "",               //-27
   "",               //-28
   "",               //-29
   "CL_INVALID_VALUE",
   "CL_INVALID_DEVICE_TYPE",
   "CL_INVALID_PLATFORM",
   "CL_INVALID_DEVICE",
   "CL_INVALID_CONTEXT",
   "CL_INVALID_QUEUE_PROPERTIES",
   "CL_INVALID_COMMAND_QUEUE",
   "CL_INVALID_HOST_PTR",
   "CL_INVALID_MEM_OBJECT",
   "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
   "CL_INVALID_IMAGE_SIZE",
   "CL_INVALID_SAMPLER",
   "CL_INVALID_BINARY",
   "CL_INVALID_BUILD_OPTIONS",
   "CL_INVALID_PROGRAM",
   "CL_INVALID_PROGRAM_EXECUTABLE",
   "CL_INVALID_KERNEL_NAME",
   "CL_INVALID_KERNEL_DEFINITION",
   "CL_INVALID_KERNEL",
   "CL_INVALID_ARG_INDEX",
   "CL_INVALID_ARG_VALUE",
   "CL_INVALID_ARG_SIZE",
   "CL_INVALID_KERNEL_ARGS",
   "CL_INVALID_WORK_DIMENSION ",
   "CL_INVALID_WORK_GROUP_SIZE",
   "CL_INVALID_WORK_ITEM_SIZE",
   "CL_INVALID_GLOBAL_OFFSET",
   "CL_INVALID_EVENT_WAIT_LIST",
   "CL_INVALID_EVENT",
   "CL_INVALID_OPERATION",
   "CL_INVALID_GL_OBJECT",
   "CL_INVALID_BUFFER_SIZE",
   "CL_INVALID_MIP_LEVEL",
   "CL_INVALID_GLOBAL_WORK_SIZE"};

int cl_errChk(const cl_int status, const char * msg) 
{
   if(status != CL_SUCCESS) {
      printf("OpenCL Error: %d %s %s\n", status, cl_errs[-status], msg);
      return 1;
   }
   return 0;
}


//! \brief GPU initialization
//! GPU initialization fuction
//! It should be called before any other APIs
void GPU_init() 
{
  printf("*** GPU_init()\n");
  cl_int status;          
  FILE *fp;
  char *source;
  long int size;
  size_t szParmDataBytes;
  int defplat = 0;
  int defdev = 0;

  cl_uint numPlatforms;
  cl_uint numDevices;

  cl_platform_id platform = NULL;
  cl_device_id device = NULL;

  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  printf("Number of platforms detected:%d\n", numPlatforms);

  if (numPlatforms > 0) {
    cl_platform_id *platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    for(unsigned int i = 0; i < numPlatforms ; i++) {
      char pbuf[100];
      printf("Platform %d:\n", i);
      status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(pbuf), pbuf, NULL);
      printf("\tVendor: %s\n", pbuf);
      status = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(pbuf), pbuf, NULL);
      printf("\t  Name: %s\n", pbuf);  
    }
    printf("Using platform %d\n", defplat);
    platform = platforms[defplat];
  }
  else {
    printf("No OpenCL platforms found\n");
    exit(0);
  }

  cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
  cl_context_properties *cprops = cps;

  // Context
  clGPUContext = clCreateContextFromType(cprops, CL_DEVICE_TYPE_ALL, NULL, NULL, &status);
  if(cl_errChk(status, "creating context")) {
    exit(1);
  }

  status = clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
  cl_device_id *devices = (cl_device_id*)malloc(szParmDataBytes);
  status |= clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, devices, NULL);
  if(cl_errChk(status, "getting context info")) {
    exit(1);
  }

  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
  if(cl_errChk(status, "checking for devices")) {
    exit(1);
  }

  if(numDevices == 0) {
    printf("There are no devices for Platform 0\n");
    exit(0);
  }

  devices = (cl_device_id*)malloc(sizeof(cl_device_id)*numDevices);
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
  if(cl_errChk(status, "getting device IDs")) {
    exit(1);
  }

  printf("Number of devices for Platform 0 is %u\n", numDevices);

  for(unsigned int i = 0; i < numDevices; i++) {
    char dbuf[100];
    printf("Device %d:\n", i);
    status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(dbuf), dbuf, NULL);
    printf("\tVendor: %s\n", dbuf);
    status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(dbuf), dbuf, NULL);
    printf("\t  Name: %s\n\n", dbuf);
  }

  printf("Using device %d\n", defdev);
  device = devices[defdev];
 
  // Command queue
  clCommandQueue = clCreateCommandQueue(clGPUContext, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if(cl_errChk(status, "creating command queue")) {
    exit(1);
  }

  fp  = fopen("projection.cl", "r");
  if(fp == NULL) {
    printf("Could not open kernel file\n");
    exit(-1);
  }
  status = fseek(fp, 0, SEEK_END);
  if(status != 0) {
    printf("Error seeking to end of file\n");
    exit(-1);
  }
  size = ftell(fp);
  if(size < 0) {
    printf("Error getting file position\n");
    exit(-1);
  }
  status = fseek(fp, 0, SEEK_SET);
  if(status != 0) {
    printf("Error seeking to start of file\n");
    exit(-1);
  }
  source = (char *)malloc(size + 1);
  if(source == NULL) {
    printf("Error allocating space for the kernel source\n");
    exit(-1);
  }

  fread(source, size, 1, fp);
  source[size] = '\0';

  clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source, NULL, &status);
  if(cl_errChk(status, "creating program")) {
    exit(1);
  }

  free(source);
  fclose(fp);

  status = clBuildProgram(clProgram, 0, NULL, "-cl-fast-relaxed-math -cl-nv-verbose", NULL, NULL); 

  cl_build_status build_status;
  clGetProgramBuildInfo(clProgram, device, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
   
  if(build_status == CL_SUCCESS) {
    printf("* clBuildProgram succeeded!\n");
  }
  else {
    printf("* clBuildProgram failed!\n");
    //exit(1);
  }
   
  char *build_log;
  size_t ret_val_size;
  clGetProgramBuildInfo(clProgram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
  build_log = (char *) malloc(ret_val_size+1);
  clGetProgramBuildInfo(clProgram, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);    
  build_log[ret_val_size] = '\0';
  printf("%s\n", build_log);

  forwardProjection = clCreateKernel(clProgram, "forwardProjection", &status);
  if(cl_errChk(status, "creating gpu kernel")) {
    exit(1);
  }
			
  backwardProjection = clCreateKernel(clProgram, "backwardProjection", &status);
  if(cl_errChk(status, "creating gpu kernel")) {
    exit(1);
  }
  printf("Done initting\n");
}


//! \brief GPU cleaning up
//! GPU cleaning up fuction
//! It should be called at the end of the program
void  GPU_cleanup() 
{
  printf("*** GPU_cleanup()\n");
  if(forwardProjection) 
    clReleaseKernel(forwardProjection);
  if(backwardProjection)
    clReleaseKernel(backwardProjection);
  if(clProgram)
    clReleaseProgram(clProgram);
  if(clCommandQueue)
    clReleaseCommandQueue(clCommandQueue);
  if(clGPUContext)
    clReleaseContext(clGPUContext);
}

struct timeval startTimersCpu[8];
struct timeval stopTimersCpu[8];

double calctime(struct timeval tv_start, struct timeval tv_end)
{
   double start, end, total;
   end = tv_end.tv_sec*1000.0 + ((double)tv_end.tv_usec)/((double)1000);
   start = tv_start.tv_sec*1000.0 + ((double)tv_start.tv_usec)/((double)1000);
   total = end - start;

   return total;
}

void cl_startTimer(cl_event event) {

   cl_int status;
   status = clWaitForEvents(1, &event);
   if(cl_errChk(status, "waiting for events")) {
      exit(1);
   }
   status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &timerStart, NULL);
   if(cl_errChk(status, "getting start profile info")) {
      exit(1);
   }
}

void cl_stopTimer(cl_event event, int FPorBP) {

   float time;
   cl_int status;

   status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &timerEnd, NULL);
   if(cl_errChk(status, "getting stop profile info")) {
      exit(1);
   }

   time = (timerEnd-timerStart)*1.0e-6f;
   //printf("GPU timer: %.3f ms\n", time);

   if(FPorBP == FP)
     cumTimeFP += time;
   else
     cumTimeBP += time;

   cumTime += time;
}

void cl_startTimerCpu(int timer) 
{
   clFinish(clCommandQueue);
   gettimeofday(&startTimersCpu[timer], NULL);
}

void cl_stopTimerCpu(int timer) 
{
   double time;
   clFinish(clCommandQueue);
   gettimeofday(&stopTimersCpu[timer], NULL);
   time = calctime(startTimersCpu[timer], stopTimersCpu[timer]);   
   printf("CPU timer %d: %.2f ms\n", timer, time);
}

//! \brief GPU forward projection initialization
//! GPU initialization fuction
//! This function should be called after GPU_init() and before GPU_forwardProjection()
//! @param NUM_PROJ           Total projection count
//! @param NUM_DET_COL        Width of detector
//! @param NUM_DET_ROW        Height of detector
//! @param NUM_ROI_X          ROI size (x axis)
//! @param NUM_ROI_Y          ROI size (y axis)
//! @param NUM_ROI_Z          ROI size (z axis)
//! @param det                3-dimentional array that contains detector values
//! @param roi                3-dimentional array that contains ROI values
//! @param delta_det_col      Relative angles of detector cell with respect to xy coordiate
//! @param roi_xplane_first   Coordinate of the first X plane of ROI
//! @param roi_xplane_last    Coordinate of the last X plane of ROI
//! @param roi_yplane_first   Coordinate of the first Y plane of ROI
//! @param roi_yplane_last    Coordinate of the last Y plane of ROI
//! @param roi_zplane_first   Coordinate of the first Z plane of ROI
//! @param roi_zplane_last    Coordinate of the last Z plane of ROI
//! @param DIST_SRC           Radius of x-ray source from isocenter
//! @param DIST_DET           Radius of detector from isocenter
//! @param roi_unit_x         Size of X component of ROI unit
//! @param roi_unit_y         Size of Y component of ROI unit
//! @param roi_unit_z         Size of Z component of ROI unit
void GPU_forwardProjection_init(int NUM_PROJ, int NUM_DET_COL, int NUM_DET_ROW, 
				int NUM_ROI_X, int NUM_ROI_Y, int NUM_ROI_Z, 
				float *det, float *roi, float *delta_det_col,
				float roi_xplane_first, float roi_xplane_last, 
				float roi_yplane_first, float roi_yplane_last,
				float roi_zplane_first, float roi_zplane_last, 
				float DIST_SRC, float DIST_DET,  
				float roi_unit_x, float roi_unit_y, float roi_unit_z
				) 
{
  printf("\n*** GPU_forwardProjection_init()\n");

  numFP++;

  // padding to make a favorable work group size, not used now
  int Nch4GPU = NUM_DET_COL;
  float *det4GPU = (float *)malloc(sizeof(float) * Nch4GPU * NUM_DET_ROW * NUM_PROJ);
  for(int i = 0; i < NUM_PROJ; i++) {
    for(int j = 0; j < NUM_DET_ROW; j++) {
      //for(int k = 0; k < Nch4GPU; k++) {
      for(int k = 0; k < NUM_DET_COL; k++) {
	//if(k < NUM_DET_COL)
	if(k < Nch4GPU)
	  det4GPU[i*NUM_DET_ROW*Nch4GPU + j*Nch4GPU + k] = det[i*NUM_DET_ROW*NUM_DET_COL + j*NUM_DET_COL + k];
	//else
	//det4GPU[i*NUM_DET_ROW*Nch4GPU + j*Nch4GPU + k] = 0.0f;
      }
    }
  }
  float * delta_det_col4GPU = (float *)malloc(sizeof(float)*Nch4GPU);
  //for(int k = 0; k < Nch4GPU; k++) {
  for(int k = 0; k < NUM_DET_COL; k++) {
    //if(k < NUM_DET_COL)
    if(k < Nch4GPU)
      delta_det_col4GPU[k] = delta_det_col[k];
    //else
    //delta_det_col4GPU[k] = 0.0f;    
  }

  int size_det4GPU = sizeof(float) * NUM_PROJ * Nch4GPU * NUM_DET_ROW;
  int size_roi = sizeof(float) * NUM_ROI_X * NUM_ROI_Y * NUM_ROI_Z;
  int size_delta_det_col4GPU = sizeof(float) * Nch4GPU;
  int size_delta_z = sizeof(float) * NUM_DET_ROW;

  // Memory buffer allocation
  cl_int status;
  d_det = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, size_det4GPU, NULL, &status);
#ifndef TEXTURE
  d_roi = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, size_roi, NULL, &status);
#else
  cl_image_format imgFmt;
  imgFmt.image_channel_order = CL_R;
  imgFmt.image_channel_data_type = CL_FLOAT;
  d_roi = clCreateImage3D(clGPUContext, CL_MEM_READ_ONLY, &imgFmt, NUM_ROI_X, NUM_ROI_Y, NUM_ROI_Z, NUM_ROI_X* sizeof(float), NUM_ROI_X*NUM_ROI_Y* sizeof(float), roi, &status);
#endif
  d_delta_det_col = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, size_delta_det_col4GPU, NULL, &status);
  d_delta_z = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, size_delta_z, NULL, &status);
  if(status != CL_SUCCESS) {
    printf("OpenCL Error: %d %s creating buffers in forward projection init\n", status, cl_errs[-status]);
    exit(1);
  }

  status = clEnqueueWriteBuffer(clCommandQueue, d_det, CL_TRUE, 0, size_det4GPU, det4GPU, 0, NULL, NULL); 
#ifndef TEXTURE
  status = clEnqueueWriteBuffer(clCommandQueue, d_roi, CL_TRUE, 0, size_roi, roi, 0, NULL, NULL); 
#else
  const size_t szTexOrigin[3] = {0, 0, 0};
  const size_t szTexRegion[3] = {NUM_ROI_X, NUM_ROI_Y, NUM_ROI_Z};
  status = clEnqueueWriteImage(clCommandQueue, d_roi, CL_FALSE, szTexOrigin, szTexRegion, 0, 0, roi, 0, NULL, NULL);
#endif
  status = clEnqueueWriteBuffer(clCommandQueue, d_delta_det_col, CL_TRUE, 0, size_delta_det_col4GPU, delta_det_col4GPU, 0, NULL, NULL); 
  if(status != CL_SUCCESS) {
    printf("OpenCL Error: %d %s copying data from CPU to GPU in forward projection init\n", status, cl_errs[-status]);
    exit(1);
  }

  status  = clSetKernelArg(forwardProjection, 0, sizeof(cl_mem), (void *)&d_det);
  status |= clSetKernelArg(forwardProjection, 1, sizeof(cl_mem), (void *)&d_roi);
  status |= clSetKernelArg(forwardProjection, 2, sizeof(cl_mem), (void *)&d_delta_det_col);
  // Arg #3 keeps changing (delta_z)
  status |= clSetKernelArg(forwardProjection, 4, sizeof(cl_float), (void*)&roi_xplane_first);
  status |= clSetKernelArg(forwardProjection, 5, sizeof(cl_float), (void*)&roi_xplane_last);
  status |= clSetKernelArg(forwardProjection, 6, sizeof(cl_float), (void*)&roi_yplane_first);
  status |= clSetKernelArg(forwardProjection, 7, sizeof(cl_float), (void*)&roi_yplane_last);
  status |= clSetKernelArg(forwardProjection, 8, sizeof(cl_float), (void*)&roi_zplane_first);
  status |= clSetKernelArg(forwardProjection, 9, sizeof(cl_float), (void*)&roi_zplane_last);
  // Arg #10, #11, #12, #13, #14 keep changing
  status |= clSetKernelArg(forwardProjection, 15, sizeof(cl_int), (void*)&NUM_DET_ROW);
  status |= clSetKernelArg(forwardProjection, 16, sizeof(cl_int), (void*)&Nch4GPU);
  status |= clSetKernelArg(forwardProjection, 17, sizeof(cl_int), (void*)&NUM_ROI_X);
  status |= clSetKernelArg(forwardProjection, 18, sizeof(cl_int), (void*)&NUM_ROI_Y);
  status |= clSetKernelArg(forwardProjection, 19, sizeof(cl_int), (void*)&NUM_ROI_Z);
  status |= clSetKernelArg(forwardProjection, 20, sizeof(cl_float), (void*)&roi_unit_x);
  status |= clSetKernelArg(forwardProjection, 21, sizeof(cl_float), (void*)&roi_unit_y);
  status |= clSetKernelArg(forwardProjection, 22, sizeof(cl_float), (void*)&roi_unit_z);

  workSize_x = NUM_DET_COL;
  workSize_y = NUM_DET_ROW;

  if(cl_errChk(status, "setting kernel arguments")) {
    exit(1);
  }
  
  free(det4GPU);
  free(delta_det_col4GPU);
}


//! \brief GPU forward projection
//! This function does one projection from source to multiple detecter cells
//! This function should be called after GPU_forwardProjection_init()
//! @param delta_z           Array that holds Z coordinate values
//! @param proj_cnt          Current projection count
//! @param DELTA
//! @param ALPHA
//! @param NUM_DET_ROW       Height of detector
//! @param DIST_SRC           
//! @param DIST_DET
void GPU_forwardProjection(float *delta_z,
			   int proj_cnt,
			   float DELTA, float ALPHA, int NUM_DET_ROW,
			   float DIST_SRC, float DIST_DET ) 
{
  cl_int status;
  cl_event event;
  int size_delta_z = sizeof(float)*NUM_DET_ROW;
  status = clEnqueueWriteBuffer(clCommandQueue, d_delta_z, CL_TRUE, 0, size_delta_z, &delta_z[0], 0, NULL, NULL); 
  status |= clSetKernelArg(forwardProjection, 3, sizeof(cl_mem), (void *)&d_delta_z);
  status |= clSetKernelArg(forwardProjection, 10, sizeof(cl_int), (void*)&proj_cnt);
  status |= clSetKernelArg(forwardProjection, 11, sizeof(cl_float), (void*)&DELTA);
  status |= clSetKernelArg(forwardProjection, 12, sizeof(cl_float), (void*)&ALPHA);
  status |= clSetKernelArg(forwardProjection, 13, sizeof(cl_float), (void*)&DIST_SRC);
  status |= clSetKernelArg(forwardProjection, 14, sizeof(cl_float), (void*)&DIST_DET);

  size_t globalWorkSize[] = {workSize_x, workSize_y}; // 400, 32
  size_t localWorkSize[] = {16, 2};
  // actually execute the kernel
  status = clEnqueueNDRangeKernel(clCommandQueue, forwardProjection, 2, NULL,
				  globalWorkSize, localWorkSize, 0, NULL, &event);

  cl_startTimer(event);
  cl_stopTimer(event, FP);

  // cleanup and check for errors
  if(cl_errChk(status, "running forwardProjection kernel")) {
    exit(1);
  }

  clReleaseEvent(event);

}

//! \brief GPU forward projection finalization
//! This function copies the result back from GPU to CPU
//! This function should be called after GPU_forwardProjection()
//! @param det              3-dimensional array that contains projection values
//! @param NUM_PROJ         Total number of projection count
//! @param NUM_DET_COL      Width of the detector
//! @param NUM_DET_ROW      Height of the detector
void GPU_forwardProjectionDone(float *det, int NUM_PROJ, int NUM_DET_COL, int NUM_DET_ROW) 
{
  printf("*** GPU_forwardProjectionDone()\n");

  int Nch4GPU = NUM_DET_COL;
  float *det4GPU = (float *)malloc(sizeof(float) * Nch4GPU * NUM_DET_ROW * NUM_PROJ);
  clEnqueueReadBuffer(clCommandQueue, d_det, CL_TRUE, 0, (sizeof(float) * NUM_PROJ * Nch4GPU * NUM_DET_ROW), 
		      det4GPU, 0, NULL, NULL); 

  // Padding, not used currently
  for(int i = 0; i < NUM_PROJ; i++) {
    for(int j = 0; j < NUM_DET_ROW; j++) {
      //for(int k = 0; k < Nch4GPU; k++) { 
      for(int k = 0; k < NUM_DET_COL; k++) { 
	//if(k < NUM_DET_COL)
	if(k < Nch4GPU)
	  det[i*NUM_DET_ROW*NUM_DET_COL + j*NUM_DET_COL + k] = det4GPU[i*NUM_DET_ROW*Nch4GPU + j*Nch4GPU + k];
      }
    }
  }

  free(det4GPU);
}

//! \brief GPU forward projection deinitialization
//! This function release the memory object and print out timing information
//! This function should be called after GPU_forwardProjectionDone()
void GPU_forwardProjection_deinit() 
{
  printf("*** GPU_forwardProjection_deinit()\n");
  clReleaseMemObject(d_det);
  clReleaseMemObject(d_roi);
  clReleaseMemObject(d_delta_det_col);
  clReleaseMemObject(d_delta_z);  
  cl_timingInfo();
}


//! \brief GPU backward projection initialization
//! GPU initialization fuction
//! This function should be called after GPU_init() and before GPU_backwardProjection()
//! @param NUM_PROJ           Total projection count
//! @param NUM_DET_COL        Width of detector
//! @param NUM_DET_ROW        Height of detector
//! @param NUM_ROI_X          ROI size (x axis)
//! @param NUM_ROI_Y          ROI size (y axis)
//! @param NUM_ROI_Z          ROI size (z axis)
//! @param det                3-dimentional array that contains detector values
//! @param roi                3-dimentional array that contains ROI values
//! @param delta_det_col      Relative angles of detector cell with respect to xy coordiate
//! @param roi_xplane_first   Coordinate of the first X plane of ROI
//! @param roi_xplane_last    Coordinate of the last X plane of ROI
//! @param roi_yplane_first   Coordinate of the first Y plane of ROI
//! @param roi_yplane_last    Coordinate of the last Y plane of ROI
//! @param roi_zplane_first   Coordinate of the first Z plane of ROI
//! @param roi_zplane_last    Coordinate of the last Z plane of ROI
//! @param DIST_SRC           Radius of x-ray source from isocenter
//! @param DIST_DET           Radius of detector from isocenter
//! @param roi_unit_x         Size of X component of ROI unit
//! @param roi_unit_y         Size of Y component of ROI unit
//! @param roi_unit_z         Size of Z component of ROI unit
void GPU_backwardProjection_init(int NUM_PROJ, int NUM_DET_COL, int NUM_DET_ROW, 
				 int NUM_ROI_X, int NUM_ROI_Y, int NUM_ROI_Z, 
				 float *det, float *roi, float *delta_det_col,
				 float roi_xplane_first, float roi_xplane_last, 
				 float roi_yplane_first, float roi_yplane_last,
				 float roi_zplane_first, float roi_zplane_last, 
				 float DIST_SRC, float DIST_DET,  
				 float roi_unit_x, float roi_unit_y, float roi_unit_z
				 ) 
{
  printf("\n*** GPU_backwardProjection_init()\n");

  numBP++;

  int size_det = sizeof(float) * NUM_PROJ * NUM_DET_COL * NUM_DET_ROW;
  int size_roi = sizeof(float) * NUM_ROI_X * NUM_ROI_Y * NUM_ROI_Z;
  int size_delta_det_col = sizeof(float) * NUM_DET_COL;
  int size_delta_z = sizeof(float) * NUM_DET_ROW;
				
  cl_int status1;
  d_det = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, size_det, NULL, &status1);
#ifndef TEXTURE_WRITE
  d_roi = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, size_roi, NULL, &status1);
#else
  cl_image_format imgFmt;
  imgFmt.image_channel_order = CL_R;
  imgFmt.image_channel_data_type = CL_FLOAT;
  d_roi = clCreateImage3D(clGPUContext, CL_MEM_WRITE_ONLY, &imgFmt, NUM_ROI_X, NUM_ROI_Y, NUM_ROI_Z, NUM_ROI_X* sizeof(float), NUM_ROI_X*NUM_ROI_Y* sizeof(float), roi, &status1);
#endif

  d_delta_det_col = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, size_delta_det_col, NULL, &status1);
  d_delta_z = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY, size_delta_z, NULL, &status1);

  status1 = clEnqueueWriteBuffer(clCommandQueue, d_det, CL_TRUE, 0, size_det, det, 0, NULL, NULL); 
#ifndef TEXTURE_WRITE
  status1 = clEnqueueWriteBuffer(clCommandQueue, d_roi, CL_TRUE, 0, size_roi, roi, 0, NULL, NULL); 
#else
  const size_t szTexOrigin[3] = {0, 0, 0};
  const size_t szTexRegion[3] = {NUM_ROI_X, NUM_ROI_Y, NUM_ROI_Z};
  status1 = clEnqueueWriteImage(clCommandQueue, d_roi, CL_FALSE, szTexOrigin, szTexRegion, 0, 0, roi, 0, NULL, NULL);
#endif
  status1 = clEnqueueWriteBuffer(clCommandQueue, d_delta_det_col, CL_TRUE, 0, size_delta_det_col, delta_det_col, 0, NULL, NULL); 

  cl_int status;
  status  = clSetKernelArg(backwardProjection, 0, sizeof(cl_mem), (void *)&d_det);
  status |= clSetKernelArg(backwardProjection, 1, sizeof(cl_mem), (void *)&d_roi);
  status |= clSetKernelArg(backwardProjection, 2, sizeof(cl_mem), (void *)&d_delta_det_col);
  // Arg #3 keeps changing (delta_z)
  status |= clSetKernelArg(backwardProjection, 4, sizeof(cl_float), (void*)&roi_xplane_first);
  status |= clSetKernelArg(backwardProjection, 5, sizeof(cl_float), (void*)&roi_xplane_last);
  status |= clSetKernelArg(backwardProjection, 6, sizeof(cl_float), (void*)&roi_yplane_first);
  status |= clSetKernelArg(backwardProjection, 7, sizeof(cl_float), (void*)&roi_yplane_last);
  status |= clSetKernelArg(backwardProjection, 8, sizeof(cl_float), (void*)&roi_zplane_first);
  status |= clSetKernelArg(backwardProjection, 9, sizeof(cl_float), (void*)&roi_zplane_last);
  // Arg #10, #11, #12, #13, #14
  status |= clSetKernelArg(backwardProjection, 15, sizeof(cl_int), (void*)&NUM_DET_ROW);
  status |= clSetKernelArg(backwardProjection, 16, sizeof(cl_int), (void*)&NUM_DET_COL);
  status |= clSetKernelArg(backwardProjection, 17, sizeof(cl_int), (void*)&NUM_ROI_X);
  status |= clSetKernelArg(backwardProjection, 18, sizeof(cl_int), (void*)&NUM_ROI_Y);
  status |= clSetKernelArg(backwardProjection, 19, sizeof(cl_int), (void*)&NUM_ROI_Z);
  status |= clSetKernelArg(backwardProjection, 20, sizeof(cl_float), (void*)&roi_unit_x);
  status |= clSetKernelArg(backwardProjection, 21, sizeof(cl_float), (void*)&roi_unit_y);
  status |= clSetKernelArg(backwardProjection, 22, sizeof(cl_float), (void*)&roi_unit_z);

  workSize_x = NUM_DET_COL;
  workSize_y = NUM_DET_ROW;
  
  if(cl_errChk(status, "setting back projection arguments")) {
    exit(1);
  }
}


//! \brief GPU backward projection
//! This function does one backward projection from source to multiple detecter cells
//! This function should be called after GPU_backwardProjection_init()
//! @param delta_z           Array that holds Z coordinate values
//! @param proj_cnt          Current projection count
//! @param DELTA
//! @param ALPHA
//! @param NUM_DET_ROW       Height of detector
//! @param DIST_SRC           
//! @param DIST_DET
void GPU_backwardProjection(float *delta_z,
			    int proj_cnt,
			    float DELTA, float ALPHA, int NUM_DET_ROW,
			    float DIST_SRC, float DIST_DET ) 
{
  cl_int status;
  cl_event event;
  int size_delta_z = sizeof(float) * NUM_DET_ROW;
  status = clEnqueueWriteBuffer(clCommandQueue, d_delta_z, CL_TRUE, 0, size_delta_z, &delta_z[0], 0, NULL, NULL); 

  status |= clSetKernelArg(backwardProjection, 3, sizeof(cl_mem), (void *)&d_delta_z);
  status |= clSetKernelArg(backwardProjection, 10, sizeof(cl_int), (void*)&proj_cnt);
  status |= clSetKernelArg(backwardProjection, 11, sizeof(cl_float), (void*)&DELTA);
  status |= clSetKernelArg(backwardProjection, 12, sizeof(cl_float), (void*)&ALPHA);
  status |= clSetKernelArg(backwardProjection, 13, sizeof(cl_float), (void*)&DIST_SRC);
  status |= clSetKernelArg(backwardProjection, 14, sizeof(cl_float), (void*)&DIST_DET);

  size_t globalWorkSize[] = {workSize_x, workSize_y};
  size_t localWorkSize[] = {16, 2};

  status = clEnqueueNDRangeKernel(clCommandQueue, backwardProjection, 2, NULL,
				  globalWorkSize, localWorkSize, 0, NULL, &event);

  cl_startTimer(event);
  cl_stopTimer(event, BP);

  if(cl_errChk(status, "running forwardProjection kernel")) {
    exit(1);
  }

  clReleaseEvent(event);
}

//! \brief GPU backward projection finalization
//! This function copies the result back from GPU to CPU
//! This function should be called after GPU_backwardProjection()
//! @param roi             3D array that contains ROI values
//! @param NUM_ROI_X       X dimension size of ROI
//! @param NUM_ROI_Y       Y dimension size of ROI
//! @param NUM_ROI_Z       Z dimension size of ROI
void GPU_backwardProjectionDone(float *roi, int NUM_ROI_X, int NUM_ROI_Y, int NUM_ROI_Z) 
{
  printf("*** GPU_backwardProjectionDone()\n");
  clEnqueueReadBuffer(clCommandQueue, d_roi, CL_TRUE, 0, (sizeof(float)*NUM_ROI_X*NUM_ROI_Y*NUM_ROI_Z), roi, 0, NULL, NULL); 
}

//! \brief GPU backward projection deinitialization
//! This function release the memory object and print out timing information
//! This function should be called after GPU_backwardProjectionDone()
void GPU_backwardProjection_deinit() 
{
  printf("*** GPU_backwardProjection_deinit()\n");
  clReleaseMemObject(d_det);
  clReleaseMemObject(d_roi);
  clReleaseMemObject(d_delta_det_col);
  clReleaseMemObject(d_delta_z);  
  cl_timingInfo();
}
