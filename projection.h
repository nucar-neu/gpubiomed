#ifndef __PROJECTION_H__
#define __PROJECTION_H__

#include <CL/cl.h>

#define TEXTURE
//#define TEXTURE_WRITE

void GPU_init();
void GPU_cleanup();

// Forward Projection API
void GPU_forwardProjection_init(int NUM_PROJ, int NUM_DET_COL, int NUM_DET_ROW, 
				int NUM_ROI_X, int NUM_ROI_Y, int NUM_ROI_Z, 
				float *det, float *roi, float *delta_det_col,
				float roi_xplane_first, float roi_xplane_last, 
				float roi_yplane_first, float roi_yplane_last,
				float roi_zplane_first, float roi_zplane_last, 
				float DIST_SRC, float DIST_DET,  
				float roi_unit_x, float roi_unit_y, float roi_unit_z
				);
void GPU_forwardProjection(float *delta_z,
			   int proj_cnt,
			   float DELTA, float ALPHA, int NUM_DET_ROW,
			   float DIST_SRC, float DIST_DET
			   );
void GPU_forwardProjectionDone(float *det, int NUM_PROJ, int NUM_DET_COL, int NUM_DET_ROW);
void GPU_forwardProjection_deinit();

// Backward Projection API
void GPU_backwardProjection_init(int NUM_PROJ, int NUM_DET_COL, int NUM_DET_ROW, 
				 int NUM_ROI_X, int NUM_ROI_Y, int NUM_ROI_Z, 
				 float *det, float *roi, float *delta_det_col,
				 float roi_xplane_first, float roi_xplane_last, 
				 float roi_yplane_first, float roi_yplane_last,
				 float roi_zplane_first, float roi_zplane_last, 
				 float DIST_SRC, float DIST_DET,  
				 float roi_unit_x, float roi_unit_y, float roi_unit_z 
				 );
void GPU_backwardProjection(float *delta_z,
			    int proj_cnt,
			    float DELTA, float ALPHA, int NUM_DET_ROW,
			    float DIST_SRC, float DIST_DET
			    );
void GPU_backwardProjectionDone(float *roi, int NUM_ROI_X, int NUM_ROI_Y, int NUM_ROI_Z);
void GPU_backwardProjection_deinit();

#endif // __PROJECTION_H__
