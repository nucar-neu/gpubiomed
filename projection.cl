#define TEXTURE
//#define TEXTURE_WRITE

//#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

//! \brief GPU forward projection kernel
//! This function does one forward projection on GPU
//! @param det              3-dimentional array that contains projection values
//! @param roi              3-dimentional array that contains ROI values
//! @param delta_det_col    Relative angles of detector cell with respect to xy coordinate
//! @param delta_z          Relative angles of detector cell with respect to z coordinate
//! @param x0               Coordinate of the first X plane of ROI
//! @param xN               Coordinate of the last X plane of ROI
//! @param y0               Coordinate of the first Y plane of ROI
//! @param yN               Coordinate of the last Y plane of ROI
//! @param z0               Coordinate of the first Z plane of ROI
//! @param zN               Coordinate of the last Z plane of ROI
//! @param proj_cnt         Current projection angle count
//! @param ZS                
//! @param ALPHA                
//! @param DIST_SRC                
//! @param DIST_DET                
//! @param NUM_DET_ROW      Height of detector
//! @param NUM_DET_COL      Width of detector
//! @param NUM_ROI_X        ROI size (X axis)
//! @param NUM_ROI_Y        ROI size (Y axis)
//! @param NUM_ROI_Z        ROI size (Z axis)
//! @param roi_unit_x       Size of X component of ROI unit
//! @param roi_unit_y       Size of Y component of ROI unit
//! @param roi_unit_z       Size of Z component of ROI unit
__kernel void forwardProjection( __global float * det,             // 0
#ifndef TEXTURE
				 __global float * roi,             // 1
#else
				 __read_only image3d_t roi,        // 1
#endif
				 __global float * delta_det_col,   // 2
				 __global float * delta_z,         // 3
				 float x0,                         // 4
				 float xN,                         // 5
				 float y0,                         // 6
				 float yN,                         // 7
				 float z0,                         // 8
				 float zN,                         // 9
				 int proj_cnt,                     // 10
				 float ZS,                         // 11
				 float ALPHA,                      // 12
				 float DIST_SRC, float DIST_DET,   // 13, 14
				 int NUM_DET_ROW, int NUM_DET_COL, // 15, 16
				 int NUM_ROI_X,                    // 17
				 int NUM_ROI_Y,                    // 18
				 int NUM_ROI_Z,                    // 19
				 float roi_unit_x,                 // 20
				 float roi_unit_y,                 // 21
				 float roi_unit_z                  // 22
				 )
{
  // i1 :: tx, i2 :: ty
  int tx = get_global_id(0);
  int ty = get_global_id(1);

#ifdef TEXTURE
  const sampler_t samplerA = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#endif

  float ytemp = 0.0f;
  float ytemp2 = 0.0f;
  
  float X1, X2, Y1, Y2, Z1, Z2;

  X1 = DIST_SRC * cos(ALPHA);
  Y1 = DIST_SRC * sin(ALPHA);
  Z1 = ZS;
  
  float phi = ALPHA + delta_det_col[tx];
  X2 = -(DIST_SRC+DIST_DET)*cos(phi) + DIST_SRC*cos(ALPHA);
  Y2 = -(DIST_SRC+DIST_DET)*sin(phi) + DIST_SRC*sin(ALPHA);
  Z2 = delta_z[ty];

  int xu, yu, zu;

  xu = yu = zu = 1;
  if(X2 < X1) xu = -1;		    
  if(Y2 < Y1) yu = -1;		    
  if(Z2 < Z1) zu = -1;

  // euclidian distance
  float dist = sqrt((X2-X1)*(X2-X1) + (Y2-Y1)*(Y2-Y1) + (Z2-Z1)*(Z2-Z1));
  
  float ax0, axN, ay0, ayN,az0, azN;
  float xmin, ymin, zmin, xmax, ymax, zmax;
  float amin, amax;
  float ax, ay, az, ac, axu, ayu, azu, da;
  int xidx, yidx, zidx, img_idx;
  float	alpha_mid;

  // following three else statements doesn't affect the execution time
  if(X2 != X1) {
    ax0 = (x0-X1)/(X2-X1);
    axN = (xN-X1)/(X2-X1);
    axu = roi_unit_x/fabs(X2-X1);    // increament of alpha x
  }   
  else { 
    ax0 = 0.0f; axN = 1.0f; axu = 0.0f;  
  }
  
  if(Y2 != Y1) {
    ay0 = (y0-Y1)/(Y2-Y1);
    ayN = (yN-Y1)/(Y2-Y1);
    ayu = roi_unit_y/fabs(Y2-Y1);
  }  
  else {    
    ay0 = 0.0f; ayN = 1.0f; ayu = 0.0f;  
  }
  
  if(Z2 != Z1) {
    az0 = (z0-Z1)/(Z2-Z1);
    azN = (zN-Z1)/(Z2-Z1);
    azu = roi_unit_z/fabs(Z2-Z1);
  }  
  else {  
    az0 = 0.0f; azN = 1.0f; azu = 0.0f;  
  }

  xmin = fmin(ax0, axN);   ymin = fmin(ay0, ayN);   zmin = fmin(az0, azN);
  xmax = fmax(ax0, axN);   ymax = fmax(ay0, ayN);   zmax = fmax(az0, azN);
  
  amin = fmax(xmin, fmax(ymin,zmin));
  amax = fmin(xmax, fmin(ymax,zmax));		    

  if(amin < 0 || amin > 1 || amax < 0 || amax >1 || amax < amin) {
    //cout << " -- ray does not hit roi!" << endl;
    // Projection ray does not cross roi
    return;
  }

  int min_x, max_x, min_y, max_y, min_z, max_z;
  if(X2 > X1) {
    if(amin == xmin) {
      min_x = 0;   // x index of first voxel
      ax = amin + axu;
    }
    else {
      min_x = (int)( (X1 + amin*(X2-X1)-x0)/roi_unit_x + 1.0 ); // Ceil operation
      ax = (min_x*roi_unit_x - X1 + x0)/(X2-X1);
      //ax = (Xplane[min_x] - X1)/(X2-X1);
    }
  }
  else if( X2 < X1) {
    if(amin == xmin) {
      max_x = NUM_ROI_X;
      ax = amin + axu;
    }
    else {
      max_x = (int)((X1 + amin*(X2-X1)-x0)/roi_unit_x);
      ax = (max_x*roi_unit_x - X1 +x0)/(X2-X1);
      //ax = (Xplane[max_x] - X1)/(X2-X1);
    }
  }
  else {
    max_x = (int)((X2-x0)/roi_unit_x);
    min_x = max_x + 1;
  }		      

  if(Y2 > Y1) {
    if(amin == ymin) {
      min_y = 0;
      ay = amin + ayu;
    }
    else {
      min_y = (int)((Y1 + amin*(Y2-Y1)-y0)/roi_unit_y+1.0); // Ceil operation
      ay = (min_y*roi_unit_y - Y1 + y0)/(Y2-Y1);
      //ay = (Yplane[min_y] - Y1)/(Y2-Y1);
    }

  }
  else if ( Y2 < Y1) {
    if(amin == ymin) {
      max_y = NUM_ROI_Y;
      ay = amin + ayu;
    }
    else {
      max_y = (int)((Y1 + amin*(Y2-Y1)-y0)/roi_unit_y); // Ceil operation
      ay = (max_y*roi_unit_y - Y1 +y0)/(Y2-Y1);
      //ay =  (Yplane[max_y] - Y1)/(Y2-Y1);
    }
  }
  else {
    max_y = (int)((Y2-y0)/roi_unit_y);
    min_y = max_y + 1;
  }

  if(Z2 > Z1) {
    if(amin == zmin) {
      min_z = 0;
      az = amin + azu;
    }
    else {
      min_z = (int)((Z1 + amin*(Z2-Z1)-z0)/roi_unit_z+1.0); // Ceil operation
      az = (min_z*roi_unit_z - Z1 + z0)/(Z2-Z1);
      //az = (Zplane[min_z] - Z1)/(Z2-Z1);
    }
  }
  else if( Z2 < Z1) {
    if(amin == zmin) {
      max_z = NUM_ROI_Z;
      az = amin + azu;
    }
    else {
      max_z = (int)((Z1 + amin*(Z2-Z1)-z0)/roi_unit_z); // Ceil operation
      // az = (Zplane[max_z] - Z1)/(Z2-Z1);
      az = (max_z*roi_unit_z - Z1 +z0)/(Z2-Z1);
    }
  }
  else {
    max_z = (Z2-z0)/roi_unit_z;
    min_z = max_z+1;
  }

  alpha_mid = (fmin(ax, fmin(ay, az)) + amin) / 2.0f;

  // at this point, we assume that alpha_mid is valid 
  // and fully effective number, +0.0 < alpha_mid < +1.0
  xidx = (X1 + alpha_mid*(X2-X1) - x0)/roi_unit_x;
  yidx = (Y1 + alpha_mid*(Y2-Y1) - y0)/roi_unit_y;
  zidx = (Z1 + alpha_mid*(Z2-Z1) - z0)/roi_unit_z;
  /*
  if(X1 <= X2)
    xidx = fabs(X1 + alpha_mid*(X2-X1) - x0)/roi_unit_x;
  else
    xidx = fabs(X1 - alpha_mid*(X2-X1) - x0)/roi_unit_x;
  if(Y1 <= Y2)
    yidx = fabs(Y1 + alpha_mid*(Y2-Y1) - y0)/roi_unit_y;
  else
    yidx = fabs(Y1 - alpha_mid*(Y2-Y1) - y0)/roi_unit_y;
  if(Z1 <= Z2)
    zidx = fabs(Z1 + alpha_mid*(Z2-Z1) - z0)/roi_unit_z;
  else
    zidx = fabs(Z1 - alpha_mid*(Z2-Z1) - z0)/roi_unit_z;
  */
  ac = amin;

  while(ac <= amax) { 
    // this if doesn't affect the execution time
    if( xidx < 0 || xidx > NUM_ROI_X - 1 ||
	yidx < 0 || yidx > NUM_ROI_Y - 1 ||
	zidx < 0 || zidx > NUM_ROI_Z - 1 ) {
      img_idx = -1;
    }		
    else
      img_idx = NUM_ROI_X*NUM_ROI_Y*zidx + NUM_ROI_X*xidx + yidx;
		    
    if(fmin(ax,fmin(ay,az)) == ax) {
      da = ax-ac;
      xidx += xu;
      ac = ax;
      ax = ax + axu;
    }
    else if(fmin(ax,fmin(ay,az)) == ay) {
      da = ay-ac;
      yidx += yu;
      ac = ay;
      ay = ay + ayu;
    }
    else {
      da = az-ac;
      zidx += zu;
      ac = az;
      az = az + azu;
    }
#ifndef TEXTURE
      ytemp2 += da * roi[img_idx];
#else
      int4 pos = {yidx, xidx, zidx, 0};
      ytemp2 += da * (read_imagef(roi, samplerA, pos)).x;
#endif
  }

  ytemp2 *= dist;
  ytemp += ytemp2;
  det[proj_cnt*NUM_DET_COL*NUM_DET_ROW + ty*NUM_DET_COL + tx] = ytemp;
}


//! \brief GPU forward projection kernel
//! This function does one forward projection on GPU
//! @param det              3-dimentional array that contains projection values
//! @param roi              3-dimentional array that contains ROI values
//! @param delta_det_col    Relative angles of detector cell with respect to xy coordinate
//! @param delta_z          Relative angles of detector cell with respect to z coordinate
//! @param x0               Coordinate of the first X plane of ROI
//! @param xN               Coordinate of the last X plane of ROI
//! @param y0               Coordinate of the first Y plane of ROI
//! @param yN               Coordinate of the last Y plane of ROI
//! @param z0               Coordinate of the first Z plane of ROI
//! @param zN               Coordinate of the last Z plane of ROI
//! @param proj_cnt         Current projection angle count
//! @param ZS                
//! @param ALPHA                
//! @param DIST_SRC                
//! @param DIST_DET                
//! @param NUM_DET_ROW      Height of detector
//! @param NUM_DET_COL      Width of detector
//! @param NUM_ROI_X        ROI size (X axis)
//! @param NUM_ROI_Y        ROI size (Y axis)
//! @param NUM_ROI_Z        ROI size (Z axis)
//! @param roi_unit_x       Size of X component of ROI unit
//! @param roi_unit_y       Size of Y component of ROI unit
//! @param roi_unit_z       Size of Z component of ROI unit
__kernel void backwardProjection( __global float * det,             // 0
#ifndef TEXTURE_WRITE
				 __global float * roi,              // 1
#else
				 __read_only image3d_t roi,         // 1
#endif
				 __global float * delta_det_col,    // 2
				 __global float * delta_z,          // 3
				 float x0,                          // 4
				 float xN,                          // 5
				 float y0,                          // 6
				 float yN,                          // 7
				 float z0,                          // 8
				 float zN,                          // 9
				 int proj_cnt,                      // 10
				 float ZS,                          // 11
				 float ALPHA,                       // 12
				 float DIST_SRC, float DIST_DET,    // 13, 14
				 int NUM_DET_ROW, int NUM_DET_COL,  // 15, 16
				 int NUM_ROI_X,                     // 17
				 int NUM_ROI_Y,                     // 18
				 int NUM_ROI_Z,                     // 19
				 float roi_unit_x,                  // 20
				 float roi_unit_y,                  // 21
				 float roi_unit_z                   // 22
				 ) 
{
  // i1 :: tx, i2 :: ty
  int tx = get_global_id(0);
  int ty = get_global_id(1);

#ifdef TEXTURE_WRITE
  const sampler_t samplerA = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
#endif

  float ytemp = 0.0f;
  float ytemp2 = 0.0f;
  
  float X1, X2, Y1, Y2, Z1, Z2;

  ytemp = det[proj_cnt*NUM_DET_COL*NUM_DET_ROW + ty*NUM_DET_COL + tx];

  X1 = DIST_SRC * cos(ALPHA);
  Y1 = DIST_SRC * sin(ALPHA);
  Z1 = ZS;
  
  float phi = ALPHA + delta_det_col[tx];
  X2 = -(DIST_SRC+DIST_DET)*cos(phi) + DIST_SRC*cos(ALPHA);
  Y2 = -(DIST_SRC+DIST_DET)*sin(phi) + DIST_SRC*sin(ALPHA);
  Z2 = delta_z[ty];

  int xu, yu, zu;

  xu = yu = zu = 1;
  if(X2 < X1) xu = -1;		    
  if(Y2 < Y1) yu = -1;		    
  if(Z2 < Z1) zu = -1;

  // euclidian distance
  float dist = sqrt((X2-X1)*(X2-X1) + (Y2-Y1)*(Y2-Y1) + (Z2-Z1)*(Z2-Z1));
  
  float ax0, axN, ay0, ayN,az0, azN;
  float xmin, ymin, zmin, xmax, ymax, zmax;
  float amin, amax;
  float ax, ay, az, ac, axu, ayu, azu, da;
  int xidx, yidx, zidx, img_idx;
  float	alpha_mid;

  if(X2 != X1) {
    ax0 = (x0-X1)/(X2-X1);
    axN = (xN-X1)/(X2-X1);
    axu = roi_unit_x/fabs(X2-X1);         // increament of alpha x
  }   
  else {
    ax0 = 0.0f; axN = 1.0f; axu = 0.0f;  
  }
  
  if(Y2 != Y1) {
    ay0 = (y0-Y1)/(Y2-Y1);
    ayN = (yN-Y1)/(Y2-Y1);
    ayu = roi_unit_y/fabs(Y2-Y1);
  }  
  else { 
    ay0 = 0.0f; ayN = 1.0f; ayu = 0.0f;  
  }
  
  if(Z2 != Z1) {
    az0 = (z0-Z1)/(Z2-Z1);
    azN = (zN-Z1)/(Z2-Z1);
    azu = roi_unit_z/fabs(Z2-Z1);
  }  
  else {
    az0 = 0.0f; azN = 1.0f; azu = 0.0f;  
  }

  xmin = fmin(ax0, axN);   ymin = fmin(ay0, ayN);   zmin = fmin(az0, azN);
  xmax = fmax(ax0, axN);   ymax = fmax(ay0, ayN);   zmax = fmax(az0, azN);
  
  amin = fmax(xmin, fmax(ymin,zmin));
  amax = fmin(xmax, fmin(ymax,zmax));		    

  if(amin < 0 || amin > 1 || amax < 0 || amax >1 || amax < amin) {
    //cout << " -- ray does not hit roi!" << endl;
    return;
  }

  int min_x, max_x, min_y, max_y, min_z, max_z;
  if(X2 > X1) {
    if(amin == xmin) {
      min_x = 0;   // x index of first voxel
      ax = amin + axu;
    }
    else {
      min_x = (int)( (X1 + amin*(X2-X1)-x0)/roi_unit_x + 1.0 ); // Ceil operation
      ax = (min_x*roi_unit_x - X1 + x0)/(X2-X1);
      //ax = (Xplane[min_x] - X1)/(X2-X1);
    }
  }
  else if( X2 < X1) {
    if(amin == xmin) {
      max_x = NUM_ROI_X;
      ax = amin + axu;
    }
    else {
      max_x = (int)((X1 + amin*(X2-X1)-x0)/roi_unit_x);
      ax = (max_x*roi_unit_x - X1 +x0)/(X2-X1);
      //ax = (Xplane[max_x] - X1)/(X2-X1);
    }
  }
  else {
    max_x = (int)((X2-x0)/roi_unit_x);
    min_x = max_x + 1;
  }		      

  if(Y2 > Y1) {
    if(amin == ymin) {
      min_y = 0;
      ay = amin + ayu;
    }
    else {
      min_y = (int)((Y1 + amin*(Y2-Y1)-y0)/roi_unit_y+1.0); // Ceil operation
      ay = (min_y*roi_unit_y - Y1 + y0)/(Y2-Y1);
      //ay = (Yplane[min_y] - Y1)/(Y2-Y1);
    }
  }
  else if ( Y2 < Y1) {
    if(amin == ymin) {
      max_y = NUM_ROI_Y;
      ay = amin + ayu;
    }
    else {
      max_y = (int)((Y1 + amin*(Y2-Y1)-y0)/roi_unit_y); // Ceil operation
      ay = (max_y*roi_unit_y - Y1 +y0)/(Y2-Y1);
      //ay =  (Yplane[max_y] - Y1)/(Y2-Y1);
    }
  }
  else {
    max_y = (int)((Y2-y0)/roi_unit_y);
    min_y = max_y + 1;
  }

  if(Z2 > Z1) {
    if(amin == zmin) {
      min_z = 0;
      az = amin + azu;
    }
    else {
      min_z = (int)((Z1 + amin*(Z2-Z1)-z0)/roi_unit_z+1.0); // Ceil operation
      az = (min_z*roi_unit_z - Z1 + z0)/(Z2-Z1);
      //az = (Zplane[min_z] - Z1)/(Z2-Z1);
    }
  }
  else if( Z2 < Z1) {
    if(amin == zmin) {
      max_z = NUM_ROI_Z;
      az = amin + azu;
    }
    else {
      max_z = (int)((Z1 + amin*(Z2-Z1)-z0)/roi_unit_z); // Ceil operation
      // az = (Zplane[max_z] - Z1)/(Z2-Z1);
      az = (max_z*roi_unit_z - Z1 +z0)/(Z2-Z1);
    }
  }
  else {
    max_z = (Z2-z0)/roi_unit_z;
    min_z = max_z+1;
  }

  alpha_mid = (fmin(ax, fmin(ay, az)) + amin) / 2.0f;

  xidx = (X1 + alpha_mid*(X2-X1) - x0)/roi_unit_x;
  yidx = (Y1 + alpha_mid*(Y2-Y1) - y0)/roi_unit_y;
  zidx = (Z1 + alpha_mid*(Z2-Z1) - z0)/roi_unit_z;
  /*
  if(X1 <= X2)
    xidx = fabs(X1 + alpha_mid*(X2-X1) - x0)/roi_unit_x;
  else
    xidx = fabs(X1 - alpha_mid*(X2-X1) - x0)/roi_unit_x;
  if(Y1 <= Y2)
    yidx = fabs(Y1 + alpha_mid*(Y2-Y1) - y0)/roi_unit_y;
  else
    yidx = fabs(Y1 - alpha_mid*(Y2-Y1) - y0)/roi_unit_y;
  if(Z1 <= Z2)
    zidx = fabs(Z1 + alpha_mid*(Z2-Z1) - z0)/roi_unit_z;
  else
    zidx = fabs(Z1 - alpha_mid*(Z2-Z1) - z0)/roi_unit_z;
  */
  ac = amin;

  float tmp_var = ytemp * dist;

  while(ac <= amax) { 
    if( xidx < 0 || xidx > NUM_ROI_X - 1 ||
	yidx < 0 || yidx > NUM_ROI_Y - 1 ||
	zidx < 0 || zidx > NUM_ROI_Z - 1 ) {
      img_idx = -1;
    }		
    else
      img_idx = NUM_ROI_X*NUM_ROI_Y*zidx + NUM_ROI_X*xidx + yidx;
		    
    if(fmin(ax,fmin(ay,az)) == ax) {
      da = ax-ac;
      xidx += xu;
      ac = ax;
      ax = ax + axu;
    }
    else if(fmin(ax,fmin(ay,az)) == ay) {
      da = ay-ac;
      yidx += yu;
      ac = ay;
      ay = ay + ayu;
    }
    else {
      da = az-ac;
      zidx += zu;
      ac = az;
      az = az + azu;
    }
    if(img_idx >= 0) {
#ifndef TEXTURE_WRITE
      roi[img_idx] += da * tmp_var;
#else
      int4 pos = {yidx, xidx, zidx, 0};
      write_imagef(roi, pos, da*tmp_var);
#endif
    }
  }
}
