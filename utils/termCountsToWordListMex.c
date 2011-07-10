#include "mex.h"

/**
   here's how this works
   prhs[0] = columns [row vector]
   prhs[1] = counts  [row vector]
   
**/

void mexFunction(
		 int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]
		 )
{
  if (nrhs != 2 || nlhs != 1){ mexErrMsgTxt("out = func(cols,counts)"); }
  
  double* cols, *counts, *out;
  int i,j,ctr,tot_counts,num_cols;
  
  tot_counts = 0;
  cols = mxGetPr(prhs[0]);
  counts = mxGetPr(prhs[1]);
  num_cols = mxGetN(prhs[0]);

  for (i = 0; i < num_cols; i++) tot_counts += counts[i]; 

  plhs[0] = mxCreateDoubleMatrix(1,tot_counts,mxREAL);
  out = mxGetPr(plhs[0]);

  ctr = 0;
  for (i = 0; i < num_cols ;i++){
    for (j = 0; j < counts[i]; j++){
      out[ctr++] = cols[i];
    }
  }
}
