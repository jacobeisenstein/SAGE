#include <mex.h>
#include <assert.h>
#include <math.h>
#include "util.h"

/*function [theta q_a score lv_score word_score sigma] = tamEStep(x,beta,alpha,a_log_prior,sigma)
  my_counts = [N x 1] term counts for all unique terms in the doc
  my_beta = [N x K x A] log topic-word distribs for all unique terms in the doc
  alpha = [1 x K] doc-topic prior
  a_log_prior = [1 x A] doc-aspect prior

  output: 
  sigma = [1 x K] variational posterior distrib over theta
  q_a = [1 x A] variational posterior distrib over aspect
  phi = [N x K] variational posterior over z (per token topic)

  todo: include additional terms from KL(P(theta) || Q(theta)) that involve
  alpha, so that you can do alpha updating. also, optimize further using fast exp or something.
*/

#define THRESH 0.00001
#define MAX_ITS 25
#define FAST_EXP 0

int DEBUG=0;

static union 
{
  double d;
  struct {
#ifdef LITTLE_ENDIAN
    int j,i;
#else 
    int i,j;
#endif
  } n;
} _eco;

#define EXP_A (1048576/0.69314718055994530942)
#define EXP_C 60801
#define EXP(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)

void my_assert(int condition){
  if (!condition)
    mexErrMsgTxt("assertion failed");
}

void logToSimplex4(const double* source, double* target, int K, int fast_exp){
  double log_sum_exp,max_val;
  int k;
  max_val = source[0];
  for (k = 1; k < K; k++) 
    if (source[k] > max_val) max_val = source[k];

  log_sum_exp = 0;
  for (k = 0; k < K; k++){
    if (fast_exp) log_sum_exp += EXP(source[k] - max_val);
    else log_sum_exp += exp(source[k] - max_val);
  }
  log_sum_exp = log(log_sum_exp); 
  for (k = 0; k < K; k++){
    if (fast_exp) target[k] = EXP(source[k] - max_val - log_sum_exp);
    else target[k] = exp(source[k] - log_sum_exp - max_val);
  }
}

void logToSimplex(const double* source, double* target, int K){
  logToSimplex4(source,target,K,FAST_EXP);
}

int getIdx(int i, int j, int k, int I, int J, int K){
  return i + j * I + k * I * J;
}

double get(double* arr, int i, int j, int k, int I, int J, int K){
  /*  if (i % 10 == 2 && j % 10 ==3 && k % 10 == 4)*/
  /* printf("%.3f %i %i/%i %i/%i %i/%i\n",arr[idx],idx,i,I,j,J,k,K); */
  return arr[getIdx(i,j,k,I,J,K)];
}

double set(double* arr, double val, int i, int j, int k, int I, int J, int K){
  arr[getIdx(i,j,k,I,J,K)] = val;
}

/*
  function [sigma q_a out_counts score lv_score word_score] 
  = tamEStep(x,beta,alpha,a_log_prior,sigma)
*/

void mexFunction(
		 int nlhs,       mxArray *plhs[],
		 int nrhs, const mxArray *prhs[]
		 )
{
  if (nrhs < 4 || nrhs > 5) {
    mexErrMsgTxt("function [sigma q_a phi] = tamEStepInner(c,beta,alpha,a_log_prior,[sigma])");
    return; 
  }
  mwSize N,A,K;
  int n,a,k,it;
  double word_score, lv_score, log_p_k_theta_denom, log_p_w_a, old_score, change;
  mwSize* beta_dims;
  beta_dims = mxGetDimensions(prhs[1]);  

  N = mxGetM(prhs[0]);

  /*  if (N == 399)
    DEBUG=1;
  else
    DEBUG=0;
*/

  K = beta_dims[1];  

  if (mxGetNumberOfDimensions(prhs[1]) > 2) 
    A = beta_dims[2];  
  else A = 1;

  mxAssert (beta_dims[0] == N,"beta dims must match counts");
  mxAssert (K == mxGetN(prhs[2]),"beta dims must match K");
  mxAssert (A == mxGetN(prhs[3]),"beta dims must match A");
  if (DEBUG)
    printf("dimensions N=%i K=%i A=%i\n",N,K,A);
  
  double* counts = mxGetPr(prhs[0]);
  double* beta = mxGetPr(prhs[1]); 
  double* alpha = mxGetPr(prhs[2]);
  double* log_p_a = mxGetPr(prhs[3]);

  double* sigma;
  if (nrhs == 5) {
    sigma = mxGetPr(prhs[4]);
    if (DEBUG){
      printf("reading sigma:\n");
      for (k = 0; k < K; k++){
	printf("%.3f ",sigma[k]);
      }
      printf("\n");
    }
  }
  else {
    sigma = mxCalloc(K,sizeof(double));
    for (k = 0; k < K; k++){
      sigma[k] = alpha[k];
      for (n = 0; n < N; n++){
	sigma[k] += counts[n] / (double) K;
      }
    }
  }
  
  double* sigma_buffer = mxCalloc(K,sizeof(double));
  double* phi = mxCalloc(K,sizeof(double));
  double* phi_buffer = mxCalloc(K,sizeof(double));
  double* q_a = mxCalloc(A,sizeof(double));
  double* a_buffer = mxCalloc(A,sizeof(double));
  double* log_p_k_theta = mxCalloc(K,sizeof(double));
  double* out; /* for output counts */
  
  logToSimplex(log_p_a,q_a,A);

  old_score = 0;
  for (it = 0; it < MAX_ITS; it++){
    word_score = 0.0; lv_score = 0.0;

    for (k = 0; k < K; k++) {
      sigma_buffer[k] = alpha[k]; /*this is for storing sigma later */
      log_p_k_theta_denom += sigma[k];
    }
    log_p_k_theta_denom = digamma(log_p_k_theta_denom);

    /* parts of KL( P(theta) || Q(theta) ) 
       Note we are leaving out constant parts:
       -gammaln(sum(sigma)) + gammaln(sum(alpha) - sum(gammaln(alpha))
    */
    for (k = 0; k < K; k++){
      log_p_k_theta[k] = digamma(sigma[k]) - log_p_k_theta_denom;
      lv_score += log_p_k_theta[k] * (alpha[k] - sigma[k]);
      lv_score += gammaln(sigma[k]);
    }
    for (a = 0; a < A; a++){
      if (q_a[a] > 0) 
	lv_score += q_a[a] * (log_p_a[a] - log(q_a[a])); /* E[log P(a | var_theta)] - E[log q(a)]*/
      a_buffer[a] = log_p_a[a]; /* for updating q(a) */
    }

    for (n = 0; n < N; n++){
      /* compute phi */
      for (k = 0; k < K; k++){
	phi_buffer[k] = log_p_k_theta[k];
	for (a = 0; a < A; a++){
	  phi_buffer[k] += q_a[a] * get(beta,n,k,a,N,K,A);
	}
      }
      logToSimplex(phi_buffer,phi,K);

      /* E[log p(z | theta)] - E[log Q(z)] */
      for (k = 0; k < K; k++){
	if (phi[k] > 0){
	  lv_score += counts[n] * phi[k] * (log_p_k_theta[k] - log(phi[k]));
	  phi[k] *= counts[n]; /* this is for efficiency later */
	  sigma_buffer[k] += phi[k];
	  for (a = 0; a < A; a++){
	    log_p_w_a = get(beta,n,k,a,N,K,A);
	    a_buffer[a] += phi[k] * log_p_w_a;
	    word_score += q_a[a] * phi[k] * log_p_w_a;
	  }
	}
      }
    }
    logToSimplex(a_buffer,q_a,A);
    for (k = 0; k < K; k++) sigma[k] = sigma_buffer[k];

    /* print status */
    if (DEBUG){
      for (a = 0; a < A; a++) printf("%.3e ",log(q_a[a]));
      printf("\n");
      for (k = 0; k < K; k++) printf("%.3f ",sigma[k]);
      printf("\n");
    }

    change = (word_score + lv_score - old_score) / abs(old_score);
    if (DEBUG) printf("Iteration %i (%.3e): %.6e = %.3e + %.3e\n",it,change,word_score + lv_score,word_score,lv_score);
    if (change < THRESH && it > 0){
      break;
    }
    old_score = word_score + lv_score;
  }
  
  /************************* OUTPUT ********************/
  /* theta */
  if (nlhs > 0){
    plhs[0] = mxCreateDoubleMatrix(1,K,mxREAL);
    out = mxGetPr(plhs[0]);
    for (k = 0; k < K; k++) out[k] = sigma[k]; 
  }
  if (nlhs > 1){
    plhs[1] = mxCreateDoubleMatrix(1,A,mxREAL);
    out = mxGetPr(plhs[1]);
    for (a = 0; a < A; a++) out[a] = q_a[a];
  }

  if (nlhs > 2){
    int beta_dims_out[3];

    beta_dims_out[0] = N;
    beta_dims_out[1] = K;
    beta_dims_out[2] = A;
    
    plhs[2] = mxCreateNumericArray(3,beta_dims_out,mxDOUBLE_CLASS,mxREAL);
    out = mxGetPr(plhs[2]);
    for (n = 0; n < N; n++){
      for (k = 0; k < K; k++){
	phi_buffer[k] = log_p_k_theta[k];
	for (a = 0; a < A; a++)
	  phi_buffer[k] += q_a[a] * get(beta,n,k,a,N,K,A);
      }
      logToSimplex(phi_buffer,phi,K);
      for (k = 0; k < K; k++){
	for (a = 0; a < A; a++)
	  set(out,counts[n] * q_a[a] * phi[k],n,k,a,N,K,A);
      }
    }
  }
  if (nlhs > 3)
    plhs[3] = mxCreateDoubleScalar(word_score + lv_score);
  if (nlhs > 4)
    plhs[4] = mxCreateDoubleScalar(word_score);
  if (nlhs > 5)
    plhs[5] = mxCreateDoubleScalar(lv_score);
  
}
  
