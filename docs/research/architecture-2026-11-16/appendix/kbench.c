// Scratch micro-benchmark for the architecture review (NOT part of the repository).
// In-memory synthetic BF16 matrix, no file I/O. Compares:
//  A) repo semantics: BF16->FP64 expand into arena, then ordered scalar FP64 GEMV (two passes)
//  B) fused BF16->FP64 AVX2 GEMV (4x4 accumulators, row-aligned)
//  C) fused BF16->FP32 AVX2 GEMV (4x8 accumulators, row-aligned), FP32 accumulate
//  D) fused BF16->FP32 GEMM with T right-hand sides (prefill batching), FP32
#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

// --- A: same code as src/cpu_kernel.c (copied semantics) ---
static int expand(const unsigned char*raw,size_t count,double*out){
  for(size_t i=0;i<count;++i){uint32_t w=(uint32_t)raw[2*i]|(uint32_t)raw[2*i+1]<<8;
    if((w&0x7f80u)==0x7f80u)return 1;uint32_t b=w<<16;float v;memcpy(&v,&b,4);out[i]=(double)v;}
  return 0;}
__attribute__((optimize("no-fast-math"),noinline))
static void linear_ordered(const double*w,size_t rows,size_t cols,const double*x,double*y){
  for(size_t r=0;r<rows;++r){double s=0.;const double*a=w+r*cols;
    for(size_t j=0;j<cols;++j)s+=a[j]*x[j];y[r]=s;}
}
// --- B: fused BF16 -> FP64, AVX2+FMA, fixed lane structure (deterministic per row) ---
static inline double row_f64(const uint16_t*a,const double*x,size_t cols){
  __m256d s0=_mm256_setzero_pd(),s1=s0,s2=s0,s3=s0;size_t j=0;
  for(;j+16<=cols;j+=16){
    __m256i u0=_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a+j))),16);
    __m256i u1=_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a+j+8))),16);
    __m256 f0=_mm256_castsi256_ps(u0),f1=_mm256_castsi256_ps(u1);
    s0=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(f0)),_mm256_loadu_pd(x+j),s0);
    s1=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(f0,1)),_mm256_loadu_pd(x+j+4),s1);
    s2=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(f1)),_mm256_loadu_pd(x+j+8),s2);
    s3=_mm256_fmadd_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(f1,1)),_mm256_loadu_pd(x+j+12),s3);}
  __m256d s=_mm256_add_pd(_mm256_add_pd(s0,s1),_mm256_add_pd(s2,s3));
  double t[4];_mm256_storeu_pd(t,s);double r=(t[0]+t[1])+(t[2]+t[3]);
  for(;j<cols;++j){uint32_t b=(uint32_t)a[j]<<16;float v;memcpy(&v,&b,4);r+=(double)v*x[j];}
  return r;}
// --- C: fused BF16 -> FP32, AVX2+FMA ---
static inline float row_f32(const uint16_t*a,const float*x,size_t cols){
  __m256 s0=_mm256_setzero_ps(),s1=s0,s2=s0,s3=s0;size_t j=0;
  for(;j+32<=cols;j+=32){
#define LD(k) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a+j+k))),16))
    s0=_mm256_fmadd_ps(LD(0),_mm256_loadu_ps(x+j),s0);
    s1=_mm256_fmadd_ps(LD(8),_mm256_loadu_ps(x+j+8),s1);
    s2=_mm256_fmadd_ps(LD(16),_mm256_loadu_ps(x+j+16),s2);
    s3=_mm256_fmadd_ps(LD(24),_mm256_loadu_ps(x+j+24),s3);}
  __m256 s=_mm256_add_ps(_mm256_add_ps(s0,s1),_mm256_add_ps(s2,s3));
  float t[8];_mm256_storeu_ps(t,s);float r=((t[0]+t[1])+(t[2]+t[3]))+((t[4]+t[5])+(t[6]+t[7]));
  for(;j<cols;++j){uint32_t b=(uint32_t)a[j]<<16;float v;memcpy(&v,&b,4);r+=v*x[j];}
  return r;}
// --- D: GEMM, T right-hand sides: Y[r,t] = sum_j W[r,j]*X[j,t]; X stored row-major cols x T (FP32)
// row of W is expanded once to FP32 (stack tile), then axpy over X rows: y_r[0:T] += w_rj * X[j,0:T]
static void gemm_rows_f32(const uint16_t*w,size_t r0,size_t r1,size_t cols,const float*X,size_t T,float*Y,float*wrow){
  for(size_t r=r0;r<r1;++r){const uint16_t*a=w+r*cols;float*y=Y+r*T;
    for(size_t j=0;j+8<=cols;j+=8)_mm256_storeu_ps(wrow+j,_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(a+j))),16)));
    memset(y,0,T*sizeof(float));
    // 4 columns of W at a time to cut passes over y
    size_t j=0;
    for(;j+4<=cols;j+=4){__m256 a0=_mm256_set1_ps(wrow[j]),a1=_mm256_set1_ps(wrow[j+1]),a2=_mm256_set1_ps(wrow[j+2]),a3=_mm256_set1_ps(wrow[j+3]);
      const float*x0=X+j*T,*x1=x0+T,*x2=x1+T,*x3=x2+T;
      for(size_t t=0;t+8<=T;t+=8){__m256 v=_mm256_loadu_ps(y+t);
        v=_mm256_fmadd_ps(a0,_mm256_loadu_ps(x0+t),v);v=_mm256_fmadd_ps(a1,_mm256_loadu_ps(x1+t),v);
        v=_mm256_fmadd_ps(a2,_mm256_loadu_ps(x2+t),v);v=_mm256_fmadd_ps(a3,_mm256_loadu_ps(x3+t),v);
        _mm256_storeu_ps(y+t,v);}}
  }
}
int main(int argc,char**argv){
  size_t rows=argc>1?strtoull(argv[1],0,10):4736, cols=argc>2?strtoull(argv[2],0,10):3584;
  int maxthreads=argc>3?atoi(argv[3]):omp_get_max_threads();
  size_t n=rows*cols;
  uint16_t*w=aligned_alloc(64,((n*2+63)/64)*64);double*x=aligned_alloc(64,cols*8);float*xf=aligned_alloc(64,cols*4);
  double*y=aligned_alloc(64,rows*8),*y2=aligned_alloc(64,rows*8);float*yf=aligned_alloc(64,rows*4);
  uint64_t z=0x9e3779b97f4a7c15ull;
  for(size_t i=0;i<n;++i){z^=z<<13;z^=z>>7;z^=z<<17;uint32_t e=118+(z>>60);w[i]=(uint16_t)(((z>>40)&0x8000u)|(e<<7)|(z&127));}
  for(size_t j=0;j<cols;++j){z^=z<<13;z^=z>>7;z^=z<<17;x[j]=((double)(z>>11)/9007199254740992.0-0.5)*4;xf[j]=(float)x[j];}
  printf("matrix %zux%zu = %.1f M weights, %.1f MB BF16; threads available %d\n",rows,cols,n/1e6,n*2/1e6,maxthreads);
  // A: tile-wise expand + ordered gemv, single thread, tile = 262144 elements like the runtime (row-aligned here)
  size_t tile_rows=262144/cols; if(!tile_rows)tile_rows=1; double*arena=aligned_alloc(64,tile_rows*cols*8);
  double te=0,tg=0;int reps=0;double t0=now();
  do{ for(size_t r=0;r<rows;r+=tile_rows){size_t m=r+tile_rows<=rows?tile_rows:rows-r;
        double a=now();expand((const unsigned char*)(w+r*cols),m*cols,arena);double b=now();
        linear_ordered(arena,m,cols,x,y+r);double c=now();te+=b-a;tg+=c-b;}
      ++reps;}while(now()-t0<1.0);
  printf("A repo-semantics 1T: expand %.2f Gw/s (%.2f GB/s BF16), ordered FP64 gemv %.2f Gw/s, combined %.2f Gw/s (%.2f GB/s BF16)\n",
    reps*n/te/1e9,reps*n*2/te/1e9,reps*n/tg/1e9,reps*n/(te+tg)/1e9,reps*n*2/(te+tg)/1e9);
  int tl[]={1,2,4,6,8,12,14,20};
  for(int k=0;k<8;++k){int T=tl[k];if(T>maxthreads)break;omp_set_num_threads(T);
    reps=0;t0=now();
    do{
#pragma omp parallel for schedule(static)
      for(size_t r=0;r<rows;++r)y2[r]=row_f64(w+r*cols,x,cols);
      ++reps;}while(now()-t0<0.7);
    double tb=now()-t0;int repb=reps;
    reps=0;t0=now();
    do{
#pragma omp parallel for schedule(static)
      for(size_t r=0;r<rows;++r)yf[r]=row_f32(w+r*cols,xf,cols);
      ++reps;}while(now()-t0<0.7);
    double tc=now()-t0;
    printf("T=%2d  B fused BF16->FP64 AVX2: %6.2f Gw/s (%6.2f GB/s BF16) | C fused BF16->FP32 AVX2: %6.2f Gw/s (%6.2f GB/s BF16)\n",
      T,repb*n/tb/1e9,repb*n*2/tb/1e9,reps*n/tc/1e9,reps*n*2/tc/1e9);}
  // numerical sanity: A vs B vs C
  double maxab=0,maxac=0,maxy=0;for(size_t r=0;r<rows;++r){double d=y[r]-y2[r];if(d<0)d=-d;if(d>maxab)maxab=d;d=y[r]-yf[r];if(d<0)d=-d;if(d>maxac)maxac=d;double v=y[r]<0?-y[r]:y[r];if(v>maxy)maxy=v;}
  printf("max|y|=%.3g  max|A-B|=%.3g (FP64 reorder)  max|A-C|=%.3g (FP32 accumulate)\n",maxy,maxab,maxac);
  // D: GEMM batched prefill, Tk tokens
  size_t Ts[]={16,64,256};
  for(int q=0;q<3;++q){size_t Tk=Ts[q];float*X=aligned_alloc(64,cols*Tk*4),*Y=aligned_alloc(64,rows*Tk*4);
    for(size_t i=0;i<cols*Tk;++i){z^=z<<13;z^=z>>7;z^=z<<17;X[i]=(float)(((double)(z>>11)/9007199254740992.0-0.5)*4);}
    int Tl[]={1,6,14,20};
    for(int k=0;k<4;++k){int T=Tl[k];if(T>maxthreads)break;omp_set_num_threads(T);reps=0;t0=now();
      do{
#pragma omp parallel
        {float*wrow=aligned_alloc(64,((cols*4+63)/64)*64);
#pragma omp for schedule(static)
          for(size_t r=0;r<rows;r+=8){size_t e=r+8<=rows?r+8:rows;gemm_rows_f32(w,r,e,cols,X,Tk,Y,wrow);}
          free(wrow);}
        ++reps;}while(now()-t0<0.7);
      double td=now()-t0;
      printf("D GEMM tokens=%3zu T=%2d: %.2f Gw/s weights (%.2f GB/s BF16), %.1f GFLOP/s\n",Tk,T,reps*n/td/1e9,reps*n*2/td/1e9,2.0*reps*n*Tk/td/1e9);}
    free(X);free(Y);}
  return 0;}