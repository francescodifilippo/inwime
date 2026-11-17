// Scratch micro-benchmark #2 (NOT part of the repository): bit-identical ordered-per-row kernel,
// vectorized ACROSS rows (each SIMD lane = one output row, strict increasing-column order, mul then add, no FMA).
#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
// reference: exact semantics of src/cpu_kernel.c (expand then ordered scalar)
__attribute__((noinline)) static void ref_rows(const uint16_t*w,size_t rows,size_t cols,const double*x,double*y){
  for(size_t r=0;r<rows;++r){double s=0.;const uint16_t*a=w+r*cols;
    for(size_t j=0;j<cols;++j){uint32_t b=(uint32_t)a[j]<<16;float v;memcpy(&v,&b,4);s+=(double)v*x[j];}
    y[r]=s;}}
static inline __m256d ld4(const uint16_t*p){ // 4 BF16 -> 4 doubles (exact)
  __m128i u=_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)p)),16);
  return _mm256_cvtps_pd(_mm_castsi128_ps(u));}
// 8 rows at once: two accumulators, each lane one row. Strictly ordered per row.
static void rows8(const uint16_t*w,size_t cols,const double*x,double*y){
  const uint16_t*r0=w,*r1=w+cols,*r2=w+2*cols,*r3=w+3*cols,*r4=w+4*cols,*r5=w+5*cols,*r6=w+6*cols,*r7=w+7*cols;
  __m256d sa=_mm256_setzero_pd(),sb=_mm256_setzero_pd();size_t j=0;
  for(;j+4<=cols;j+=4){
    __m256d a0=ld4(r0+j),a1=ld4(r1+j),a2=ld4(r2+j),a3=ld4(r3+j);
    __m256d b0=ld4(r4+j),b1=ld4(r5+j),b2=ld4(r6+j),b3=ld4(r7+j);
    // transpose 4x4: columns across rows
    __m256d t0=_mm256_unpacklo_pd(a0,a1),t1=_mm256_unpackhi_pd(a0,a1),t2=_mm256_unpacklo_pd(a2,a3),t3=_mm256_unpackhi_pd(a2,a3);
    __m256d c0=_mm256_permute2f128_pd(t0,t2,0x20),c1=_mm256_permute2f128_pd(t1,t3,0x20),c2=_mm256_permute2f128_pd(t0,t2,0x31),c3=_mm256_permute2f128_pd(t1,t3,0x31);
    __m256d u0=_mm256_unpacklo_pd(b0,b1),u1=_mm256_unpackhi_pd(b0,b1),u2=_mm256_unpacklo_pd(b2,b3),u3=_mm256_unpackhi_pd(b2,b3);
    __m256d d0=_mm256_permute2f128_pd(u0,u2,0x20),d1=_mm256_permute2f128_pd(u1,u3,0x20),d2=_mm256_permute2f128_pd(u0,u2,0x31),d3=_mm256_permute2f128_pd(u1,u3,0x31);
    __m256d x0=_mm256_broadcast_sd(x+j),x1=_mm256_broadcast_sd(x+j+1),x2=_mm256_broadcast_sd(x+j+2),x3=_mm256_broadcast_sd(x+j+3);
    sa=_mm256_add_pd(sa,_mm256_mul_pd(c0,x0));sb=_mm256_add_pd(sb,_mm256_mul_pd(d0,x0));
    sa=_mm256_add_pd(sa,_mm256_mul_pd(c1,x1));sb=_mm256_add_pd(sb,_mm256_mul_pd(d1,x1));
    sa=_mm256_add_pd(sa,_mm256_mul_pd(c2,x2));sb=_mm256_add_pd(sb,_mm256_mul_pd(d2,x2));
    sa=_mm256_add_pd(sa,_mm256_mul_pd(c3,x3));sb=_mm256_add_pd(sb,_mm256_mul_pd(d3,x3));}
  double s[8];_mm256_storeu_pd(s,sa);_mm256_storeu_pd(s+4,sb);
  const uint16_t*rr[8]={r0,r1,r2,r3,r4,r5,r6,r7};
  for(int k=0;k<8;++k){double t=s[k];for(size_t q=j;q<cols;++q){uint32_t b=(uint32_t)rr[k][q]<<16;float v;memcpy(&v,&b,4);t+=(double)v*x[q];}y[k]=t;}
}
// 16 rows at once (4 accumulators) to hide add latency further
static void rows16(const uint16_t*w,size_t cols,const double*x,double*y){
  __m256d s[4];for(int g=0;g<4;++g)s[g]=_mm256_setzero_pd();size_t j=0;
  for(;j+4<=cols;j+=4){
    __m256d x0=_mm256_broadcast_sd(x+j),x1=_mm256_broadcast_sd(x+j+1),x2=_mm256_broadcast_sd(x+j+2),x3=_mm256_broadcast_sd(x+j+3);
    __m256d c[4][4];
    for(int g=0;g<4;++g){const uint16_t*b=w+(size_t)g*4*cols+j;
      __m256d a0=ld4(b),a1=ld4(b+cols),a2=ld4(b+2*cols),a3=ld4(b+3*cols);
      __m256d t0=_mm256_unpacklo_pd(a0,a1),t1=_mm256_unpackhi_pd(a0,a1),t2=_mm256_unpacklo_pd(a2,a3),t3=_mm256_unpackhi_pd(a2,a3);
      c[g][0]=_mm256_permute2f128_pd(t0,t2,0x20);c[g][1]=_mm256_permute2f128_pd(t1,t3,0x20);c[g][2]=_mm256_permute2f128_pd(t0,t2,0x31);c[g][3]=_mm256_permute2f128_pd(t1,t3,0x31);}
    for(int g=0;g<4;++g)s[g]=_mm256_add_pd(s[g],_mm256_mul_pd(c[g][0],x0));
    for(int g=0;g<4;++g)s[g]=_mm256_add_pd(s[g],_mm256_mul_pd(c[g][1],x1));
    for(int g=0;g<4;++g)s[g]=_mm256_add_pd(s[g],_mm256_mul_pd(c[g][2],x2));
    for(int g=0;g<4;++g)s[g]=_mm256_add_pd(s[g],_mm256_mul_pd(c[g][3],x3));}
  double t[16];for(int g=0;g<4;++g)_mm256_storeu_pd(t+4*g,s[g]);
  for(int k=0;k<16;++k){double v=t[k];const uint16_t*r=w+(size_t)k*cols;for(size_t q=j;q<cols;++q){uint32_t b=(uint32_t)r[q]<<16;float f;memcpy(&f,&b,4);v+=(double)f*x[q];}y[k]=v;}
}
int main(int argc,char**argv){
  size_t rows=argc>1?strtoull(argv[1],0,10):18944, cols=argc>2?strtoull(argv[2],0,10):3584;int maxthreads=omp_get_max_threads();
  size_t n=rows*cols;uint16_t*w=aligned_alloc(64,((n*2+63)/64)*64);double*x=aligned_alloc(64,cols*8),*y=aligned_alloc(64,rows*8),*y8=aligned_alloc(64,rows*8),*y16=aligned_alloc(64,rows*8);
  uint64_t z=0x9e3779b97f4a7c15ull;
  for(size_t i=0;i<n;++i){z^=z<<13;z^=z>>7;z^=z<<17;uint32_t e=118+(z>>60);w[i]=(uint16_t)(((z>>40)&0x8000u)|(e<<7)|(z&127));}
  for(size_t j=0;j<cols;++j){z^=z<<13;z^=z>>7;z^=z<<17;x[j]=((double)(z>>11)/9007199254740992.0-0.5)*4;}
  printf("matrix %zux%zu = %.1f M weights (%.1f MB BF16)\n",rows,cols,n/1e6,n*2/1e6);
  double t0=now();int reps=0;do{ref_rows(w,rows,cols,x,y);++reps;}while(now()-t0<1.0);double tr=now()-t0;
  printf("REF fused scalar ordered 1T: %.2f Gw/s\n",reps*n/tr/1e9);
  int tl[]={1,2,4,6,8,12,14};
  for(int k=0;k<7;++k){int T=tl[k];if(T>maxthreads)break;omp_set_num_threads(T);
    reps=0;t0=now();do{
#pragma omp parallel for schedule(static)
      for(size_t r=0;r<rows/8;++r)rows8(w+r*8*cols,cols,x,y8+r*8);
      ++reps;}while(now()-t0<0.7);double t8=now()-t0;int r8=reps;
    reps=0;t0=now();do{
#pragma omp parallel for schedule(static)
      for(size_t r=0;r<rows/16;++r)rows16(w+r*16*cols,cols,x,y16+r*16);
      ++reps;}while(now()-t0<0.7);double t16=now()-t0;
    printf("T=%2d  E8 ordered-per-row x8 rows: %6.2f Gw/s (%6.2f GB/s BF16) | E16 x16 rows: %6.2f Gw/s (%6.2f GB/s BF16)\n",T,r8*n/t8/1e9,r8*n*2/t8/1e9,reps*n/t16/1e9,reps*n*2/t16/1e9);}
  size_t m8=(rows/8)*8,m16=(rows/16)*16;
  printf("bit-identical to ordered scalar reference: E8 %s, E16 %s\n",memcmp(y,y8,m8*8)?"NO":"YES",memcmp(y,y16,m16*8)?"NO":"YES");
  return 0;}