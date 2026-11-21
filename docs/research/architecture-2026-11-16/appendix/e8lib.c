// Scratch (NOT part of the repository): 8-row ordered-per-row fused BF16->FP64 GEMV as a shared library,
// to compare bit-for-bit against the repository's compiled kernel build/libinwime_cpu.so.
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
static inline __m256d ld4(const uint16_t*p){
  __m128i u=_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*)p)),16);
  return _mm256_cvtps_pd(_mm_castsi128_ps(u));}
static void rows8(const uint16_t*w,size_t cols,const double*x,double*y){
  const uint16_t*r[8];for(int k=0;k<8;++k)r[k]=w+(size_t)k*cols;
  __m256d sa=_mm256_setzero_pd(),sb=_mm256_setzero_pd();size_t j=0;
  for(;j+4<=cols;j+=4){
    __m256d a0=ld4(r[0]+j),a1=ld4(r[1]+j),a2=ld4(r[2]+j),a3=ld4(r[3]+j);
    __m256d b0=ld4(r[4]+j),b1=ld4(r[5]+j),b2=ld4(r[6]+j),b3=ld4(r[7]+j);
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
  for(int k=0;k<8;++k){double t=s[k];for(size_t q=j;q<cols;++q){uint32_t b=(uint32_t)r[k][q]<<16;float v;memcpy(&v,&b,4);t+=(double)v*x[q];}y[k]=t;}
}
void e8_gemv(const uint16_t*w,size_t rows,size_t cols,const double*x,double*y){
  size_t r=0;for(;r+8<=rows;r+=8)rows8(w+r*cols,cols,x,y+r);
  for(;r<rows;++r){double s=0.;const uint16_t*a=w+r*cols;for(size_t j=0;j<cols;++j){uint32_t b=(uint32_t)a[j]<<16;float v;memcpy(&v,&b,4);s+=(double)v*x[j];}y[r]=s;}
}