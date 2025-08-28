import os
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define iDivUp(a,b) (((a)+(b)-1)/(b))

// --- Sigmoid helpers ---
__device__ __forceinline__ float sig(float z){ return 1.f / (1.f + __expf(-z)); }
__device__ __forceinline__ float sigp_from_s(float s){ return s*(1.f - s); }  // σ'(z)=s(1-s)

// ---- 2D ECC with sigmoid gates (center c, 8-neighborhood) ----
struct ECC2D {
  // returns change and fills per-neighbor gate values + derivatives if requested
  __device__ static float change_and_partials(
      const float c, const float t, const float b, const float l, const float r,
      const float tl, const float tr, const float bl, const float br,
      const float a,
      // outputs:
      float* dC_dc,        // d_change/d_c
      float* dC_dt, float* dC_db, float* dC_dl, float* dC_dr,
      float* dC_dtl, float* dC_dtr, float* dC_dbl, float* dC_dbr)
  {
    // Face gates (T,B,L,R) and corners (TL,TR,BL,BR)
    const float T  = sig(a*(t  - c));
    const float B  = sig(a*(b  - c));
    const float L  = sig(a*(l  - c));
    const float R  = sig(a*(r  - c));
    const float TL = sig(a*(tl - c));
    const float TR = sig(a*(tr - c));
    const float BL = sig(a*(bl - c));
    const float BR = sig(a*(br - c));

    // Derivatives wrt c (negative) and wrt neighbor (positive)
    const float dTdc  = -a * sigp_from_s(T),  dBdc  = -a * sigp_from_s(B);
    const float dLdc  = -a * sigp_from_s(L),  dRdc  = -a * sigp_from_s(R);
    const float dTLdc = -a * sigp_from_s(TL), dTRdc = -a * sigp_from_s(TR);
    const float dBLdc = -a * sigp_from_s(BL), dBRdc = -a * sigp_from_s(BR);

    const float dTdt  =  a * sigp_from_s(T),  dBdb  =  a * sigp_from_s(B);
    const float dLdl  =  a * sigp_from_s(L),  dRdr  =  a * sigp_from_s(R);
    const float dTLdtl=  a * sigp_from_s(TL), dTRdtr=  a * sigp_from_s(TR);
    const float dBLdbl=  a * sigp_from_s(BL), dBRdbr=  a * sigp_from_s(BR);

    // 2D Euler increment (sigmoid-smoothed)
    const float change = 1.f
      - T - B - L - R
      + T*L*TL + T*R*TR + B*L*BL + B*R*BR;

    // d_change/d_c
    if (dC_dc){
      float dC = 0.f;
      dC += -dTdc - dBdc - dLdc - dRdc;
      dC += dTdc*L*TL + T*dLdc*TL + T*L*dTLdc;
      dC += dTdc*R*TR + T*dRdc*TR + T*R*dTRdc;
      dC += dBdc*L*BL + B*dLdc*BL + B*L*dBLdc;
      dC += dBdc*R*BR + B*dRdc*BR + B*R*dBRdc;
      *dC_dc = dC;
    }

    // d_change/d_neighbors
    if (dC_dt)  *dC_dt  = -dTdt  + dTdt*L*TL + dTdt*R*TR;
    if (dC_db)  *dC_db  = -dBdb  + dBdb*L*BL + dBdb*R*BR;
    if (dC_dl)  *dC_dl  = -dLdl  + T*dLdl*TL + B*dLdl*BL;
    if (dC_dr)  *dC_dr  = -dRdr  + T*dRdr*TR + B*dRdr*BR;
    if (dC_dtl) *dC_dtl =  T*L*dTLdtl;
    if (dC_dtr) *dC_dtr =  T*R*dTRdtr;
    if (dC_dbl) *dC_dbl =  B*L*dBLdbl;
    if (dC_dbr) *dC_dbr =  B*R*dBRdbr;

    return change;
  }
};

// ---- 3D ECC "change" (center c, 6 faces, 12 edges, 8 corners) ----
struct ECC3D {
  // neighbor indexing helpers
  __device__ static float change_and_partials(
      const float c,
      // 6 faces: x- x+ y- y+ z- z+ (L,R,T,D,F,B) in this order
      const float fL, const float fR, const float fT, const float fD, const float fF, const float fB,
      // 12 edge voxels: (L,T),(R,T),(L,D),(R,D), (L,F),(R,F),(L,B),(R,B), (T,F),(T,B),(D,F),(D,B)
      const float eLT, const float eRT, const float eLD, const float eRD,
      const float eLF, const float eRF, const float eLB, const float eRB,
      const float eTF, const float eTB, const float eDF, const float eDB,
      // 8 corner voxels: (L,T,F),(R,T,F),(L,T,B),(R,T,B),(L,D,F),(R,D,F),(L,D,B),(R,D,B)
      const float cLTF, const float cRTF, const float cLTB, const float cRTB,
      const float cLDF, const float cRDF, const float cLDB, const float cRDB,
      const float a,
      // outputs (27 partials): center + 6 faces + 12 edges + 8 corners
      float* dC_dc,
      float d_faces[6], float d_edges[12], float d_corners[8])
  {
    // Face gates sigmoid(a*(n-c))
    float F[6] = {
      sig(a*(fL - c)), sig(a*(fR - c)), sig(a*(fT - c)), sig(a*(fD - c)), sig(a*(fF - c)), sig(a*(fB - c))
    };
    float dFdc[6] = {
      -a*sigp_from_s(F[0]), -a*sigp_from_s(F[1]), -a*sigp_from_s(F[2]),
      -a*sigp_from_s(F[3]), -a*sigp_from_s(F[4]), -a*sigp_from_s(F[5])
    };
    float dFdn[6] = {
       a*sigp_from_s(F[0]),  a*sigp_from_s(F[1]),  a*sigp_from_s(F[2]),
       a*sigp_from_s(F[3]),  a*sigp_from_s(F[4]),  a*sigp_from_s(F[5])
    };

    // Edge gates
    float E[12] = {
      sig(a*(eLT - c)), sig(a*(eRT - c)), sig(a*(eLD - c)), sig(a*(eRD - c)),
      sig(a*(eLF - c)), sig(a*(eRF - c)), sig(a*(eLB - c)), sig(a*(eRB - c)),
      sig(a*(eTF - c)), sig(a*(eTB - c)), sig(a*(eDF - c)), sig(a*(eDB - c))
    };
    float dEdc[12], dEdn[12];
    #pragma unroll
    for (int i=0;i<12;++i){ dEdc[i] = -a*sigp_from_s(E[i]); dEdn[i] =  a*sigp_from_s(E[i]); }

    // Corner gates
    float C[8] = {
      sig(a*(cLTF - c)), sig(a*(cRTF - c)), sig(a*(cLTB - c)), sig(a*(cRTB - c)),
      sig(a*(cLDF - c)), sig(a*(cRDF - c)), sig(a*(cLDB - c)), sig(a*(cRDB - c))
    };
    float dCdc_[8], dCdn_[8];
    #pragma unroll
    for (int i=0;i<8;++i){ dCdc_[i] = -a*sigp_from_s(C[i]); dCdn_[i] =  a*sigp_from_s(C[i]); }

    // change = +1 - sum faces + sum edges(Fi*Fj*E) - sum corners(Fx*Fy*Fz*C)
    float change = 1.f;
    #pragma unroll
    for (int i=0;i<6;++i) change -= F[i];

    // Edge topology: pairs of faces participating per edge (indices into F[])
    const int edge_f0[12] = {0,1,0,1, 0,1,0,1, 2,2,3,3}; // L/R with T/T/D/D, then L/R with F/F/B/B, then T/T/D/D with F/B/F/B
    const int edge_f1[12] = {2,2,3,3, 4,4,5,5, 4,5,4,5};

    // Corner topology: triplets of faces per corner
    const int corn_fx[8] = {0,1,0,1, 0,1,0,1};  // L/R
    const int corn_fy[8] = {2,2,2,2, 3,3,3,3};  // T/D
    const int corn_fz[8] = {4,4,5,5, 4,4,5,5};  // F/B

    #pragma unroll
    for (int e=0;e<12;++e){
      change += F[edge_f0[e]] * F[edge_f1[e]] * E[e];
    }
    #pragma unroll
    for (int cidx=0;cidx<8;++cidx){
      change -= F[corn_fx[cidx]] * F[corn_fy[cidx]] * F[corn_fz[cidx]] * C[cidx];
    }

    // Center partial
    if (dC_dc){
      float dC = 0.f;
      #pragma unroll
      for (int i=0;i<6;++i) dC -= dFdc[i];
      #pragma unroll
      for (int e=0;e<12;++e){
        int i = edge_f0[e], j = edge_f1[e];
        dC += dFdc[i]*F[j]*E[e] + F[i]*dFdc[j]*E[e] + F[i]*F[j]*dEdc[e];
      }
      #pragma unroll
      for (int ci=0;ci<8;++ci){
        int ix = corn_fx[ci], iy = corn_fy[ci], iz = corn_fz[ci];
        dC -= dFdc[ix]*F[iy]*F[iz]*C[ci] + F[ix]*dFdc[iy]*F[iz]*C[ci]
            + F[ix]*F[iy]*dFdc[iz]*C[ci] + F[ix]*F[iy]*F[iz]*dCdc_[ci];
      }
      *dC_dc = dC;
    }

    // Face partials
    if (d_faces){
      #pragma unroll
      for (int i=0;i<6;++i) d_faces[i] = -dFdn[i];
      #pragma unroll
      for (int e=0;e<12;++e){
        int i = edge_f0[e], j = edge_f1[e];
        d_faces[i] += dFdn[i]*F[j]*E[e];
        d_faces[j] += dFdn[j]*F[i]*E[e];
      }
      #pragma unroll
      for (int ci=0;ci<8;++ci){
        int ix = corn_fx[ci], iy = corn_fy[ci], iz = corn_fz[ci];
        float prod_others = F[iy]*F[iz]*C[ci];
        d_faces[ix] -= dFdn[ix]*prod_others;
        prod_others = F[ix]*F[iz]*C[ci];
        d_faces[iy] -= dFdn[iy]*prod_others;
        prod_others = F[ix]*F[iy]*C[ci];
        d_faces[iz] -= dFdn[iz]*prod_others;
      }
    }

    // Edge partials
    if (d_edges){
      #pragma unroll
      for (int e=0;e<12;++e){
        int i = edge_f0[e], j = edge_f1[e];
        d_edges[e] = F[i]*F[j]*dEdn[e];
      }
    }

    // Corner partials
    if (d_corners){
      #pragma unroll
      for (int ci=0;ci<8;++ci){
        int ix = corn_fx[ci], iy = corn_fy[ci], iz = corn_fz[ci];
        d_corners[ci] = - F[ix]*F[iy]*F[iz]*dCdn_[ci];
      }
    }

    return change;
  }
};

// ---- soft-binning helpers (shared by 2D & 3D) ----
// Given thresholds[0..T-1], center c, alpha a,
// compute (1) scalar g_scalar = Σ_k gh[k]*(s_k - s_{k-1})
//         (2) c_term = -a*change*Σ_k (gh[k]-gh[k+1]) * σ'(a(τ_k - c))
//         (3) grad_tau[k] += change * a * σ'(a(τ_k - c)) * (gh[k] - gh[k+1])
// Also return sscan if you need forward weights (w_k = s_k - s_{k-1})
template <bool DO_TAU_GRAD>
__device__ __forceinline__
void softbin_accumulate(
    const float* __restrict__ thr, int T, float c, float a,
    const float* __restrict__ gh, // grad_hist
    float change,
    float* __restrict__ out_g_scalar,
    float* __restrict__ out_c_term,
    float* __restrict__ gthr // size T (may be null if DO_TAU_GRAD=false)
){
  float gsum = 0.f;
  float sum_pdiff = 0.f;
  float s_prev = 0.f;
  #pragma unroll 1
  for (int k=0;k<T;++k){
    const float dk = a*(thr[k] - c);
    const float s  = sig(dk);
    const float p  = sigp_from_s(s);
    const float gh_k = gh[k];
    const float gh_next = (k+1<T) ? gh[k+1] : 0.f;
    const float w = s - s_prev;

    gsum += gh_k * w;
    const float gh_diff = gh_k - gh_next;
    sum_pdiff += gh_diff * p;

    if constexpr (DO_TAU_GRAD){
      if (gthr) atomicAdd(&gthr[k], change * a * p * gh_diff);
    }
    s_prev = s;
  }
  *out_g_scalar = gsum;
  *out_c_term   = -a * change * sum_pdiff;
}

// ------------------------- KERNELS ----------------------------

// Forward 2D
__global__ void ecc2d_forward_kernel(
  const float* __restrict__ img, int H, int W,
  const float* __restrict__ thr, int T, float a,
  float* __restrict__ hist)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= W || iy >= H) return;

  const int idx = iy*W + ix;
  const float c = img[idx];

  // neighbors with +/-INF outside
  const float INF = INFINITY, NINF = -INFINITY;
  const float t  = (iy>0   ) ? img[idx - W]     : INF;
  const float b  = (iy+1<H ) ? img[idx + W]     : NINF;
  const float l  = (ix>0   ) ? img[idx - 1]     : INF;
  const float r  = (ix+1<W ) ? img[idx + 1]     : NINF;
  const float tl = (ix>0 && iy>0          ) ? img[idx - W - 1] : INF;
  const float tr = (ix+1<W && iy>0        ) ? img[idx - W + 1] : INF;
  const float bl = (ix>0 && iy+1<H        ) ? img[idx + W - 1] : NINF;
  const float br = (ix+1<W && iy+1<H      ) ? img[idx + W + 1] : NINF;

  float dC_dc;
  float change = ECC2D::change_and_partials(
      c,t,b,l,r,tl,tr,bl,br,a,
      &dC_dc, nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr);

  // Accumulate hist: w_k = σ(a(τ_k - c)) - σ(a(τ_{k-1} - c))
  float s_prev=0.f;
  for (int k=0;k<T;++k){
    float s = sig(a*(thr[k]-c));
    float w = s - s_prev;
    if (w != 0.f) atomicAdd(&hist[k], change * w);
    s_prev = s;
  }
}

// Backward 2D -> returns grad_img (H,W) and grad_thr (T)
__global__ void ecc2d_backward_kernel(
  const float* __restrict__ img, int H, int W,
  const float* __restrict__ thr, int T, float a,
  const float* __restrict__ gh, // grad_hist[T]
  float* __restrict__ gimg,     // (H,W)
  float* __restrict__ gthr)     // (T)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ix >= W || iy >= H) return;

  const int idx = iy*W + ix;
  const float c = img[idx];

  const float INF = INFINITY, NINF = -INFINITY;
  const float t  = (iy>0   ) ? img[idx - W]     : INF;
  const float b  = (iy+1<H ) ? img[idx + W]     : NINF;
  const float l  = (ix>0   ) ? img[idx - 1]     : INF;
  const float r  = (ix+1<W ) ? img[idx + 1]     : NINF;
  const float tl = (ix>0 && iy>0          ) ? img[idx - W - 1] : INF;
  const float tr = (ix+1<W && iy>0        ) ? img[idx - W + 1] : INF;
  const float bl = (ix>0 && iy+1<H        ) ? img[idx + W - 1] : NINF;
  const float br = (ix+1<W && iy+1<H      ) ? img[idx + W + 1] : NINF;

  float dC_dc, dT,dB,dL,dR,dTL,dTR,dBL,dBR;
  const float change = ECC2D::change_and_partials(
      c,t,b,l,r,tl,tr,bl,br,a,
      &dC_dc, &dT,&dB,&dL,&dR,&dTL,&dTR,&dBL,&dBR);

  float g_scalar=0.f, c_term=0.f;
  softbin_accumulate<true>(thr, T, c, a, gh, change, &g_scalar, &c_term, gthr);

  // center and neighbors
  atomicAdd(&gimg[idx],                 g_scalar*dC_dc + c_term);
  if (iy>0)        atomicAdd(&gimg[idx - W],     g_scalar*dT);
  if (iy+1<H)      atomicAdd(&gimg[idx + W],     g_scalar*dB);
  if (ix>0)        atomicAdd(&gimg[idx - 1],     g_scalar*dL);
  if (ix+1<W)      atomicAdd(&gimg[idx + 1],     g_scalar*dR);
  if (ix>0 && iy>0)              atomicAdd(&gimg[idx - W - 1], g_scalar*dTL);
  if (ix+1<W && iy>0)            atomicAdd(&gimg[idx - W + 1], g_scalar*dTR);
  if (ix>0 && iy+1<H)            atomicAdd(&gimg[idx + W - 1], g_scalar*dBL);
  if (ix+1<W && iy+1<H)          atomicAdd(&gimg[idx + W + 1], g_scalar*dBR);
}

// Forward 3D
__global__ void ecc3d_forward_kernel(
  const float* __restrict__ vol, int D, int H, int W,
  const float* __restrict__ thr, int T, float a,
  float* __restrict__ hist)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix>=W || iy>=H || iz>=D) return;

  const int strideHW = H*W;
  const int idx = iz*strideHW + iy*W + ix;
  const float c = vol[idx];

  const float INF = INFINITY, NINF = -INFINITY;

  auto at = [&](int x,int y,int z)->float{
    if (x<0||x>=W||y<0||y>=H||z<0||z>=D) return (y<0||x<0||z<0) ? INF : NINF;
    return vol[z*strideHW + y*W + x];
  };

  // faces
  float fL = at(ix-1,iy,iz), fR = at(ix+1,iy,iz);
  float fT = at(ix,iy-1,iz), fD = at(ix,iy+1,iz);
  float fF = at(ix,iy,iz-1), fB = at(ix,iy,iz+1);

  // edges
  float eLT=at(ix-1,iy-1,iz), eRT=at(ix+1,iy-1,iz), eLD=at(ix-1,iy+1,iz), eRD=at(ix+1,iy+1,iz);
  float eLF=at(ix-1,iy,iz-1), eRF=at(ix+1,iy,iz-1), eLB=at(ix-1,iy,iz+1), eRB=at(ix+1,iy,iz+1);
  float eTF=at(ix,iy-1,iz-1), eTB=at(ix,iy-1,iz+1), eDF=at(ix,iy+1,iz-1), eDB=at(ix,iy+1,iz+1);

  // corners
  float cLTF=at(ix-1,iy-1,iz-1), cRTF=at(ix+1,iy-1,iz-1),
        cLTB=at(ix-1,iy-1,iz+1), cRTB=at(ix+1,iy-1,iz+1),
        cLDF=at(ix-1,iy+1,iz-1), cRDF=at(ix+1,iy+1,iz-1),
        cLDB=at(ix-1,iy+1,iz+1), cRDB=at(ix+1,iy+1,iz+1);

  float dCdc;
  float change = ECC3D::change_and_partials(
    c, fL,fR,fT,fD,fF,fB,
    eLT,eRT,eLD,eRD, eLF,eRF,eLB,eRB, eTF,eTB,eDF,eDB,
    cLTF,cRTF,cLTB,cRTB,cLDF,cRDF,cLDB,cRDB,
    a, &dCdc, nullptr,nullptr,nullptr);

  float s_prev=0.f;
  for (int k=0;k<T;++k){
    float s = sig(a*(thr[k]-c));
    float w = s - s_prev;
    if (w!=0.f) atomicAdd(&hist[k], change * w);
    s_prev = s;
  }
}

// Backward 3D
__global__ void ecc3d_backward_kernel(
  const float* __restrict__ vol, int D, int H, int W,
  const float* __restrict__ thr, int T, float a,
  const float* __restrict__ gh,
  float* __restrict__ gvol,
  float* __restrict__ gthr)
{
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  int iz = blockIdx.z * blockDim.z + threadIdx.z;
  if (ix>=W || iy>=H || iz>=D) return;

  const int strideHW = H*W;
  const int idx = iz*strideHW + iy*W + ix;
  const float c = vol[idx];

  const float INF = INFINITY, NINF = -INFINITY;
  auto at = [&](int x,int y,int z)->float{
    if (x<0||x>=W||y<0||y>=H||z<0||z>=D) return (y<0||x<0||z<0) ? INF : NINF;
    return vol[z*strideHW + y*W + x];
  };

  // faces
  float fL = at(ix-1,iy,iz), fR = at(ix+1,iy,iz);
  float fT = at(ix,iy-1,iz), fD = at(ix,iy+1,iz);
  float fF = at(ix,iy,iz-1), fB = at(ix,iy,iz+1);

  // edges
  float eLT=at(ix-1,iy-1,iz), eRT=at(ix+1,iy-1,iz), eLD=at(ix-1,iy+1,iz), eRD=at(ix+1,iy+1,iz);
  float eLF=at(ix-1,iy,iz-1), eRF=at(ix+1,iy,iz-1), eLB=at(ix-1,iy,iz+1), eRB=at(ix+1,iy,iz+1);
  float eTF=at(ix,iy-1,iz-1), eTB=at(ix,iy-1,iz+1), eDF=at(ix,iy+1,iz-1), eDB=at(ix,iy+1,iz+1);

  // corners
  float cLTF=at(ix-1,iy-1,iz-1), cRTF=at(ix+1,iy-1,iz-1),
        cLTB=at(ix-1,iy-1,iz+1), cRTB=at(ix+1,iy-1,iz+1),
        cLDF=at(ix-1,iy+1,iz-1), cRDF=at(ix+1,iy+1,iz-1),
        cLDB=at(ix-1,iy+1,iz+1), cRDB=at(ix+1,iy+1,iz+1);

  float d_faces[6], d_edges[12], d_corners[8], dCdc;
  const float change = ECC3D::change_and_partials(
    c, fL,fR,fT,fD,fF,fB,
    eLT,eRT,eLD,eRD, eLF,eRF,eLB,eRB, eTF,eTB,eDF,eDB,
    cLTF,cRTF,cLTB,cRTB,cLDF,cRDF,cLDB,cRDB,
    a, &dCdc, d_faces, d_edges, d_corners);

  float g_scalar=0.f, c_term=0.f;
  softbin_accumulate<true>(thr, T, c, a, gh, change, &g_scalar, &c_term, gthr);

  // scatter center
  atomicAdd(&gvol[idx], g_scalar*dCdc + c_term);

  // face offsets
  auto off = [&](int dx,int dy,int dz)->int{
    int x=ix+dx, y=iy+dy, z=iz+dz;
    if (x<0||x>=W||y<0||y>=H||z<0||z>=D) return -1;
    return z*strideHW + y*W + x;
  };

  const int fOff[6][3] = {{-1,0,0},{+1,0,0},{0,-1,0},{0,+1,0},{0,0,-1},{0,0,+1}};
  for (int i=0;i<6;++i){
    int j = off(fOff[i][0],fOff[i][1],fOff[i][2]);
    if (j>=0) atomicAdd(&gvol[j], g_scalar * d_faces[i]);
  }

  // edges
  const int eOff[12][3] = {{-1,-1,0},{+1,-1,0},{-1,+1,0},{+1,+1,0},
                           {-1,0,-1},{+1,0,-1},{-1,0,+1},{+1,0,+1},
                           {0,-1,-1},{0,-1,+1},{0,+1,-1},{0,+1,+1}};
  for (int e=0;e<12;++e){
    int j = off(eOff[e][0],eOff[e][1],eOff[e][2]);
    if (j>=0) atomicAdd(&gvol[j], g_scalar * d_edges[e]);
  }

  // corners
  const int cOff[8][3] = {{-1,-1,-1},{+1,-1,-1},{-1,-1,+1},{+1,-1,+1},
                          {-1,+1,-1},{+1,+1,-1},{-1,+1,+1},{+1,+1,+1}};
  for (int ci=0;ci<8;++ci){
    int j = off(cOff[ci][0],cOff[ci][1],cOff[ci][2]);
    if (j>=0) atomicAdd(&gvol[j], g_scalar * d_corners[ci]);
  }
}

// ------------------ Host wrappers --------------------
torch::Tensor ecc2d_forward(torch::Tensor img, torch::Tensor thr, double alpha){
  TORCH_CHECK(img.is_cuda() && thr.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(img.dtype()==torch::kFloat32 && thr.dtype()==torch::kFloat32, "float32 only");
  TORCH_CHECK(img.dim()==2, "img must be [H,W]");
  const int H = img.size(0), W = img.size(1), T = thr.size(0);
  auto hist = torch::zeros({T}, img.options());
  dim3 block(64, 8);
  dim3 grid(iDivUp(W, block.x), iDivUp(H, block.y));
  auto stream = at::cuda::getCurrentCUDAStream();
  ecc2d_forward_kernel<<<grid, block, 0, stream>>>(
      img.data_ptr<float>(), H, W,
      thr.data_ptr<float>(), T, (float)alpha,
      hist.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return hist;
}

std::vector<torch::Tensor> ecc2d_backward(torch::Tensor img, torch::Tensor thr,
                                          torch::Tensor gh, double alpha){
  TORCH_CHECK(img.is_cuda() && thr.is_cuda() && gh.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(img.dtype()==torch::kFloat32 && thr.dtype()==torch::kFloat32 && gh.dtype()==torch::kFloat32, "float32 only");
  TORCH_CHECK(img.dim()==2, "img must be [H,W]");
  const int H = img.size(0), W = img.size(1), T = thr.size(0);
  auto gimg = torch::zeros_like(img);
  auto gthr = torch::zeros_like(thr);
  dim3 block(64, 8);
  dim3 grid(iDivUp(W, block.x), iDivUp(H, block.y));
  auto stream = at::cuda::getCurrentCUDAStream();
  ecc2d_backward_kernel<<<grid, block, 0, stream>>>(
      img.data_ptr<float>(), H, W,
      thr.data_ptr<float>(), T, (float)alpha,
      gh.data_ptr<float>(),
      gimg.data_ptr<float>(), gthr.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gimg, gthr};
}

torch::Tensor ecc3d_forward(torch::Tensor vol, torch::Tensor thr, double alpha){
  TORCH_CHECK(vol.is_cuda() && thr.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(vol.dtype()==torch::kFloat32 && thr.dtype()==torch::kFloat32, "float32 only");
  TORCH_CHECK(vol.dim()==3, "vol must be [D,H,W]");
  const int D = vol.size(0), H = vol.size(1), W = vol.size(2), T = thr.size(0);
  auto hist = torch::zeros({T}, vol.options());
  dim3 block(64,4,2);
  dim3 grid(iDivUp(W, block.x), iDivUp(H, block.y), iDivUp(D, block.z));
  auto stream = at::cuda::getCurrentCUDAStream();
  ecc3d_forward_kernel<<<grid, block, 0, stream>>>(
      vol.data_ptr<float>(), D,H,W, thr.data_ptr<float>(), T, (float)alpha, hist.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return hist;
}

std::vector<torch::Tensor> ecc3d_backward(torch::Tensor vol, torch::Tensor thr,
                                          torch::Tensor gh, double alpha){
  TORCH_CHECK(vol.is_cuda() && thr.is_cuda() && gh.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(vol.dtype()==torch::kFloat32 && thr.dtype()==torch::kFloat32 && gh.dtype()==torch::kFloat32, "float32 only");
  TORCH_CHECK(vol.dim()==3, "vol must be [D,H,W]");
  const int D = vol.size(0), H = vol.size(1), W = vol.size(2), T = thr.size(0);
  auto gvol = torch::zeros_like(vol);
  auto gthr = torch::zeros_like(thr);
  dim3 block(64,4,2);
  dim3 grid(iDivUp(W, block.x), iDivUp(H, block.y), iDivUp(D, block.z));
  auto stream = at::cuda::getCurrentCUDAStream();
  ecc3d_backward_kernel<<<grid, block, 0, stream>>>(
      vol.data_ptr<float>(), D,H,W, thr.data_ptr<float>(), T, (float)alpha,
      gh.data_ptr<float>(), gvol.data_ptr<float>(), gthr.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gvol, gthr};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("ecc2d_forward", &ecc2d_forward, "ECC 2D sigmoid soft-binning forward");
  m.def("ecc2d_backward",&ecc2d_backward,"ECC 2D sigmoid soft-binning backward");
  m.def("ecc3d_forward", &ecc3d_forward, "ECC 3D sigmoid soft-binning forward");
  m.def("ecc3d_backward",&ecc3d_backward,"ECC 3D sigmoid soft-binning backward");
}
''';

ecc = load_inline(
    name="ecc_sigsoftbin",
    cpp_sources="",
    cuda_sources=cuda_src,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    verbose=False,
)


# -------- PyTorch Autograd wrappers (Python side) --------
class ECC2DSoftBin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img: torch.Tensor, thr: torch.Tensor, alpha: float = 16.0):
        hist = ecc.ecc2d_forward(img, thr, float(alpha))
        ctx.save_for_backward(img, thr)
        ctx.alpha = float(alpha)
        return hist

    @staticmethod
    def backward(ctx, grad_hist: torch.Tensor):
        img, thr = ctx.saved_tensors
        grad_hist = grad_hist.contiguous().to(torch.float32)
        gimg, gthr = ecc.ecc2d_backward(img, thr, grad_hist, ctx.alpha)
        return gimg, gthr, None


def ecc2d_soft(img: torch.Tensor, thr: torch.Tensor, alpha: float = 16.0):
    return ECC2DSoftBin.apply(img.contiguous().to(torch.float32),
                              thr.contiguous().to(torch.float32),
                              alpha)


class ECC3DSoftBin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vol: torch.Tensor, thr: torch.Tensor, alpha: float = 16.0):
        hist = ecc.ecc3d_forward(vol, thr, float(alpha))
        ctx.save_for_backward(vol, thr)
        ctx.alpha = float(alpha)
        return hist

    @staticmethod
    def backward(ctx, grad_hist: torch.Tensor):
        vol, thr = ctx.saved_tensors
        grad_hist = grad_hist.contiguous().to(torch.float32)
        gvol, gthr = ecc.ecc3d_backward(vol, thr, grad_hist, ctx.alpha)
        return gvol, gthr, None


def ecc3d_soft(vol: torch.Tensor, thr: torch.Tensor, alpha: float = 16.0):
    return ECC3DSoftBin.apply(vol.contiguous().to(torch.float32),
                              thr.contiguous().to(torch.float32),
                              alpha)


if __name__ == "__main__":
    torch.manual_seed(0)
    H, W = 128, 128
    img = torch.randn(H, W, device="cuda", dtype=torch.float32, requires_grad=True)
    thr = torch.linspace(-1, 1, steps=129, device="cuda", dtype=torch.float32).clone().requires_grad_(True)
    hist2d = ecc2d_soft(img, thr, alpha=16.0)
    loss = hist2d.square().mean()
    loss.backward()
    print("2D:", hist2d.shape, img.grad.abs().mean().item(), thr.grad.abs().mean().item())

    D, H, W = 128, 128, 128
    vol = torch.randn(D, H, W, device="cuda", dtype=torch.float32, requires_grad=True)
    thr3 = torch.linspace(-0.5, 0.5, steps=65, device="cuda", dtype=torch.float32).clone().requires_grad_(True)
    hist3d = ecc3d_soft(vol, thr3, alpha=12.0)
    (hist3d.sum()).backward()
    print("3D:", hist3d.shape, vol.grad.abs().mean().item(), thr3.grad.abs().mean().item())
