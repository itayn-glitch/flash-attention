// Standalone microbench: cp.async "double-hop" dequant-on-load for head-512 KV.
//
// De-risks M5 option-2 before threading ElementKV through the FA3 mainloop:
//   1) SRAM budget: request the real per-stage dynamic smem at head-512 tiles;
//      launch succeeds iff it fits H100's 227KB opt-in cap.
//   2) register-cast overhead: is fp8->bf16 convert (+per-tensor descale) hidden
//      behind the HBM load, and does streaming 1-byte fp8 beat 2-byte bf16 ~2x?
//
// Two kernels stream a large KV array (>> L2) tile-by-tile, like the mainloop
// N-loop, so we measure steady-state HBM bandwidth, not cache.
//   A) bf16 baseline : cp.async bf16 gmem -> bf16 smem                (2 B/elem read)
//   B) fp8 dequant   : cp.async fp8  gmem -> fp8 smem -> regs
//                      cvt+descale -> bf16 smem                       (1 B/elem read)
// A checksum consumed at the end defeats dead-code elimination so the convert
// really executes.
//
// Build: nvcc -O3 -arch=sm_90a dequant_dblhop_microbench.cu -o dblhop
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2;}}while(0)

// Real tile: head-512, kBlockN rows of KV per tile. Two configs of interest:
//   kBlockN=64  -> bf16 K+V=128KB, +fp8 stage 64KB, +Q 64KB = 256KB (OVERFLOW)
//   kBlockN=32  -> bf16 K+V=64KB,  +fp8 stage 32KB, +Q 64KB = 160KB (FITS)
constexpr int HEAD = 512;
constexpr int THREADS = 256;              // one mma warpgroup + producer-ish; 256 keeps it simple

// One tile = ROWS x HEAD elements. ROWS*HEAD must be divisible by THREADS*8 (128-bit fp8=16B? no:
// fp8 is 1B, load 16 fp8 = 16B per thread via cp.async 16B chunk).
template<int ROWS>
__global__ void bf16_stream(const __nv_bfloat16* __restrict__ g, int ntiles, int nt_total, float* __restrict__ out) {
  extern __shared__ char smem[];
  __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(smem);   // ROWS*HEAD bf16 = ROWS*HEAD*2 B
  const int tid = threadIdx.x;
  const int TILE = ROWS * HEAD;                                 // elems per tile
  const int CH = 8;                                             // 8 bf16 = 16B cp.async chunk
  const int nchunk = TILE / CH;                                 // chunks per tile
  float acc = 0.f;
  for (int t = 0; t < ntiles; ++t) {
    const __nv_bfloat16* gt = g + (size_t)(((size_t)blockIdx.x + (size_t)t*gridDim.x) % nt_total) * TILE; // march past L2
    for (int c = tid; c < nchunk; c += THREADS)
      __pipeline_memcpy_async(&s[c*CH], &gt[c*CH], 16);
    __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();
    // consume so the load isn't DCE'd
    for (int i = tid; i < TILE; i += THREADS) acc += __bfloat162float(s[i]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = acc;
}

template<int ROWS>
__global__ void fp8_dequant_stream(const __nv_fp8_e4m3* __restrict__ g, float descale,
                                   int ntiles, int nt_total, float* __restrict__ out) {
  extern __shared__ char smem[];
  __nv_fp8_e4m3* sf = reinterpret_cast<__nv_fp8_e4m3*>(smem);   // fp8 stage: ROWS*HEAD*1 B
  __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(smem + ROWS*HEAD); // bf16 tile after it
  const int tid = threadIdx.x;
  const int TILE = ROWS * HEAD;
  const int CH = 16;                                            // 16 fp8 = 16B cp.async chunk
  const int nchunk = TILE / CH;
  float acc = 0.f;
  for (int t = 0; t < ntiles; ++t) {
    const __nv_fp8_e4m3* gt = g + (size_t)(((size_t)blockIdx.x + (size_t)t*gridDim.x) % nt_total) * TILE;
    // hop 1: fp8 gmem -> fp8 smem (cp.async)
    for (int c = tid; c < nchunk; c += THREADS)
      __pipeline_memcpy_async(&sf[c*CH], &gt[c*CH], 16);
    __pipeline_commit(); __pipeline_wait_prior(0); __syncthreads();
    // hop 2: fp8 smem -> regs -> cvt+descale -> bf16 smem
    for (int i = tid; i < TILE; i += THREADS) {
      float v = (float)sf[i] * descale;
      sb[i] = __float2bfloat16(v);
    }
    __syncthreads();
    for (int i = tid; i < TILE; i += THREADS) acc += __bfloat162float(sb[i]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = acc;
}

// Vectorized fp8->bf16 convert: 4 fp8 -> float4 (HW cvt) -> descale -> 2x bf16x2.
// Processes TILE elems, 4 per thread-step. Requires TILE % 4 == 0.
__device__ __forceinline__ void convert_vec(const __nv_fp8_e4m3* sf, __nv_bfloat16* sb,
                                            int TILE, int tid, float descale) {
  const __nv_fp8x4_e4m3* sf4 = reinterpret_cast<const __nv_fp8x4_e4m3*>(sf);
  __nv_bfloat162* sb2 = reinterpret_cast<__nv_bfloat162*>(sb);
  const int n4 = TILE / 4;
  for (int j = tid; j < n4; j += THREADS) {
    float4 f = static_cast<float4>(sf4[j]);          // HW fp8x4 -> float4
    f.x *= descale; f.y *= descale; f.z *= descale; f.w *= descale;
    sb2[2*j]   = __floats2bfloat162_rn(f.x, f.y);
    sb2[2*j+1] = __floats2bfloat162_rn(f.z, f.w);
  }
}

template<int ROWS>
__global__ void convert_only_vec(int reps, float descale, float* __restrict__ out) {
  extern __shared__ char smem[];
  const int TILE = ROWS * HEAD;
  __nv_fp8_e4m3* sf = reinterpret_cast<__nv_fp8_e4m3*>(smem);
  __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(smem + TILE);
  const int tid = threadIdx.x;
  for (int i = tid; i < TILE; i += THREADS) sf[i] = __nv_fp8_e4m3(1.0f);
  __syncthreads();
  float acc = 0.f;
  for (int r = 0; r < reps; ++r) {
    convert_vec(sf, sb, TILE, tid, descale);
    __syncthreads();
    if ((r & 63) == 0) for (int i = tid; i < TILE; i += THREADS) acc += __bfloat162float(sb[i]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = acc;
}

// Pipelined 2-stage fp8 dequant with VECTORIZED convert.
template<int ROWS>
__global__ void fp8_dequant_pipe_vec(const __nv_fp8_e4m3* __restrict__ g, float descale,
                                     int ntiles, int nt_total, float* __restrict__ out) {
  extern __shared__ char smem[];
  const int TILE = ROWS * HEAD;
  __nv_fp8_e4m3* sf0 = reinterpret_cast<__nv_fp8_e4m3*>(smem);
  __nv_fp8_e4m3* sf1 = reinterpret_cast<__nv_fp8_e4m3*>(smem + TILE);
  __nv_bfloat16* sb  = reinterpret_cast<__nv_bfloat16*>(smem + 2*TILE);
  __nv_fp8_e4m3* stg[2] = {sf0, sf1};
  const int tid = threadIdx.x;
  const int CH = 16, nchunk = TILE / CH;
  auto tileptr = [&](int t){ return g + (size_t)(((size_t)blockIdx.x + (size_t)t*gridDim.x) % nt_total) * TILE; };
  auto issue = [&](int t, int slot){
    const __nv_fp8_e4m3* gt = tileptr(t);
    for (int c = tid; c < nchunk; c += THREADS) __pipeline_memcpy_async(&stg[slot][c*CH], &gt[c*CH], 16);
    __pipeline_commit();
  };
  float acc = 0.f;
  issue(0, 0);
  for (int t = 0; t < ntiles; ++t) {
    if (t + 1 < ntiles) issue(t+1, (t+1)&1);
    __pipeline_wait_prior(t + 1 < ntiles ? 1 : 0); __syncthreads();
    convert_vec(stg[t&1], sb, TILE, tid, descale);
    __syncthreads();
    for (int i = tid; i < TILE; i += THREADS) acc += __bfloat162float(sb[i]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = acc;
}

// Double-buffered (2-stage) fp8 dequant: convert+consume of tile t overlaps the
// cp.async HBM load of tile t+1 -- the REAL pipelined mainloop structure.
template<int ROWS>
__global__ void fp8_dequant_pipe(const __nv_fp8_e4m3* __restrict__ g, float descale,
                                 int ntiles, int nt_total, float* __restrict__ out) {
  extern __shared__ char smem[];
  const int TILE = ROWS * HEAD;
  __nv_fp8_e4m3* sf0 = reinterpret_cast<__nv_fp8_e4m3*>(smem);              // stage 0 fp8
  __nv_fp8_e4m3* sf1 = reinterpret_cast<__nv_fp8_e4m3*>(smem + TILE);       // stage 1 fp8
  __nv_bfloat16* sb  = reinterpret_cast<__nv_bfloat16*>(smem + 2*TILE);     // bf16 tile
  __nv_fp8_e4m3* stg[2] = {sf0, sf1};
  const int tid = threadIdx.x;
  const int CH = 16, nchunk = TILE / CH;
  auto tileptr = [&](int t){ return g + (size_t)(((size_t)blockIdx.x + (size_t)t*gridDim.x) % nt_total) * TILE; };
  auto issue = [&](int t, int slot){
    const __nv_fp8_e4m3* gt = tileptr(t);
    for (int c = tid; c < nchunk; c += THREADS) __pipeline_memcpy_async(&stg[slot][c*CH], &gt[c*CH], 16);
    __pipeline_commit();
  };
  float acc = 0.f;
  issue(0, 0);                                  // prologue: load tile 0
  for (int t = 0; t < ntiles; ++t) {
    if (t + 1 < ntiles) issue(t+1, (t+1)&1);    // launch next tile's HBM load (overlaps below)
    __pipeline_wait_prior(t + 1 < ntiles ? 1 : 0); __syncthreads();  // wait for tile t only
    __nv_fp8_e4m3* cur = stg[t&1];
    for (int i = tid; i < TILE; i += THREADS) { float v = (float)cur[i]*descale; sb[i] = __float2bfloat16(v); }
    __syncthreads();
    for (int i = tid; i < TILE; i += THREADS) acc += __bfloat162float(sb[i]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = acc;
}

// Pure convert throughput ceiling: data resident in smem (no HBM), loop the
// fp8->bf16 cvt+descale K times. Answers "can convert outpace HBM delivery?"
template<int ROWS>
__global__ void convert_only(int reps, float descale, float* __restrict__ out) {
  extern __shared__ char smem[];
  const int TILE = ROWS * HEAD;
  __nv_fp8_e4m3* sf = reinterpret_cast<__nv_fp8_e4m3*>(smem);
  __nv_bfloat16* sb = reinterpret_cast<__nv_bfloat16*>(smem + TILE);
  const int tid = threadIdx.x;
  for (int i = tid; i < TILE; i += THREADS) sf[i] = __nv_fp8_e4m3(1.0f);
  __syncthreads();
  float acc = 0.f;
  for (int r = 0; r < reps; ++r) {
    for (int i = tid; i < TILE; i += THREADS) { float v = (float)sf[i]*descale; sb[i] = __float2bfloat16(v); }
    __syncthreads();
    if ((r & 63) == 0) for (int i = tid; i < TILE; i += THREADS) acc += __bfloat162float(sb[i]);
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = acc;
}

template<int ROWS>
int run(const char* tag) {
  const int TILE = ROWS * HEAD;
  const int NBLOCKS = 132 * 2;                 // > SM count
  // Footprint MUST exceed H100 L2 (50MB) so reads hit HBM, not cache. Use ~1.5GB
  // of tiles; each block marches through the whole buffer across NTILES iters.
  const size_t FOOTPRINT = (size_t)1536 * 1024 * 1024;   // 1.5 GB (bf16 sizing)
  const size_t NT_TOTAL = FOOTPRINT / ((size_t)TILE * 2);// # bf16 tiles spanning footprint
  const int NTILES = 800;                                // steady-state iters/block
  const size_t nElem = NT_TOTAL * TILE;        // element count (bf16=2B, fp8=1B alloc separately)
  // smem needs
  size_t sm_bf16 = (size_t)TILE * 2;
  size_t sm_fp8  = (size_t)TILE * 1 + (size_t)TILE * 2;  // fp8 stage + bf16 tile (double-hop peak)
  printf("[%s] ROWS=%d TILE=%d  smem bf16-baseline=%zuKB  fp8-doublehop=%zuKB (cap 227KB)\n",
         tag, ROWS, TILE, sm_bf16/1024, sm_fp8/1024);

  // buffers
  __nv_bfloat16* gb; __nv_fp8_e4m3* gf; float* out;
  CK(cudaMalloc(&gb, nElem*sizeof(__nv_bfloat16)));
  CK(cudaMalloc(&gf, nElem*sizeof(__nv_fp8_e4m3)));
  CK(cudaMalloc(&out, NBLOCKS*sizeof(float)));
  CK(cudaMemset(gb, 1, nElem*sizeof(__nv_bfloat16)));
  CK(cudaMemset(gf, 1, nElem*sizeof(__nv_fp8_e4m3)));

  cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  float ms; const float descale = 0.5f;

  // ---- bf16 baseline ----
  CK(cudaFuncSetAttribute(bf16_stream<ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_bf16));
  bf16_stream<ROWS><<<NBLOCKS,THREADS,sm_bf16>>>(gb, NTILES, (int)NT_TOTAL, out); // warmup
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  bf16_stream<ROWS><<<NBLOCKS,THREADS,sm_bf16>>>(gb, NTILES, (int)NT_TOTAL, out);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms,a,b));
  double bytes_bf16 = (double)NBLOCKS*NTILES*TILE*2.0;
  printf("[%s] bf16   : %.3f ms  %.1f GB/s (read %.1f MB/iter)\n",
         tag, ms, bytes_bf16/1e6/ms, (double)NBLOCKS*TILE*2.0/1e6);

  // ---- fp8 dequant double-hop ----
  cudaError_t se = cudaFuncSetAttribute(fp8_dequant_stream<ROWS>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, sm_fp8);
  if (se != cudaSuccess) {
    printf("[%s] fp8 double-hop: smem %zuKB REJECTED by driver: %s  -> config does NOT fit\n",
           tag, sm_fp8/1024, cudaGetErrorString(se));
  } else {
    fp8_dequant_stream<ROWS><<<NBLOCKS,THREADS,sm_fp8>>>(gf, descale, NTILES, (int)NT_TOTAL, out); // warmup
    cudaError_t le = cudaDeviceSynchronize();
    if (le != cudaSuccess) {
      printf("[%s] fp8 double-hop: launch failed: %s\n", tag, cudaGetErrorString(le));
    } else {
      CK(cudaEventRecord(a));
      fp8_dequant_stream<ROWS><<<NBLOCKS,THREADS,sm_fp8>>>(gf, descale, NTILES, (int)NT_TOTAL, out);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms,a,b));
      double bytes_fp8 = (double)NBLOCKS*NTILES*TILE*1.0;
      printf("[%s] fp8 dq (serial)  : %.3f ms  eff %.1f GB/s (read %.1f MB/iter)  <- HALF bytes, worst-case\n",
             tag, ms, bytes_fp8/1e6/ms, (double)NBLOCKS*TILE*1.0/1e6);
    }
  }

  // ---- fp8 dequant PIPELINED (2-stage) ----
  size_t sm_pipe = (size_t)TILE*2 /*two fp8 stages*/ + (size_t)TILE*2 /*bf16 tile*/;
  cudaError_t pe = cudaFuncSetAttribute(fp8_dequant_pipe<ROWS>,
                     cudaFuncAttributeMaxDynamicSharedMemorySize, sm_pipe);
  if (pe != cudaSuccess) {
    printf("[%s] fp8 pipe: smem %zuKB REJECTED: %s\n", tag, sm_pipe/1024, cudaGetErrorString(pe));
  } else {
    fp8_dequant_pipe<ROWS><<<NBLOCKS,THREADS,sm_pipe>>>(gf, descale, NTILES, (int)NT_TOTAL, out);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    fp8_dequant_pipe<ROWS><<<NBLOCKS,THREADS,sm_pipe>>>(gf, descale, NTILES, (int)NT_TOTAL, out);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms,a,b));
    double bytes_fp8 = (double)NBLOCKS*NTILES*TILE*1.0;
    printf("[%s] fp8 dq (pipe 2-stg): %.3f ms  eff %.1f GB/s (smem %zuKB)  <- realistic\n",
           tag, ms, bytes_fp8/1e6/ms, sm_pipe/1024);
  }

  // ---- pure convert throughput ceiling ----
  size_t sm_cvt = (size_t)TILE*1 + (size_t)TILE*2;
  CK(cudaFuncSetAttribute(convert_only<ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_cvt));
  const int REPS = 20000;
  convert_only<ROWS><<<NBLOCKS,THREADS,sm_cvt>>>(REPS, descale, out);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  convert_only<ROWS><<<NBLOCKS,THREADS,sm_cvt>>>(REPS, descale, out);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms,a,b));
  double cvt_out_bytes = (double)NBLOCKS*REPS*TILE*2.0;  // bf16 produced
  printf("[%s] convert ceiling (scalar): %.3f ms  %.0f GB/s bf16 out  (~%.0f GB/s fp8 in)\n",
         tag, ms, cvt_out_bytes/1e6/ms, cvt_out_bytes/2.0/1e6/ms);

  // ---- pure convert throughput ceiling (VECTORIZED) ----
  convert_only_vec<ROWS><<<NBLOCKS,THREADS,sm_cvt>>>(REPS, descale, out);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a));
  convert_only_vec<ROWS><<<NBLOCKS,THREADS,sm_cvt>>>(REPS, descale, out);
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms,a,b));
  printf("[%s] convert ceiling (vec)   : %.3f ms  %.0f GB/s bf16 out  (~%.0f GB/s fp8 in)\n",
         tag, ms, cvt_out_bytes/1e6/ms, cvt_out_bytes/2.0/1e6/ms);

  // ---- fp8 dequant PIPELINED with VECTORIZED convert ----
  if (pe == cudaSuccess) {
    CK(cudaFuncSetAttribute(fp8_dequant_pipe_vec<ROWS>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_pipe));
    fp8_dequant_pipe_vec<ROWS><<<NBLOCKS,THREADS,sm_pipe>>>(gf, descale, NTILES, (int)NT_TOTAL, out);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(a));
    fp8_dequant_pipe_vec<ROWS><<<NBLOCKS,THREADS,sm_pipe>>>(gf, descale, NTILES, (int)NT_TOTAL, out);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&ms,a,b));
    double bytes_fp8 = (double)NBLOCKS*NTILES*TILE*1.0;
    printf("[%s] fp8 dq (pipe+vec)  : %.3f ms  eff %.1f GB/s  <- realistic+optimized\n",
           tag, ms, bytes_fp8/1e6/ms);
  }
  CK(cudaFree(gb)); CK(cudaFree(gf)); CK(cudaFree(out));
  printf("\n");
  return 0;
}

int main() {
  int dev; CK(cudaGetDevice(&dev));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
  printf("GPU: %s  SMs=%d  smemPerBlockOptin=%zuKB  busWidth=%d-bit\n",
         p.name, p.multiProcessorCount, p.sharedMemPerBlockOptin/1024,
         p.memoryBusWidth);
  printf("(H100 HBM3 peak ~3350 GB/s)\n\n");
  run<32>("kBlockN=32");
  run<64>("kBlockN=64");
  // Full-kernel smem budget (head-512, Q=64x512 bf16=64KB resident), for kStages=1:
  //   naive  = bf16_K + bf16_V + fp8stage_K + fp8stage_V + Q
  //   reuse  = bf16_K + bf16_V + fp8stage_shared(max K,V) + Q   (convert K, reuse buf for V)
  for (int N : {32, 64}) {
    double bf16KV = 2.0 * N * HEAD * 2 / 1024.0;      // K+V bf16
    double fp8one = (double)N * HEAD * 1 / 1024.0;    // one fp8 stage
    double Q = 64.0 * HEAD * 2 / 1024.0;
    printf("[budget kBlockN=%d] naive=%.0fKB  reuse=%.0fKB  (Q=%.0fKB, cap 227KB)\n",
           N, bf16KV + 2*fp8one + Q, bf16KV + fp8one + Q, Q);
  }
  return 0;
}
