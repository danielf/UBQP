#include <time.h>
#include <stdio.h>

#define THREADS 512

__device__ float* cQ = NULL;
__device__ float* cb_orig = NULL;

__device__ float* cb = NULL;

__device__ int* val = NULL;
__device__ int* best_val = NULL;

__device__ float sol;
__device__ float best_sol;

static __global__ void cuda_copy(float* t, float* s) {
	int i = threadIdx.x;
	t[i] = s[i];
}

static __global__ void cuda_zero_int(int* t) {
	int i = threadIdx.x;
	t[i] = 0;
}
static __global__ void cuda_changed(int idx, int n) {
	__shared__ int update;
	__shared__ float mult;
	int va;
	int i = threadIdx.x;
	if (i == 0) {
		va = val[idx];
		if (va == 0) {
			sol -= cb[idx];
			val[idx] = 1;
			mult = -1;
		} else {
			sol += cb[idx];
			val[idx] = 0;
			mult = 1;
		}
	}
	__syncthreads();
	cb[i] += mult*cQ[idx*n+i];
	if (i == 0) {
		update = 0;
		if (sol < best_sol) {
			best_sol = sol;
			update = 1;
		}
	}
	__syncthreads();
	if (update == 1) best_val[i] = val[i];
}
static void cuda_update(float* Q, float* b, int n) {
	cudaMemcpy(cQ, Q, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(cb_orig, b, n*sizeof(float), cudaMemcpyHostToDevice);
}
void cuda_initialize(int n) {
	cudaMalloc((void**)&cQ, n*n*sizeof(float));
	cudaMalloc((void**)&cb, n*sizeof(float));
	cudaMalloc((void**)&cb_orig, n*sizeof(float));
	cudaMalloc((void**)&val, n*sizeof(int));
	cudaMalloc((void**)&best_val, n*sizeof(int));
}

void cuda_finalize() {
	cudaFree(cQ);
	cudaFree(cb);
	cudaFree(cb_orig);
	cudaFree(val);
	cudaFree(best_val);
}

static __global__ void cuda_zero_sol() {
	sol = 0.;
	best_sol = 0.;
}

static void cuda_prepare_brute_force(int n) {
	cuda_copy<<< 1, n >>>(cb, cb_orig);
	cuda_zero_int<<< 1, n >>>(val);
	cuda_zero_int<<< 1, n >>>(best_val);
	cuda_zero_sol<<< 1, 1 >>>();
}
void cuda_brute_force(float* Q, float* b, int* ans, int n) {
	printf("Chamado com n = %d\n", n);
	clock_t before = clock();
	int lasti = 0;
	cuda_update(Q, b, n);
	cuda_prepare_brute_force(n);
	for (int _i = 1; _i < (1 << n); _i++) {
		int i, _changed, changed;
		i = _i ^ (_i >> 1);
		_changed = lasti ^ i;
		lasti = i;
		for (changed = -1; _changed; _changed >>= 1) changed++;
		if (_i % 1000 == 0) printf("Antes de chamar o cuda_changed %d!!\n", _i);
		cuda_changed<<<1, n>>>(changed, n);
	//	printf("Depois de chamar o cuda_changed!!\n");
	}
	cudaMemcpy(ans, best_val, n*sizeof(int), cudaMemcpyDeviceToHost);
	clock_t after = clock();
	printf("Cuda: Brute-force for %d vars in %lf secs\n", n, (1.*(after-before))/CLOCKS_PER_SEC);
}


