#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernNaiveScan(int n, int d, int *odata, const int *idata) {
            
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            if (k >= n) {
                return;
            }

            int offset = 1 << (d - 1);

            if (k >= offset) {
                odata[k] = idata[k] + idata[k - offset];
            }
            else {
                odata[k] = idata[k];
            }
		}

        __global__ void kernInclusiveToExclusive(
            int n,
            int* odata,
            const int* idata
        ) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            if (k >= n) {
                return;
            }

            odata[k] = (k == 0) ? 0 : idata[k - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            if (n <= 0) {
                timer().endGpuTimer();
                return;
            }

            const int blockSize = 256;
            const int numBlocks = (n + blockSize - 1) / blockSize;

            int* d_bufferA, * d_bufferB;
            cudaMalloc(&d_bufferA, n * sizeof(int));
            cudaMalloc(&d_bufferB, n * sizeof(int));

			int* inputBuffer = d_bufferA;
            int* outputBuffer = d_bufferB;

            cudaMemcpy(
                d_bufferA,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );
            
            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan <<<numBlocks, blockSize>>> (n, d, outputBuffer, inputBuffer);

                std::swap(inputBuffer, outputBuffer);
            }

            kernInclusiveToExclusive <<<numBlocks, blockSize>>> (n, outputBuffer, inputBuffer);

            cudaMemcpy(
                odata,
                outputBuffer,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(d_bufferA);
            cudaFree(d_bufferB);

            timer().endGpuTimer();
        }
    }
}
