#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScanUp(
            int n, int two_d, int two_d1, int activeThreads, int* data) {

            int thread = blockIdx.x * blockDim.x + threadIdx.x;
            if (thread >= activeThreads) {
                return;
            }

            long long index =
                (static_cast<long long>(thread) + 1LL) *
                static_cast<long long>(two_d1) - 1LL;

            if (index < n) {
                data[index] += data[index - two_d];
            }
        }

        __global__ void kernScanDown(
            int n, int two_d, int two_d1, int activeThreads, int* data) {

            int thread = blockIdx.x * blockDim.x + threadIdx.x;
            if (thread >= activeThreads) {
                return;
            }

            long long index =
                (static_cast<long long>(thread) + 1LL) *
                static_cast<long long>(two_d1) - 1LL;

            if (index < n) {
                int t = data[index - two_d];

                data[index - two_d] = data[index];
                data[index] += t;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO
            if (n <= 0) {
                return;
            }

            int paddedN = 1;
            while (paddedN < n) {
                paddedN <<= 1;
            }

            // pad with zeros
            int* d_data;
            cudaMalloc(&d_data, paddedN * sizeof(int));
            cudaMemcpy(
                d_data,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );
            cudaMemset(
                d_data + n,
                0,
                (paddedN - n) * sizeof(int)
            );

            timer().startGpuTimer();
            const int blockSize = 128;


            //up sweep
            for (int d = 0; d < ilog2ceil(paddedN); ++d) {
                int twod = 1 << d;
                int twod1 = twod << 1;

                int activeThreads = (paddedN + twod1 - 1) / twod1;
                int blocks = (activeThreads + blockSize - 1) / blockSize;

                kernScanUp <<<blocks, blockSize>>> (
                    paddedN, twod, twod1, activeThreads, d_data);
            }

			//set root to 0
            cudaMemset(
                d_data + paddedN - 1,
                0,
                sizeof(int)
            );

            //down sweep
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; --d) {
                int twod = 1 << d;
                int twod1 = twod << 1;

                int activeThreads = (paddedN + twod1 - 1) / twod1;
                int blocks = (activeThreads + blockSize - 1) / blockSize;

                kernScanDown <<<blocks, blockSize>>> (
                    paddedN, twod, twod1, activeThreads, d_data);
            }
            
            timer().endGpuTimer();

            cudaMemcpy(
                odata,
                d_data,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(d_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            if (n <= 0) {
                timer().endGpuTimer();
                return 0;
            }
            const int blockSize = 256;
            const int numBlocks = (n + blockSize - 1) / blockSize;

            int* d_idata, * d_odata, * d_bools;
            cudaMalloc(&d_idata, n * sizeof(int));
            cudaMalloc(&d_odata, n * sizeof(int));
            cudaMalloc(&d_bools, n * sizeof(int));

            cudaMemcpy(
                d_idata,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );
            Common::kernMapToBoolean <<< numBlocks, blockSize >>> (n, d_bools, d_idata);

			//Scan d_bools to get indices, scan array is d_data

            int paddedN = 1;
            while (paddedN < n) {
                paddedN <<= 1;
            }

            // pad with zeros
            int* d_data;
            cudaMalloc(&d_data, paddedN * sizeof(int));
            cudaMemcpy(
                d_data,
                d_bools,
                n * sizeof(int),
                cudaMemcpyDeviceToDevice
            );
            cudaMemset(
                d_data + n,
                0,
                (paddedN - n) * sizeof(int)
            );

            //up sweep
            for (int d = 0; d < ilog2ceil(paddedN); ++d) {
                int twod = 1 << d;
                int twod1 = twod << 1;

                int activeThreads = (paddedN + twod1 - 1) / twod1;
                int blocks = (activeThreads + blockSize - 1) / blockSize;

                kernScanUp << <blocks, blockSize >> > (
                    paddedN, twod, twod1, activeThreads, d_data);
            }

            //set root to 0
            cudaMemset(
                d_data + paddedN - 1,
                0,
                sizeof(int)
            );

            //down sweep
            for (int d = ilog2ceil(paddedN) - 1; d >= 0; --d) {
                int twod = 1 << d;
                int twod1 = twod << 1;

                int activeThreads = (paddedN + twod1 - 1) / twod1;
                int blocks = (activeThreads + blockSize - 1) / blockSize;

                kernScanDown << <blocks, blockSize >> > (
                    paddedN, twod, twod1, activeThreads, d_data);
            }

            //scatter using d_data as index
            Common::kernScatter << < numBlocks, blockSize >> > (n, d_odata,
                d_idata, d_bools, d_data);

			//calcalate count of non-zero elements
            int lastScan = 0;
            int lastBool = 0;
            cudaMemcpy(
                &lastBool,
                d_bools + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );
            cudaMemcpy(
                &lastScan,
                d_data + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );
			int count = lastScan + lastBool;

			//copy count elements to odata

            cudaMemcpy(
                odata,
                d_odata,
                count * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(d_data);
            cudaFree(d_idata);
            cudaFree(d_odata);
            cudaFree(d_bools);

            timer().endGpuTimer();
            return count;
        }
    }
}
