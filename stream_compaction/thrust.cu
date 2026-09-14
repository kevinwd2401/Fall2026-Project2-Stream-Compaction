#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            if (n == 0) {
                timer().endGpuTimer();
                return;
			}

            int* d_in, * d_out;
            cudaMalloc(&d_in, n * sizeof(int));
            cudaMalloc(&d_out, n * sizeof(int));
			cudaMemcpy(d_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            auto dv_in = thrust::device_pointer_cast(d_in);
            auto dv_out = thrust::device_pointer_cast(d_out);

            thrust::exclusive_scan(dv_in, dv_in + n, dv_out);

            cudaMemcpy(odata, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_in);
			cudaFree(d_out);


            timer().endGpuTimer();
        }
    }
}
