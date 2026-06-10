using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using System;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Calculates the average magnitude of optical flow between two images, automatically selecting 
        /// between GPU and CPU implementations based on CUDA availability.
        /// </summary>
        /// <param name="mat_1">The first input image.</param>
        /// <param name="mat_2">The second input image.</param>
        /// <returns>The average magnitude of the optical flow vectors.</returns>
        public static double OpticalFlowAverageMagnitude(Mat? mat_1, Mat? mat_2)
        {
            return CudaInvoke.HasCuda ? OpticalFlowAverageMagnitude_GPU(mat_1, mat_2) : OpticalFlowAverageMagnitude_CPU(mat_1, mat_2);
        }

        /// <summary>
        /// Calculates the average magnitude of optical flow between two images using the CPU-based 
        /// Farneback method.
        /// </summary>
        /// <param name="mat_1">The first input image.</param>
        /// <param name="mat_2">The second input image.</param>
        /// <returns>The average magnitude of the optical flow vectors, or <see cref="double.NaN"/> if either input image is null.</returns>
        public static double OpticalFlowAverageMagnitude_CPU(Mat? mat_1, Mat? mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            using Mat gray1 = new();
            using Mat gray2 = new();

            // Convert to grayscale
            CvInvoke.CvtColor(mat_1, gray1, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(mat_2, gray2, ColorConversion.Bgr2Gray);

            // Compute optical flow using Farneback method
            using Mat flow = new();

            CvInvoke.CalcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, OpticalflowFarnebackFlag.Default);

            // Get flow data as an array
            if (flow.GetData(false) is not float[] data)
            {
                return double.NaN;
            }

            // Calculate the average magnitude of the flow vectors
            double totalMagnitude = 0.0;
            int count = 0;

            for (int i = 0; i < data.Length / 2; i++)
            {
                // Each pair of elements represents the x and y components of the flow vector
                float flowX = data[i * 2];    // x component at position i
                float flowY = data[i * 2 + 1]; // y component at position i

                // Compute the magnitude of the flow vector
                double magnitude = Math.Sqrt(flowX * flowX + flowY * flowY);
                totalMagnitude += magnitude;
                count++;
            }

            // Return the average magnitude
            return count > 0 ? totalMagnitude / count : 0.0;
        }

        /// <summary>
        /// Calculates the average magnitude of optical flow between two images using the GPU-accelerated 
        /// Farneback method via CUDA.
        /// </summary>
        /// <param name="mat_1">The first input image.</param>
        /// <param name="mat_2">The second input image.</param>
        /// <returns>The average magnitude of the optical flow vectors, or <see cref="double.NaN"/> if either input image is null or CUDA is not available.</returns>
        public static double OpticalFlowAverageMagnitude_GPU(Mat? mat_1, Mat? mat_2)
        {
            if (mat_1 == null || mat_2 == null || !CudaInvoke.HasCuda)
            {
                return double.NaN;
            }

            using GpuMat gpuGray1 = new();
            using GpuMat gpuGray2 = new();
            using GpuMat gpuFlow = new();

            // Convert images to grayscale on GPU
            using (GpuMat gpuMat1 = new(mat_1))
            using (GpuMat gpuMat2 = new(mat_2))
            {
                CudaInvoke.CvtColor(gpuMat1, gpuGray1, ColorConversion.Bgr2Gray);
                CudaInvoke.CvtColor(gpuMat2, gpuGray2, ColorConversion.Bgr2Gray);
            }

            // Compute optical flow using CUDA Farneback method
            using (CudaFarnebackOpticalFlow farneback = new(3, 0.5, false, 15, 3, 5, 1.2))
            {
                farneback.Calc(gpuGray1, gpuGray2, gpuFlow);
            }

            float[]? data = null;

            // Download results to CPU
            using (Mat flow = new())
            {
                gpuFlow.Download(flow);

                // Extract flow vectors
                data = flow.GetData(false) as float[];
            }

            if (data == null)
            {
                return double.NaN;
            }

            // Compute average magnitude
            double totalMagnitude = 0.0;
            int count = data.Length / 2;

            for (int i = 0; i < count; i++)
            {
                float flowX = data[i * 2];
                float flowY = data[i * 2 + 1];

                totalMagnitude += Math.Sqrt(flowX * flowX + flowY * flowY);
            }

            return count > 0 ? totalMagnitude / count : 0.0;
        }
    }
}