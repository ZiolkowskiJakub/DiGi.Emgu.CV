using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using System;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double OpticalFlowAverageMagnitude(Mat mat_1, Mat mat_2)
        {
            return CudaInvoke.HasCuda ? OpticalFlowAverageMagnitude_GPU(mat_1, mat_2) : OpticalFlowAverageMagnitude_CPU(mat_1, mat_2);
        }

        public static double OpticalFlowAverageMagnitude_CPU(Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            using (Mat gray1 = new Mat())
            using (Mat gray2 = new Mat())
            {
                // Convert to grayscale
                CvInvoke.CvtColor(mat_1, gray1, ColorConversion.Bgr2Gray);
                CvInvoke.CvtColor(mat_2, gray2, ColorConversion.Bgr2Gray);

                // Compute optical flow using Farneback method
                using (Mat flow = new Mat())
                {
                    CvInvoke.CalcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, OpticalflowFarnebackFlag.Default);

                    // Get flow data as an array
                    float[] data = flow.GetData(false) as float[];
                    if (data == null)
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

            }
        }

        public static double OpticalFlowAverageMagnitude_GPU(Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null || !CudaInvoke.HasCuda)
            {
                return double.NaN;
            }

            using (GpuMat gpuGray1 = new GpuMat())
            using (GpuMat gpuGray2 = new GpuMat())
            using (GpuMat gpuFlow = new GpuMat())
            {
                // Convert images to grayscale on GPU
                using (GpuMat gpuMat1 = new GpuMat(mat_1))
                using (GpuMat gpuMat2 = new GpuMat(mat_2))
                {
                    CudaInvoke.CvtColor(gpuMat1, gpuGray1, ColorConversion.Bgr2Gray);
                    CudaInvoke.CvtColor(gpuMat2, gpuGray2, ColorConversion.Bgr2Gray);
                }

                // Compute optical flow using CUDA Farneback method
                using (CudaFarnebackOpticalFlow farneback = new CudaFarnebackOpticalFlow(3, 0.5, false, 15, 3, 5, 1.2))
                {
                    farneback.Calc(gpuGray1, gpuGray2, gpuFlow);
                }

                float[] data = null;

                // Download results to CPU
                using (Mat flow = new Mat())
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
}