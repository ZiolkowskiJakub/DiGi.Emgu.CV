using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.Structure;
using System;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double AverageColorSimilarity(this Mat mat_1, Mat mat_2)
        {
            return CudaInvoke.HasCuda ? AverageColorSimilarity_GPU(mat_1, mat_2) : AverageColorSimilarity_CPU(mat_1, mat_2);
        }

        public static double AverageColorSimilarity_CPU(this Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            MCvScalar avgColor_1 = CvInvoke.Mean(mat_1);
            MCvScalar avgColor_2 = CvInvoke.Mean(mat_2);

            // Calculate the Euclidean distance between the colors
            return Math.Sqrt(
                Math.Pow(avgColor_1.V0 - avgColor_2.V0, 2) +
                Math.Pow(avgColor_1.V1 - avgColor_2.V1, 2) +
                Math.Pow(avgColor_1.V2 - avgColor_2.V2, 2)
            );

        }

        public static double AverageColorSimilarity_GPU(Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null || !CudaInvoke.HasCuda)
            {
                return double.NaN;
            }

            using (GpuMat gpuMat1 = new GpuMat(mat_1))
            using (GpuMat gpuMat2 = new GpuMat(mat_2))
            using (GpuMat gpuReduced1 = new GpuMat())
            using (GpuMat gpuReduced2 = new GpuMat())
            {
                // Resize image to speed up mean computation
                CudaInvoke.Resize(gpuMat1, gpuReduced1, new System.Drawing.Size(8, 8));
                CudaInvoke.Resize(gpuMat2, gpuReduced2, new System.Drawing.Size(8, 8));

                // Download reduced images to CPU
                using (Mat reducedMat1 = new Mat())
                using (Mat reducedMat2 = new Mat())
                {
                    gpuReduced1.Download(reducedMat1);
                    gpuReduced2.Download(reducedMat2);

                    // Compute mean color on CPU
                    MCvScalar avgColor_1 = CvInvoke.Mean(reducedMat1);
                    MCvScalar avgColor_2 = CvInvoke.Mean(reducedMat2);

                    // Compute Euclidean distance
                    return Math.Sqrt(
                        Math.Pow(avgColor_1.V0 - avgColor_2.V0, 2) +
                        Math.Pow(avgColor_1.V1 - avgColor_2.V1, 2) +
                        Math.Pow(avgColor_1.V2 - avgColor_2.V2, 2)
                    );
                }

            }
        }
    }
}