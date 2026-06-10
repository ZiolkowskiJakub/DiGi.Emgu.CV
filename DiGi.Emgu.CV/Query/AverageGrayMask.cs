using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using System.Collections.Generic;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Generates a binary mask based on the average gray intensity of the provided image, automatically selecting between GPU and CPU implementations.
        /// </summary>
        /// <param name="mat">The input image matrix to process.</param>
        /// <returns>A boolean array representing the mask where true indicates pixels above the mean intensity, or null if the input is null.</returns>
        public static bool[]? AverageGrayMask(this Mat? mat)
        {
            return CudaInvoke.HasCuda ? AverageGrayMask_GPU(mat) : AverageGrayMask_CPU(mat);
        }

        /// <summary>
        /// Generates a binary mask based on the average gray intensity of the provided image using CPU processing.
        /// </summary>
        /// <param name="mat">The input image matrix to process.</param>
        /// <returns>A boolean array representing the mask where true indicates pixels above the mean intensity, or null if the input is null.</returns>
        public static bool[]? AverageGrayMask_CPU(this Mat? mat)
        {
            if (mat == null)
            {
                return null;
            }

            // Convert the Mat to grayscale
            using Mat mat_gray_1 = new();

            CvInvoke.CvtColor(mat, mat_gray_1, ColorConversion.Bgr2Gray);

            // Calculate the mean intensity
            double mean = CvInvoke.Mean(mat_gray_1).V0;

            // Get image dimensions
            int rows = mat_gray_1.Rows;
            int cols = mat_gray_1.Cols;

            // Get pixel data as a flattened array
            byte[] data = new byte[rows * cols];
            mat_gray_1.CopyTo(data); // Safely copy Mat data to a byte array

            // Create the mask
            List<bool> mask = [];
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    // Calculate the index in the flattened array
                    int index = y * cols + x;
                    byte pixel = data[index]; // Access pixel value using the index
                    mask.Add(pixel > mean);
                }
            }

            return [.. mask];
        }

        /// <summary>
        /// Generates a binary mask based on the average gray intensity of the provided image using GPU acceleration via CUDA.
        /// </summary>
        /// <param name="mat">The input image matrix to process.</param>
        /// <returns>A boolean array representing the mask where true indicates pixels above the mean intensity, or null if the input is null or CUDA is unavailable.</returns>
        public static bool[]? AverageGrayMask_GPU(Mat? mat)
        {
            if (mat == null || !CudaInvoke.HasCuda)
            {
                return null;
            }

            using GpuMat gpuMat = new(mat);
            using GpuMat gpuGray = new();
            using GpuMat gpuMask = new();

            // Convert to grayscale on GPU
            CudaInvoke.CvtColor(gpuMat, gpuGray, ColorConversion.Bgr2Gray);

            // Downscale image for fast mean calculation
            using GpuMat gpuSmall = new();

            CudaInvoke.Resize(gpuGray, gpuSmall, new System.Drawing.Size(8, 8)); // Reduce size for fast mean

            using Mat smallMat = new();
            gpuSmall.Download(smallMat); // Transfer small image to CPU

            double mean = CvInvoke.Mean(smallMat).V0; // Compute mean on CPU
            CudaInvoke.Threshold(gpuGray, gpuMask, mean, 255, ThresholdType.Binary); // Thresholding

            // Download the mask to CPU
            using Mat maskMat = new();

            gpuMask.Download(maskMat);

            // Convert to bool array
            if (maskMat.GetData() is byte[] data)
            {
                bool[] mask = new bool[data.Length];
                for (int i = 0; i < data.Length; i++)
                {
                    mask[i] = data[i] > 0;
                }

                return mask;
            }

            return null;
        }
    }
}