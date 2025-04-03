using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using System.Collections.Generic;


namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static bool[] AverageGrayMask(this Mat mat)
        {
            return CudaInvoke.HasCuda ? AverageGrayMask_GPU(mat) : AverageGrayMask_CPU(mat);
        }

        public static bool[] AverageGrayMask_CPU(this Mat mat)
        {
            if (mat == null)
            {
                return null;
            }

            // Convert the Mat to grayscale
            using (Mat mat_gray_1 = new Mat())
            {
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
                List<bool> mask = new List<bool>();
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

                return mask.ToArray();
            }
        }

        public static bool[] AverageGrayMask_GPU(Mat mat)
        {
            if (mat == null || !CudaInvoke.HasCuda)
            {
                return null;
            }

            using (GpuMat gpuMat = new GpuMat(mat))
            using (GpuMat gpuGray = new GpuMat())
            using (GpuMat gpuMask = new GpuMat())
            {
                // Convert to grayscale on GPU
                CudaInvoke.CvtColor(gpuMat, gpuGray, ColorConversion.Bgr2Gray);

                // Downscale image for fast mean calculation
                using (GpuMat gpuSmall = new GpuMat())
                {
                    CudaInvoke.Resize(gpuGray, gpuSmall, new System.Drawing.Size(8, 8)); // Reduce size for fast mean
                    using (Mat smallMat = new Mat())
                    {
                        gpuSmall.Download(smallMat); // Transfer small image to CPU

                        double mean = CvInvoke.Mean(smallMat).V0; // Compute mean on CPU
                        CudaInvoke.Threshold(gpuGray, gpuMask, mean, 255, ThresholdType.Binary); // Thresholding

                        // Download the mask to CPU
                        using (Mat maskMat = new Mat())
                        {
                            gpuMask.Download(maskMat);

                            // Convert to bool array
                            byte[] data = maskMat.GetData() as byte[];
                            bool[] mask = new bool[data.Length];
                            for (int i = 0; i < data.Length; i++)
                            {
                                mask[i] = data[i] > 0;
                            }

                            return mask;
                        }
                    }

                }
            }
        }
    }
}