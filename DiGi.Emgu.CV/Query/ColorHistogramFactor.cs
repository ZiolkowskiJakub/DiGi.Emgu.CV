using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using System.Collections.Generic;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double ColorHistogramFactor(this Mat mat_1, Mat mat_2)
        {
            return CudaInvoke.HasCuda ? ColorHistogramFactor_GPU(mat_1, mat_2) : ColorHistogramFactor_CPU(mat_1, mat_2);
        }

        public static double ColorHistogramFactor_CPU(this Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Convert to HSV (or another color space if needed)
            using (Mat hsvImage_1 = new Mat())
            {
                CvInvoke.CvtColor(mat_1, hsvImage_1, ColorConversion.Bgr2Hsv);

                // Calculate histograms
                using (VectorOfMat hsvChannels_1 = new VectorOfMat())
                {
                    CvInvoke.Split(hsvImage_1, hsvChannels_1);
                    using (Mat hist1 = new Mat())
                    {
                        CvInvoke.CalcHist(new VectorOfMat(hsvChannels_1[0]), new int[] { 0 }, null, hist1, new int[] { 256 }, new float[] { 0, 256 }, false);

                        // Normalize the histogram
                        CvInvoke.Normalize(hist1, hist1, 0, 1, NormType.MinMax);

                        // Convert to HSV (or another color space if needed)
                        using (Mat hsvImage_2 = new Mat())
                        {
                            CvInvoke.CvtColor(mat_2, hsvImage_2, ColorConversion.Bgr2Hsv);

                            // Calculate histograms
                            using (VectorOfMat hsvChannels_2 = new VectorOfMat())
                            {
                                CvInvoke.Split(hsvImage_2, hsvChannels_2);

                                using (Mat hist2 = new Mat())
                                {
                                    CvInvoke.CalcHist(new VectorOfMat(hsvChannels_1[0]), new int[] { 0 }, null, hist2, new int[] { 256 }, new float[] { 0, 256 }, false);

                                    // Normalize the histogram
                                    CvInvoke.Normalize(hist2, hist2, 0, 1, NormType.MinMax);

                                    return CvInvoke.CompareHist(hist1, hist2, HistogramCompMethod.Correl); // Correlation similarity
                                }

                            }

                        }

                    }

                }

            }
        }

        public static double ColorHistogramFactor_GPU(this Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null || !CudaInvoke.HasCuda)
            {
                return double.NaN;
            }

            using (GpuMat gpuMat1 = new GpuMat(mat_1))
            using (GpuMat gpuMat2 = new GpuMat(mat_2))
            using (GpuMat gpuHsvImage1 = new GpuMat(), gpuHsvImage2 = new GpuMat())
            using (VectorOfGpuMat gpuChannels1 = new VectorOfGpuMat(), gpuChannels2 = new VectorOfGpuMat())
            using (GpuMat hist1 = new GpuMat(), hist2 = new GpuMat())
            {
                // Convert to HSV on GPU
                CudaInvoke.CvtColor(gpuMat1, gpuHsvImage1, ColorConversion.Bgr2Hsv);
                CudaInvoke.CvtColor(gpuMat2, gpuHsvImage2, ColorConversion.Bgr2Hsv);

                // Split the channels (Hue, Saturation, and Value)
                CudaInvoke.Split(gpuHsvImage1, gpuChannels1);
                CudaInvoke.Split(gpuHsvImage2, gpuChannels2);

                // Calculate histograms for the Hue channel (index 0 in HSV)
                CudaInvoke.CalcHist(gpuChannels1[0], hist1);  // Calculate for mat_1
                CudaInvoke.CalcHist(gpuChannels2[0], hist2);  // Calculate for mat_2

                // Normalize the histograms on the GPU
                CudaInvoke.Normalize(hist1, hist1, 0, 1, NormType.MinMax, DepthType.Cv32F);
                CudaInvoke.Normalize(hist2, hist2, 0, 1, NormType.MinMax, DepthType.Cv32F);

                // Transfer histograms from GPU to CPU
                using (Mat hist1Cpu = new Mat())
                using (Mat hist2Cpu = new Mat())
                {
                    hist1.Download(hist1Cpu);
                    hist2.Download(hist2Cpu);

                    // Compare the histograms using correlation (CPU method)
                    return CvInvoke.CompareHist(hist1Cpu, hist2Cpu, HistogramCompMethod.Correl);
                }

            }
        }
    }
}