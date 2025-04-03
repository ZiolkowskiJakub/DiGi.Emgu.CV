using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using System.Linq;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double ORBFeatureMatchingFactor(this Mat mat_1, Mat mat_2)
        {
            return CudaInvoke.HasCuda ? ORBFeatureMatchingFactor_CPU(mat_1, mat_2) : ORBFeatureMatchingFactor_CPU(mat_1, mat_2);
        }

        public static double ORBFeatureMatchingFactor_CPU(this Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Convert the original Mat to grayscale
            using (Mat mat_gray_1 = new Mat())
            using (Mat mat_gray_2 = new Mat())
            {
                CvInvoke.CvtColor(mat_1, mat_gray_1, ColorConversion.Bgr2Gray);
                CvInvoke.CvtColor(mat_2, mat_gray_2, ColorConversion.Bgr2Gray);

                // Convert to UMat
                using (UMat uMat_1 = new UMat())
                using (UMat uMat_2 = new UMat())
                {
                    mat_gray_1.CopyTo(uMat_1);
                    mat_gray_2.CopyTo(uMat_2);

                    // Detect ORB keypoints and descriptors
                    using (ORB orb = new ORB())
                    using (VectorOfKeyPoint keyPoints_1 = new VectorOfKeyPoint())
                    using (VectorOfKeyPoint keyPoints_2 = new VectorOfKeyPoint())
                    using (Mat descriptors_1 = new Mat())
                    using (Mat descriptors_2 = new Mat())
                    {
                        orb.DetectAndCompute(uMat_1, null, keyPoints_1, descriptors_1, false);
                        orb.DetectAndCompute(uMat_2, null, keyPoints_2, descriptors_2, false);

                        // Check if descriptors are empty
                        if (descriptors_1.IsEmpty || descriptors_2.IsEmpty)
                        {
                            return double.NaN; // No descriptors found
                        }

                        // Ensure descriptors are of type CV_8U
                        if (descriptors_1.Depth != DepthType.Cv8U)
                        {
                            descriptors_1.ConvertTo(descriptors_1, DepthType.Cv8U);
                        }

                        if (descriptors_2.Depth != DepthType.Cv8U)
                        {
                            descriptors_2.ConvertTo(descriptors_2, DepthType.Cv8U);
                        }

                        // Match features using Brute-Force Matcher
                        using (BFMatcher matcher = new BFMatcher(DistanceType.Hamming))
                        using (VectorOfDMatch matches = new VectorOfDMatch())
                        {
                            matcher.Match(descriptors_1, descriptors_2, matches);

                            // Compute the average match distance
                            return matches.ToArray().Average(m => m.Distance); // Lower distance = better match
                        }
                    }

                }

            }

        }

        public static double ORBFeatureMatchingFactor_GPU(this Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null || !CudaInvoke.HasCuda)
            {
                return double.NaN;
            }

            // Convert the original Mat to grayscale
            using (Mat mat_gray_1 = new Mat())
            using (Mat mat_gray_2 = new Mat())
            {
                CvInvoke.CvtColor(mat_1, mat_gray_1, ColorConversion.Bgr2Gray);

                CvInvoke.CvtColor(mat_2, mat_gray_2, ColorConversion.Bgr2Gray);

                // Detect ORB keypoints and descriptors on CPU
                using (ORB orb = new ORB())
                using (VectorOfKeyPoint keyPoints_1 = new VectorOfKeyPoint())
                using (VectorOfKeyPoint keyPoints_2 = new VectorOfKeyPoint())
                using (Mat descriptors_1 = new Mat())
                using (Mat descriptors_2 = new Mat())
                {
                    orb.DetectAndCompute(mat_gray_1, null, keyPoints_1, descriptors_1, false);
                    orb.DetectAndCompute(mat_gray_2, null, keyPoints_2, descriptors_2, false);

                    // Check if descriptors are empty
                    if (descriptors_1.IsEmpty || descriptors_2.IsEmpty)
                    {
                        return double.NaN; // No descriptors found
                    }

                    // Ensure descriptors are of type CV_8U
                    if (descriptors_1.Depth != DepthType.Cv8U)
                    {
                        descriptors_1.ConvertTo(descriptors_1, DepthType.Cv8U);
                    }

                    if (descriptors_2.Depth != DepthType.Cv8U)
                    {
                        descriptors_2.ConvertTo(descriptors_2, DepthType.Cv8U);
                    }

                    // Upload descriptors to GPU
                    using (GpuMat gpuDescriptors_1 = new GpuMat(descriptors_1))
                    using (GpuMat gpuDescriptors_2 = new GpuMat(descriptors_2))

                    // Use CudaBFMatcher for feature matching (GPU)
                    using (CudaBFMatcher gpuMatcher = new CudaBFMatcher(DistanceType.Hamming))
                    using (VectorOfDMatch matches = new VectorOfDMatch())
                    {
                        // Match descriptors on GPU
                        gpuMatcher.Match(gpuDescriptors_1, gpuDescriptors_2, matches);

                        // Compute the average match distance
                        if (matches.Size > 0)
                        {
                            double averageDistance = matches.ToArray().Average(m => m.Distance); // Using the best match
                            return averageDistance; // Lower distance = better match
                        }
                    }
                    return double.NaN; // No matches found
                }


            }

        }
    }
}