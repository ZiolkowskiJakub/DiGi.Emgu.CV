using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using System;
using System.Linq;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        //public static double ShapeComparisonFactor(this Mat mat_1, Mat mat_2, double threshold_1 = 100, double threshold_2 = 200)
        //{
        //    if(mat_1 == null || mat_2 == null)
        //    {
        //        return double.NaN;
        //    }

        //    Mat edges_1 = new Mat();
        //    CvInvoke.Canny(mat_1, edges_1, threshold_1, threshold_2); // Perform edge detection

        //    Mat edges_2 = new Mat();
        //    CvInvoke.Canny(mat_2, edges_2, threshold_1, threshold_2);

        //    // Compare edges (e.g., using XOR to count differences)
        //    Mat diff = new Mat();
        //    CvInvoke.BitwiseXor(edges_1, edges_2, diff);

        //    // Find contours
        //    VectorOfVectorOfPoint contours_1 = new VectorOfVectorOfPoint();
        //    Mat hierarchy_1 = new Mat();
        //    CvInvoke.FindContours(edges_1, contours_1, hierarchy_1, RetrType.External, ChainApproxMethod.ChainApproxSimple);

        //    VectorOfVectorOfPoint contours_2 = new VectorOfVectorOfPoint();
        //    Mat hierarchy_2 = new Mat();
        //    CvInvoke.FindContours(edges_2, contours_2, hierarchy_2, RetrType.External, ChainApproxMethod.ChainApproxSimple);

        //    // Compare shapes using Hu Moments

        //    Mat huMomentsMat_1 = new Mat(1, 7, DepthType.Cv64F, 1);
        //    CvInvoke.HuMoments(CvInvoke.Moments(contours_1[0]), huMomentsMat_1);

        //    double[] huMoments_1 = new double[7];

        //    huMomentsMat_1.CopyTo(huMoments_1);

        //    Mat huMomentsMat_2 = new Mat(1, 7, DepthType.Cv64F, 1);
        //    CvInvoke.HuMoments(CvInvoke.Moments(contours_2[0]), huMomentsMat_2);

        //    double[] huMoments_2 = new double[7];

        //    huMomentsMat_2.CopyTo(huMoments_2);

        //    // Calculate similarity (e.g., Euclidean distance)
        //    return huMoments_1.Zip(huMoments_2, (a, b) => Math.Abs(a - b)).Sum();
        //}

        public static double ShapeComparisonFactor(this Mat mat_1, Mat mat_2, double threshold_1 = 100, double threshold_2 = 200)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Perform edge detection
            using (Mat edges_1 = new Mat())
            using (Mat edges_2 = new Mat())
            {
                CvInvoke.Canny(mat_1, edges_1, threshold_1, threshold_2);
                CvInvoke.Canny(mat_2, edges_2, threshold_1, threshold_2);

                // Find contours
                using (VectorOfVectorOfPoint contours_1 = new VectorOfVectorOfPoint())
                using (Mat hierarchy_1 = new Mat())
                {
                    CvInvoke.FindContours(edges_1, contours_1, hierarchy_1, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                    using (VectorOfVectorOfPoint contours_2 = new VectorOfVectorOfPoint())
                    using (Mat hierarchy_2 = new Mat())
                    {
                        CvInvoke.FindContours(edges_2, contours_2, hierarchy_2, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                        // Ensure at least one contour exists in both images
                        if (contours_1.Size == 0 || contours_2.Size == 0)
                        {
                            return double.NaN; // No contours to compare
                        }

                        // Compute Hu Moments for the first contour in each image
                        using (Moments moments_1 = CvInvoke.Moments(contours_1[0]))
                        using (Moments moments_2 = CvInvoke.Moments(contours_2[0]))

                        using (Mat huMomentsMat_1 = new Mat(1, 7, DepthType.Cv64F, 1))
                        using (Mat huMomentsMat_2 = new Mat(1, 7, DepthType.Cv64F, 1))
                        {
                            CvInvoke.HuMoments(moments_1, huMomentsMat_1);
                            CvInvoke.HuMoments(moments_2, huMomentsMat_2);

                            // Convert the Hu Moments Mat to double arrays
                            double[] huMoments_1 = new double[7];
                            double[] huMoments_2 = new double[7];

                            huMomentsMat_1.CopyTo(huMoments_1);
                            huMomentsMat_2.CopyTo(huMoments_2);

                            // Calculate similarity (e.g., Euclidean distance)
                            return huMoments_1.Zip(huMoments_2, (a, b) => Math.Abs(a - b)).Sum();
                        }
                    }
                }
            }
        }
    }
}