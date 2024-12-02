using Emgu.CV;
using Emgu.CV.Structure;
using System;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double AverageColorSimilarity(this Mat mat_1, Mat mat_2)
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
    }
}