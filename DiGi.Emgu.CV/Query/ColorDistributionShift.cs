using Emgu.CV;
using Emgu.CV.Structure;
using System;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double ColorDistributionShift(this Mat mat_1, Mat mat_2)
        {
            if(mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            MCvScalar mean1 = CvInvoke.Mean(mat_1);
            MCvScalar mean2 = CvInvoke.Mean(mat_2);

            return Math.Sqrt(
                Math.Pow(mean1.V0 - mean2.V0, 2) + // Blue
                Math.Pow(mean1.V1 - mean2.V1, 2) + // Green
                Math.Pow(mean1.V2 - mean2.V2, 2)   // Red
            );
        }
    }
}