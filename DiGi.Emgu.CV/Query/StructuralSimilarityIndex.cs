using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double StructuralSimilarityIndex(this Mat mat_1, Mat mat_2)
        {
            if(mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            Mat mat_gray_1 = new Mat();
            CvInvoke.CvtColor(mat_1, mat_gray_1, ColorConversion.Bgr2Gray);

            Mat mat_gray_2 = new Mat();
            CvInvoke.CvtColor(mat_2, mat_gray_2, ColorConversion.Bgr2Gray);

            Mat absDiff = new Mat();
            CvInvoke.AbsDiff(mat_gray_1, mat_gray_2, absDiff);

            MCvScalar mean = CvInvoke.Mean(absDiff);
            return 1.0 - (mean.V0 / 255.0);
        }
    }
}