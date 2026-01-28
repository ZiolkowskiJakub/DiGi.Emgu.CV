using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double HistogramCorrelation(this Mat? mat_1, Mat? mat_2, bool accumulate)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Convert images to HSV
            using Mat hsvImg1 = new();
            using Mat hsvImg2 = new();

            CvInvoke.CvtColor(mat_1, hsvImg1, ColorConversion.Bgr2Hsv);
            CvInvoke.CvtColor(mat_2, hsvImg2, ColorConversion.Bgr2Hsv);

            // Compute histograms for hue channel
            using VectorOfMat hsvChannels1 = new();
            using VectorOfMat hsvChannels2 = new();

            CvInvoke.Split(hsvImg1, hsvChannels1);
            CvInvoke.Split(hsvImg2, hsvChannels2);

            using Mat hist1 = new();
            using Mat hist2 = new();

            CvInvoke.CalcHist(new VectorOfMat(hsvChannels1[0]), [0], null, hist1, [256], [0, 256], accumulate);
            CvInvoke.CalcHist(new VectorOfMat(hsvChannels2[0]), [0], null, hist2, [256], [0, 256], accumulate);

            // Normalize and compare
            CvInvoke.Normalize(hist1, hist1, 0, 1, NormType.MinMax);
            CvInvoke.Normalize(hist2, hist2, 0, 1, NormType.MinMax);

            return CvInvoke.CompareHist(hist1, hist2, HistogramCompMethod.Correl);
        }
    }
}