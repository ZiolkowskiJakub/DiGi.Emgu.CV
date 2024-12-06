using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double ColorHistogramFactor(this Mat mat_1, Mat mat_2)
        {
            if(mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Convert to HSV (or another color space if needed)
            Mat hsvImage_1 = new Mat();
            CvInvoke.CvtColor(mat_1, hsvImage_1, ColorConversion.Bgr2Hsv);

            // Calculate histograms
            VectorOfMat hsvChannels_1 = new VectorOfMat();
            CvInvoke.Split(hsvImage_1, hsvChannels_1);

            Mat hist1 = new Mat();
            CvInvoke.CalcHist(new VectorOfMat(hsvChannels_1[0]), new int[] { 0 }, null, hist1, new int[] { 256 }, new float[] { 0, 256 }, false);

            // Normalize the histogram
            CvInvoke.Normalize(hist1, hist1, 0, 1, NormType.MinMax);

            // Convert to HSV (or another color space if needed)
            Mat hsvImage_2 = new Mat();
            CvInvoke.CvtColor(mat_2, hsvImage_2, ColorConversion.Bgr2Hsv);

            // Calculate histograms
            VectorOfMat hsvChannels_2 = new VectorOfMat();
            CvInvoke.Split(hsvImage_2, hsvChannels_2);

            Mat hist2 = new Mat();
            CvInvoke.CalcHist(new VectorOfMat(hsvChannels_1[0]), new int[] { 0 }, null, hist2, new int[] { 256 }, new float[] { 0, 256 }, false);

            // Normalize the histogram
            CvInvoke.Normalize(hist2, hist2, 0, 1, NormType.MinMax);

            return CvInvoke.CompareHist(hist1, hist2, HistogramCompMethod.Correl); // Correlation similarity
        }
    }
}