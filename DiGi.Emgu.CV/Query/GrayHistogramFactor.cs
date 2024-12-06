using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double GrayHistogramFactor(this Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Convert input Mats to grayscale
            Mat gray1 = new Mat();
            Mat gray2 = new Mat();
            CvInvoke.CvtColor(mat_1, gray1, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(mat_2, gray2, ColorConversion.Bgr2Gray);

            // Initialize histogram Mats
            Mat hist1 = new Mat();
            Mat hist2 = new Mat();

            // Parameters for histogram calculation
            int[] channels = { 0 }; // Channel to process (grayscale)
            int[] histSize = { 256 }; // Number of bins
            float[] ranges = { 0, 256 }; // Intensity range

            // Use VectorOfMat to wrap the grayscale Mats
            using (VectorOfMat vectorOfMat_1 = new VectorOfMat())
            {
                using (VectorOfMat vectorOfMat_2 = new VectorOfMat())
                {
                    vectorOfMat_1.Push(gray1);
                    vectorOfMat_2.Push(gray2);

                    // Calculate histograms
                    CvInvoke.CalcHist(vectorOfMat_1, channels, null, hist1, histSize, ranges, false);
                    CvInvoke.CalcHist(vectorOfMat_2, channels, null, hist2, histSize, ranges, false);
                }
            }

            // Normalize histograms
            CvInvoke.Normalize(hist1, hist1, 0, 1, NormType.MinMax);
            CvInvoke.Normalize(hist2, hist2, 0, 1, NormType.MinMax);

            // Compare histograms using correlation
            return CvInvoke.CompareHist(hist1, hist2, HistogramCompMethod.Correl);
        }
    }
}