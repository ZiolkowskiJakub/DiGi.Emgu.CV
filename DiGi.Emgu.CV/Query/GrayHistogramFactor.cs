using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Calculates the correlation factor between the grayscale histograms of two image matrices.
        /// </summary>
        /// <param name="mat_1">The first input image matrix.</param>
        /// <param name="mat_2">The second input image matrix.</param>
        /// <returns>The histogram correlation value as a double, or <see cref="double.NaN"/> if either input matrix is null.</returns>
        public static double GrayHistogramFactor(this Mat? mat_1, Mat? mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Convert input Mats to grayscale
            using Mat gray1 = new();
            using Mat gray2 = new();

            CvInvoke.CvtColor(mat_1, gray1, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(mat_2, gray2, ColorConversion.Bgr2Gray);

            // Initialize histogram Mats
            using Mat hist1 = new();
            using Mat hist2 = new();

            // Parameters for histogram calculation
            int[] channels = [0]; // Channel to process (grayscale)
            int[] histSize = [256]; // Number of bins
            float[] ranges = [0, 256]; // Intensity range

            // Use VectorOfMat to wrap the grayscale Mats
            using (VectorOfMat vectorOfMat_1 = new())
            {
                using VectorOfMat vectorOfMat_2 = new();

                vectorOfMat_1.Push(gray1);
                vectorOfMat_2.Push(gray2);

                // Calculate histograms
                CvInvoke.CalcHist(vectorOfMat_1, channels, null, hist1, histSize, ranges, false);
                CvInvoke.CalcHist(vectorOfMat_2, channels, null, hist2, histSize, ranges, false);
            }

            // Normalize histograms
            CvInvoke.Normalize(hist1, hist1, 0, 1, NormType.MinMax);
            CvInvoke.Normalize(hist2, hist2, 0, 1, NormType.MinMax);

            // Compare histograms using correlation
            return CvInvoke.CompareHist(hist1, hist2, HistogramCompMethod.Correl);
        }
    }
}