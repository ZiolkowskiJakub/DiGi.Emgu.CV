using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Calculates the mean and standard deviation of the Laplacian of the specified matrix.
        /// </summary>
        /// <param name="mat">The input matrix to process.</param>
        /// <param name="mean">When this method returns, contains the calculated mean value.</param>
        /// <param name="standardDeviation">When this method returns, contains the calculated standard deviation value.</param>
        public static void LaplacianFactors(this Mat? mat, out double mean, out double standardDeviation)
        {
            using Mat gray1 = new();
            using Mat noise = new();

            CvInvoke.CvtColor(mat, gray1, ColorConversion.Bgr2Gray);
            CvInvoke.Laplacian(mat, noise, DepthType.Cv64F);

            MCvScalar mCvScalar_Mean = new();
            MCvScalar MCvScalar_StandardDeviation = new();

            CvInvoke.MeanStdDev(noise, ref mCvScalar_Mean, ref MCvScalar_StandardDeviation);

            mean = mCvScalar_Mean.V0;
            standardDeviation = MCvScalar_StandardDeviation.V0;
        }

        /// <summary>
        /// Calculates the ratio of Laplacian factors between two specified matrices.
        /// </summary>
        /// <param name="mat_1">The first input matrix.</param>
        /// <param name="mat_2">The second input matrix used as the divisor for the ratio.</param>
        /// <param name="mean">When this method returns, contains the ratio of the means of the two matrices.</param>
        /// <param name="standardDeviation">When this method returns, contains the ratio of the standard deviations of the two matrices.</param>
        public static void LaplacianFactors(this Mat? mat_1, Mat? mat_2, out double mean, out double standardDeviation)
        {
            mean = double.NaN;
            standardDeviation = double.NaN;

            LaplacianFactors(mat_1, out double mean_1, out double standardDeviation_1);
            if (double.IsNaN(mean_1) && double.IsNaN(standardDeviation_1))
            {
                return;
            }

            LaplacianFactors(mat_2, out double mean_2, out double standardDeviation_2);
            if (double.IsNaN(mean_2) && double.IsNaN(standardDeviation_1))
            {
                return;
            }

            if (!double.IsNaN(mean_1) && !double.IsNaN(mean_2))
            {
                mean = mean_2 == 0 ? 0 : mean_1 / mean_2;
            }

            if (!double.IsNaN(standardDeviation_1) && !double.IsNaN(standardDeviation_2))
            {
                standardDeviation = standardDeviation_2 == 0 ? 0 : standardDeviation_1 / standardDeviation_2;
            }
        }
    }
}