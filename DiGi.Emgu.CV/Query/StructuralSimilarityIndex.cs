using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Calculates a similarity index between two matrices based on the absolute difference of their grayscale representations.
        /// </summary>
        /// <param name="mat_1">The first input matrix.</param>
        /// <param name="mat_2">The second input matrix.</param>
        /// <returns>A double value representing the similarity index, or <see cref="double.NaN"/> if either input matrix is null.</returns>
        public static double StructuralSimilarityIndex_AbsoluteDifference(this Mat? mat_1, Mat? mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            using Mat mat_gray_1 = new();
            using Mat mat_gray_2 = new();
            using Mat absDiff = new();

            CvInvoke.CvtColor(mat_1, mat_gray_1, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(mat_2, mat_gray_2, ColorConversion.Bgr2Gray);
            CvInvoke.AbsDiff(mat_gray_1, mat_gray_2, absDiff);

            MCvScalar mean = CvInvoke.Mean(absDiff);
            return 1.0 - (mean.V0 / 255.0);
        }

        /// <summary>
        /// Calculates a similarity index between two matrices using template matching with normalized cross-correlation.
        /// </summary>
        /// <param name="mat_1">The first input matrix.</param>
        /// <param name="mat_2">The second input matrix.</param>
        /// <returns>A double value representing the similarity index, or <see cref="double.NaN"/> if either input matrix is null.</returns>
        public static double StructuralSimilarityIndex_MatchTemplate(this Mat? mat_1, Mat? mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            using Mat mat_gray_1 = new();
            using Mat mat_gray_2 = new();
            using Mat mat_diff = new();

            CvInvoke.CvtColor(mat_1, mat_gray_1, ColorConversion.Bgr2Gray);

            CvInvoke.CvtColor(mat_2, mat_gray_2, ColorConversion.Bgr2Gray);

            CvInvoke.MatchTemplate(mat_gray_1, mat_gray_2, mat_diff, TemplateMatchingType.CcorrNormed);

            MCvScalar mean = CvInvoke.Mean(mat_diff);
            return 1.0 - (mean.V0 / 255.0);
        }
    }
}