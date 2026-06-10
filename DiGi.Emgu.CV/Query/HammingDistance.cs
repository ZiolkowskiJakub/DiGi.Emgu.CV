using Emgu.CV;
using System.Linq;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Calculates the Hamming distance between two matrices based on their average gray masks.
        /// </summary>
        /// <param name="mat_1">The first input matrix.</param>
        /// <param name="mat_2">The second input matrix.</param>
        /// <returns>The Hamming distance as an integer, or -1 if either input matrix is null.</returns>
        public static int HammingDistance(this Mat? mat_1, Mat? mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return -1;
            }

            bool[]? avreageGrayMask_1 = AverageGrayMask(mat_1);
            bool[]? avreageGrayMask_2 = AverageGrayMask(mat_2);

            return avreageGrayMask_1.Zip(avreageGrayMask_2, (value_1, value_2) => value_1 == value_2 ? 0 : 1).Sum();
        }
    }
}