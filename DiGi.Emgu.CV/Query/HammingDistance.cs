using Emgu.CV;
using System.Linq;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static int HammingDistance(this Mat mat_1, Mat mat_2)
        {
            if (mat_1 == null || mat_2 == null)
            {
                return -1;
            }

            bool[] avreageGrayMask_1 = AverageGrayMask(mat_1);
            bool[] avreageGrayMask_2 = AverageGrayMask(mat_2);

            return avreageGrayMask_1.Zip(avreageGrayMask_2, (value_1, value_2) => value_1 == value_2 ? 0 : 1).Sum();
        }
    }
}