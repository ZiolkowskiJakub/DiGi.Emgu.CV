using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static void LaplacianFactors(this Mat mat, out double mean, out double standardDeviation)
        {
            Mat gray1 = new Mat();

            CvInvoke.CvtColor(mat, gray1, ColorConversion.Bgr2Gray);

            Mat noise = new Mat();
            CvInvoke.Laplacian(mat, noise, DepthType.Cv64F);

            MCvScalar mCvScalar_Mean = new MCvScalar();
            MCvScalar MCvScalar_StandardDeviation = new MCvScalar();

            CvInvoke.MeanStdDev(noise, ref mCvScalar_Mean, ref MCvScalar_StandardDeviation);

            mean = mCvScalar_Mean.V0;
            standardDeviation = MCvScalar_StandardDeviation.V0;
        }

        public static void LaplacianFactors(this Mat mat_1, Mat mat_2, out double mean, out double standardDeviation)
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

            if(!double.IsNaN(mean_1) && !double.IsNaN(mean_2))
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