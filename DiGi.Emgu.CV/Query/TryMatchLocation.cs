using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Drawing;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static bool TryMatchLocation(this Mat mat_Target, Mat mat_Template, out double minValue, out double maxValue, out Point minPoint, out Point maxPoint)
        {
            minValue = double.NaN;
            maxValue = double.NaN;

            if (mat_Target == null || mat_Template == null)
            {
                return false;
            }

            using (Mat mat = new Mat())
            {
                CvInvoke.MatchTemplate(mat_Target, mat_Template, mat, TemplateMatchingType.CcorrNormed);

                CvInvoke.MinMaxLoc(mat, ref minValue, ref maxValue, ref minPoint, ref maxPoint);
            }

            return true;
        }
    }
}