using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Drawing;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Attempts to match a template image within a target image using normalized cross-correlation and retrieves the minimum and maximum values and their locations.
        /// </summary>
        /// <param name="mat_Target">The source image in which to search for the template.</param>
        /// <param name="mat_Template">The template image to be matched.</param>
        /// <param name="minValue">When this method returns, contains the minimum value of the match result.</param>
        /// <param name="maxValue">When this method returns, contains the maximum value of the match result.</param>
        /// <param name="minPoint">When this method returns, contains the location of the minimum value in the match result.</param>
        /// <param name="maxPoint">When this method returns, contains the location of the maximum value in the match result.</param>
        /// <returns>True if the matching process was successful; otherwise, false.</returns>
        public static bool TryMatchLocation(this Mat? mat_Target, Mat? mat_Template, out double minValue, out double maxValue, out Point minPoint, out Point maxPoint)
        {
            minValue = double.NaN;
            maxValue = double.NaN;

            if (mat_Target == null || mat_Template == null)
            {
                return false;
            }

            using Mat mat = new();

            CvInvoke.MatchTemplate(mat_Target, mat_Template, mat, TemplateMatchingType.CcorrNormed);

            CvInvoke.MinMaxLoc(mat, ref minValue, ref maxValue, ref minPoint, ref maxPoint);

            return true;
        }
    }
}