using DiGi.Geometry.Planar;
using DiGi.Geometry.Planar.Classes;
using DiGi.Geometry.Planar.Interfaces;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Collections.Generic;
using System.Drawing;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        /// <summary>
        /// Fills a polygonal area within the specified image with a given color.
        /// </summary>
        /// <param name="mat">The source image to be filled.</param>
        /// <param name="polygonal2D">The polygonal shape defining the area to fill.</param>
        /// <param name="color">The color used for filling.</param>
        /// <param name="invert">A flag indicating whether to invert the mask, filling the area outside the polygon instead of inside.</param>
        /// <returns>A new image with the filled area, or null if the source image is null or the polygonal shape contains fewer than three valid points.</returns>
        public static Mat? Fill(this Mat? mat, IPolygonal2D? polygonal2D, Color color, bool invert = false)
        {
            List<Point2D>? point2Ds = polygonal2D?.GetPoints();
            if (point2Ds == null)
            {
                return null;
            }

            List<Point> points = [];
            foreach (Point2D point2D in point2Ds)
            {
                Point? point = point2D?.ToDrawing_Point();
                if (point == null || !point.HasValue)
                {
                    continue;
                }

                points.Add(point.Value);
            }

            if (points == null || points.Count < 3)
            {
                return null;
            }

            return Fill(mat, [.. points], color.ToEmguCV(), invert);
        }

        /// <summary>
        /// Fills a convex polygonal area defined by an array of points within the specified image with a given MCvScalar color.
        /// </summary>
        /// <param name="mat">The source image to be filled.</param>
        /// <param name="points">An array of points defining the vertices of the polygon.</param>
        /// <param name="mCvScalar">The Emgu CV scalar color used for filling.</param>
        /// <param name="invert">A flag indicating whether to invert the mask, filling the area outside the polygon instead of inside.</param>
        /// <returns>A new image with the filled area, or null if the source image is null or the points array contains fewer than three points.</returns>
        public static Mat? Fill(this Mat? mat, Point[]? points, MCvScalar mCvScalar, bool invert = false)
        {
            if (mat == null || points == null || points.Length < 3)
            {
                return null;
            }

            // Create a mask of the same size as the input Mat
            using Mat mask = new(mat.Size, DepthType.Cv8U, 1);
            mask.SetTo(new MCvScalar(0)); // Initialize the mask as black

            // Convert the polygon points to a VectorOfPoint
            using (var vectorOfPoint = new VectorOfPoint(points))
            {
                // Fill the polygon with white (255) on the mask
                CvInvoke.FillConvexPoly(mask, vectorOfPoint, new MCvScalar(255));
            }

            // If invert is true, invert the mask
            if (invert)
            {
                CvInvoke.BitwiseNot(mask, mask);
            }

            // Create a colored Mat filled with the specified MCvScalar color
            using Mat coloredMat = new(mat.Size, mat.Depth, mat.NumberOfChannels);
            coloredMat.SetTo(mCvScalar);

            // Apply the mask to the colored Mat
            using Mat maskedColor = new();

            CvInvoke.BitwiseAnd(coloredMat, coloredMat, maskedColor, mask);

            // Apply the inverted mask to the original image to keep the rest of the image
            using Mat inverseMask = new();

            CvInvoke.BitwiseNot(mask, inverseMask);

            using Mat maskedOriginal = new();

            CvInvoke.BitwiseAnd(mat, mat, maskedOriginal, inverseMask);

            // Combine the colored region with the masked original
            Mat result = new();
            CvInvoke.Add(maskedColor, maskedOriginal, result);

            return result;
        }
    }
}