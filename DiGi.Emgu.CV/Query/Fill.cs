using DiGi.Geometry.Planar;
using DiGi.Geometry.Planar.Classes;
using DiGi.Geometry.Planar.Interfaces;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static Mat Fill(this Mat mat, IPolygonal2D polygonal2D, Color color, bool invert = false)
        {
            List<Point2D> point2Ds = polygonal2D?.GetPoints();
            if(point2Ds == null)
            {
                return null;
            }

            List<Point> points = new List<Point>();
            foreach(Point2D point2D in point2Ds)
            {
                Point? point = point2D?.ToDrawing_Point();
                if(point == null || !point.HasValue)
                {
                    continue;
                }

                points.Add(point.Value);
            }

            if(points == null || points.Count < 3)
            {
                return null;
            }

            return Fill(mat, points.ToArray(), color.ToEmguCV(), invert);
        }

        public static Mat Fill(this Mat mat, Point[] points, MCvScalar mCvScalar, bool invert = false)
        {
            if (mat == null || points == null || points.Length < 3)
            {
                return null;
            }

            // Create a mask of the same size as the input Mat
            Mat mask = new Mat(mat.Size, DepthType.Cv8U, 1);
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
            Mat coloredMat = new Mat(mat.Size, mat.Depth, mat.NumberOfChannels);
            coloredMat.SetTo(mCvScalar);

            // Apply the mask to the colored Mat
            Mat maskedColor = new Mat();
            CvInvoke.BitwiseAnd(coloredMat, coloredMat, maskedColor, mask);

            // Apply the inverted mask to the original image to keep the rest of the image
            Mat inverseMask = new Mat();
            CvInvoke.BitwiseNot(mask, inverseMask);

            Mat maskedOriginal = new Mat();
            CvInvoke.BitwiseAnd(mat, mat, maskedOriginal, inverseMask);

            // Combine the colored region with the masked original
            Mat result = new Mat();
            CvInvoke.Add(maskedColor, maskedOriginal, result);

            return result;
        }
    }
}