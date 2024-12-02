﻿using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Text;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static bool[] AverageGrayMask(this Mat mat)
        {
            if (mat == null)
            {
                return null;
            }

            Mat mat_gray_1 = new Mat();
            CvInvoke.CvtColor(mat, mat_gray_1, ColorConversion.Bgr2Gray);

            double mean = CvInvoke.Mean(mat_gray_1).V0;
            StringBuilder hash = new StringBuilder();

            Array array = mat_gray_1.GetData();
            List<bool> mask = new List<bool>();
            for (int y = 0; y < mat_gray_1.Rows; y++)
            {
                for (int x = 0; x < mat_gray_1.Cols; x++)
                {
                    double pixel = Convert.ToDouble(array.GetValue(x, y)); // Access pixel value safely
                    mask.Add(pixel > mean ? true : false);
                }
            }

            return mask.ToArray();
        }
    }
}