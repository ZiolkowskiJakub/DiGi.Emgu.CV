using Emgu.CV;
using Emgu.CV.CvEnum;
using System.Collections.Generic;

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

            // Convert the Mat to grayscale
            Mat mat_gray_1 = new Mat();
            CvInvoke.CvtColor(mat, mat_gray_1, ColorConversion.Bgr2Gray);

            // Calculate the mean intensity
            double mean = CvInvoke.Mean(mat_gray_1).V0;

            // Get image dimensions
            int rows = mat_gray_1.Rows;
            int cols = mat_gray_1.Cols;

            // Get pixel data as a flattened array
            byte[] data = new byte[rows * cols];
            mat_gray_1.CopyTo(data); // Safely copy Mat data to a byte array

            // Create the mask
            List<bool> mask = new List<bool>();
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < cols; x++)
                {
                    // Calculate the index in the flattened array
                    int index = y * cols + x;
                    byte pixel = data[index]; // Access pixel value using the index
                    mask.Add(pixel > mean);
                }
            }

            return mask.ToArray();
        }
    }
}