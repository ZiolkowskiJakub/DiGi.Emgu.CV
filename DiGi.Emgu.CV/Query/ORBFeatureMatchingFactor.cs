using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Util;
using System.Linq;

namespace DiGi.Emgu.CV
{
    public static partial class Query
    {
        public static double ORBFeatureMatchingFactor(this Mat mat_1, Mat mat_2)
        {
            if(mat_1 == null || mat_2 == null)
            {
                return double.NaN;
            }

            // Convert the original Mat to grayscale
            
            Mat mat_gray_1 = new Mat();
            CvInvoke.CvtColor(mat_1, mat_gray_1, ColorConversion.Bgr2Gray);

            Mat mat_gray_2 = new Mat();
            CvInvoke.CvtColor(mat_2, mat_gray_2, ColorConversion.Bgr2Gray);

            // Load images

            UMat uMat_1 = new UMat();
            mat_gray_1.CopyTo(uMat_1);

            UMat uMat_2 = new UMat();
            mat_gray_2.CopyTo(uMat_2);

            // Detect ORB keypoints and descriptors

            ORB orb = new ORB();
            VectorOfKeyPoint keyPoints_1 = new VectorOfKeyPoint();
            VectorOfKeyPoint keyPoints_2 = new VectorOfKeyPoint();
            Mat descriptors_1 = new Mat();
            Mat descriptors_2 = new Mat();

            orb.DetectAndCompute(uMat_1, null, keyPoints_1, descriptors_1, false);
            orb.DetectAndCompute(uMat_2, null, keyPoints_2, descriptors_2, false);

            // Match features using Brute-Force Matcher
            BFMatcher matcher = new BFMatcher(DistanceType.Hamming);
            
            VectorOfDMatch matches = new VectorOfDMatch();
            matcher.Match(descriptors_1, descriptors_2, matches);

            return matches.ToArray().Average(m => m.Distance); // Lower distance = better match
        }
    }
}