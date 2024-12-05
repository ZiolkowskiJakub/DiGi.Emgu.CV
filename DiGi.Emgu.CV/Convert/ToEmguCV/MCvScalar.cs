using Emgu.CV.Structure;

namespace DiGi.Emgu.CV
{
    public static partial class Convert
    {
        public static MCvScalar ToEmguCV(this System.Drawing.Color color)
        {
            return new MCvScalar(color.B, color.G, color.R);
        }
    }

}


