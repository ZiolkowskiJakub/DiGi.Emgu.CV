using Emgu.CV.Structure;

namespace DiGi.Emgu.CV
{
    public static partial class Convert
    {
        /// <summary>
        /// Converts a <see cref="System.Drawing.Color"/> to an <see cref="MCvScalar"/>.
        /// </summary>
        /// <param name="color">The color value to convert.</param>
        /// <returns>An <see cref="MCvScalar"/> representing the BGR values of the specified color.</returns>
        public static MCvScalar ToEmguCV(this System.Drawing.Color color)
        {
            return new MCvScalar(color.B, color.G, color.R);
        }
    }
}