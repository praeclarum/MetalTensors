using System;
using Foundation;
using NUnit.Framework;
using System.IO;

using MetalTensors;
using MetalTensors.Tensors;
using System.Runtime.CompilerServices;

namespace Tests
{
    public class ImagingTests
    {
        string OutputDirectory;

        public ImagingTests ()
        {
            var solutionDir = Environment.CurrentDirectory;
            var ibin = solutionDir.LastIndexOf ("/bin/");
            if (ibin > 0) {
                solutionDir = Path.GetDirectoryName(solutionDir.Substring (0, ibin));
            }
            OutputDirectory = Path.Combine (solutionDir, "TestOutputImages");
            Directory.CreateDirectory (OutputDirectory);
        }

        string JpegPath ([CallerMemberName] string name = "Image")
        {
            return Path.Combine (OutputDirectory, $"{name}.jpg");
        }

        NSUrl JpegUrl ([CallerMemberName] string name = "Image")
        {
            return NSUrl.FromFilename (JpegPath (name));
        }

        string PngPath ([CallerMemberName] string name = "Image")
        {
            return Path.Combine (OutputDirectory, $"{name}.png");
        }

        NSUrl PngUrl ([CallerMemberName] string name = "Image")
        {
            return NSUrl.FromFilename (PngPath (name));
        }

        [Test]
        public void SaveReadJpeg ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            image.SaveImage (JpegUrl ());
            image.SaveImage (PngUrl ());
        }

        [Test]
        public void RedGreenOnTopOfBlueWhite ()
        {
            var rgbw = new float[] {
                1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                1, 1, 1
            };
            var image = Tensor.Array (new[] { 2, 2, 3 }, rgbw);
            image.SaveImage (JpegUrl ());
            image.SaveImage (PngUrl ());
        }
    }
}
