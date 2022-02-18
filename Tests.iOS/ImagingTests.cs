using System;
using Foundation;
using NUnit.Framework;
using System.IO;

using MetalTensors;
using MetalTensors.Tensors;
using System.Runtime.CompilerServices;

using static Tests.Imaging;

namespace Tests
{
    public static class Imaging
    {
        readonly static string OutputDirectory;

        static Imaging ()
        {
            var solutionDir = Environment.CurrentDirectory;
            var ibin = solutionDir.LastIndexOf ("/bin/");
            if (ibin > 0) {
                solutionDir = Path.GetDirectoryName (solutionDir.Substring (0, ibin));
            }
            OutputDirectory = Path.Combine (solutionDir, "TestOutputImages");
            Directory.CreateDirectory (OutputDirectory);
        }

        public static string JpegPath ([CallerMemberName] string name = "Image", [CallerFilePath] string codeFile = "")
        {
            var codeName = System.IO.Path.GetFileNameWithoutExtension (codeFile);
            return Path.Combine (OutputDirectory, $"{codeName}_{name}.jpg");
        }

        public static NSUrl JpegUrl ([CallerMemberName] string name = "Image", [CallerFilePath] string codeFile = "")
        {
            return NSUrl.FromFilename (JpegPath (name: name, codeFile: codeFile));
        }

        public static string PngPath ([CallerMemberName] string name = "Image", [CallerFilePath] string codeFile = "")
        {
            var codeName = System.IO.Path.GetFileNameWithoutExtension (codeFile);
            return Path.Combine (OutputDirectory, $"{codeName}_{name}.png");
        }

        public static NSUrl PngUrl ([CallerMemberName] string name = "Image", [CallerFilePath] string codeFile = "")
        {
            return NSUrl.FromFilename (PngPath (name: name, codeFile: codeFile));
        }

        public static Tensor SaveModelJpeg (Tensor input, Tensor output, float a=1.0f, float b=0.0f, [CallerMemberName] string name = "ModelImage", [CallerFilePath] string codeFile = "")
        {
            var model = output.Model (input, trainable: false);
            return SaveModelJpeg (model, a, b, name: name, codeFile: codeFile);
        }

        public static Tensor SaveModelJpeg (Model model, float a=1.0f, float b=0.0f, [CallerMemberName] string name = "ModelImage", [CallerFilePath] string codeFile = "")
        {
            var input = GetImageInput (model.Input);
            var output = model.Predict (input);
            output.SaveImage (JpegUrl (name, codeFile));
            return output;
        }

        public static Tensor GetImageInput (Tensor input)
        {
            if (input.Shape[0] == 512)
                return Tensor.ImageResource ("elephant", "jpg");
            if (input.Shape[0] == 256) {
                var d = GetPix2pixDataSet ();
                var (ins, outs) = d.GetRow(0, MetalExtensions.Current(null));
                return ins[0];
            }
            throw new Exception ($"No sample image for shape {input.ShapeString}");
        }

        public static MetalTensors.Applications.Pix2pixApplication.Pix2pixDataSet GetPix2pixDataSet ()
        {
            var userDir = Environment.GetFolderPath (Environment.SpecialFolder.MyDocuments);
            var dataDir = Path.Combine (userDir, "Data", "datasets", "facades");
            var trainDataDir = Path.Combine (dataDir, "train");
            var data = MetalTensors.Applications.Pix2pixApplication.Pix2pixDataSet.LoadDirectory (trainDataDir);
            return data;
        }
    }

    public class ImagingTests
    {
        [Test]
        public void SaveReadJpeg ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            image.SaveImage (JpegUrl ());
            image.SaveImage (PngUrl ());
        }

        static readonly float[] rgbw = new float[] {
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
            1, 1, 1
        };

        [Test]
        public void RedGreenOnTopOfBlueWhite ()
        {
            var image = Tensor.Array (new[] { 2, 2, 3 }, rgbw);
            image.SaveImage (JpegUrl ());
            image.SaveImage (PngUrl ());
        }

        [Test]
        public void ElephantDark ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            image.SaveImage (JpegUrl (), 0.5f);
        }

        [Test]
        public void ElephantBright ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            image.SaveImage (JpegUrl (), 2.0f);
        }

        [Test]
        public void ElephantLowContrast ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            image.SaveImage (JpegUrl (), 0.5f, 0.25f);
        }

        [Test]
        public void ElephantHighContrast ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            image.SaveImage (JpegUrl (), 2.0f, 0.5f);
        }

        [Test]
        public void ElephantDarkLinear ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            var output = image.Linear (0.5f);
            output.SaveImage (JpegUrl ());
        }

        [Test]
        public void ElephantDarkLinearModel ()
        {
            var image = Tensor.ImageResource ("elephant", "jpg");
            var output = image.Linear (0.5f);
            var model = output.Model (image);
            var oimage = model.Predict (image);
            oimage.SaveImage (JpegUrl ());
        }
    }
}
