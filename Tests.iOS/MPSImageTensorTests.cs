using System;
using Foundation;
using NUnit.Framework;

using MetalTensors;
using MetalTensors.Tensors;

namespace Tests.iOS
{
    public class MPSImageTensorTests
    {
        [Test]
        public void FromUrl ()
        {
            var url = NSBundle.MainBundle.GetUrlForResource ("elephant", "jpg");
            var image = new MPSImageTensor (url);
            Assert.AreEqual (3, image.Shape.Length);
            Assert.AreEqual (512, image.Shape[0]);
            Assert.AreEqual (512, image.Shape[1]);
            Assert.AreEqual (3, image.Shape[2]);
        }

        [Test]
        public void CorrectDimensions ()
        {
            var image = Tensor.ReadImageResource ("rgbywb3x2", "png");
            Assert.AreEqual (2, image.Shape[0]);
            Assert.AreEqual (3, image.Shape[1]);
            Assert.AreEqual (3, image.Shape[2]);
        }

        [Test]
        public void CorrectColors ()
        {
            var image = Tensor.ReadImageResource ("rgbywb3x2", "png");
            Assert.AreEqual (2, image.Shape[0]);
            Assert.AreEqual (3, image.Shape[1]);
            Assert.AreEqual (3, image.Shape[2]);

            AssertColor (255,   0,   0, image.Slice (0, 0));
            AssertColor (  0, 255,   0, image.Slice (0, 1));
            AssertColor (  0,   0, 255, image.Slice (0, 2));
            AssertColor (255, 255,   0, image.Slice (1, 0));
            AssertColor (255, 255, 255, image.Slice (1, 1));
            AssertColor (  0,   0,   0, image.Slice (1, 2));
        }

        void AssertColor (float r, float g, float b, Tensor color)
        {
            Assert.AreEqual (1, color.Shape.Length);
            Assert.AreEqual (3, color.Shape[0]);
            Assert.AreEqual (r, color[0]);
            Assert.AreEqual (g, color[1]);
            Assert.AreEqual (b, color[2]);
        }

        [Test]
        public void ReadImageResourceIsMPSImage ()
        {
            var image = Tensor.ReadImageResource ("elephant", "jpg");
            Assert.IsInstanceOfType (typeof (MPSImageTensor), image);
        }

        [Test]
        public void ReadImageIsMPSImage ()
        {
            var path = NSBundle.MainBundle.PathForResource ("elephant", "jpg");
            var image = Tensor.ReadImage (path);
            Assert.IsInstanceOfType (typeof (MPSImageTensor), image);
        }
    }
}
