using System;
using Foundation;
using NUnit.Framework;

using MetalTensors;
using MetalTensors.Tensors;

using static Tests.Imaging;

namespace Tests
{
    public class UpsampleLayerTests
    {
        [Test]
        public void Double ()
        {
            var image = Tensor.ImageResource ("rgbywb3x2", "png");

            var result = image.Upsample (2, 2);

            result.SaveImage (PngUrl ());

            Assert.AreEqual (4, result.Shape[0]);
            Assert.AreEqual (6, result.Shape[1]);
            Assert.AreEqual (3, result.Shape[2]);

            Assert.AreEqual (1, result[0, 0, 0]);
            Assert.AreEqual (0, result[0, 0, 1]);
            Assert.AreEqual (0, result[0, 0, 2]);

            Assert.AreEqual (1, result[0, 1, 0]);
            Assert.AreEqual (0, result[0, 1, 1]);
            Assert.AreEqual (0, result[0, 1, 2]);

            Assert.AreEqual (0, result[0, 2, 0]);
            Assert.AreEqual (1, result[0, 2, 1]);
            Assert.AreEqual (0, result[0, 2, 2]);
        }

        [Test]
        public void DoubleBilinear ()
        {
            var image = Tensor.ImageResource ("rgbywb3x2", "png");

            var result = image.UpsampleBilinear (2, 2);

            result.SaveImage (PngUrl ());

            Assert.AreEqual (4, result.Shape[0]);
            Assert.AreEqual (6, result.Shape[1]);
            Assert.AreEqual (3, result.Shape[2]);

            Assert.AreEqual (1, result[0, 0, 0]);
            Assert.AreEqual (0, result[0, 0, 1]);
            Assert.AreEqual (0, result[0, 0, 2]);

            Assert.AreEqual (0.5f, result[0, 1, 0]);
            Assert.AreEqual (0.5f, result[0, 1, 1]);
            Assert.AreEqual (0.0f, result[0, 1, 2]);

            Assert.AreEqual (0, result[0, 2, 0]);
            Assert.AreEqual (1, result[0, 2, 1]);
            Assert.AreEqual (0, result[0, 2, 2]);
        }
    }
}
