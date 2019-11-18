using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class ConvTransposeLayerTests
    {
        [Test]
        public void Defaults ()
        {
            var image = Tensor.InputImage ("image", 5, 7, 3);
            var conv = image.ConvTranspose (32);

            Assert.AreEqual (3, conv.Shape.Length);
            Assert.AreEqual (5, conv.Shape[0]);
            Assert.AreEqual (7, conv.Shape[1]);
            Assert.AreEqual (32, conv.Shape[2]);

            Assert.IsTrue (conv[0, 0, 0] > -10.0f);
        }

        [Test]
        public void Stride2 ()
        {
            var image = Tensor.Ones (5, 7, 3);
            var conv = image.ConvTranspose (32, 4, stride: 2);

            Assert.AreEqual (3, conv.Shape.Length);
            Assert.AreEqual (10, conv.Shape[0]);
            Assert.AreEqual (14, conv.Shape[1]);
            Assert.AreEqual (32, conv.Shape[2]);

            Assert.IsTrue (conv[0, 0, 0] > -10.0f);
        }
    }
}
