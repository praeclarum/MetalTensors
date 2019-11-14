using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class DenseLayerTests
    {
        [Test]
        public void Defaults ()
        {
            var image = Tensor.ReadImageResource ("elephant", "jpg");
            var output = image.Dense (32);

            Assert.AreEqual (3, output.Shape.Length);
            Assert.AreEqual (512, output.Shape[0]);
            Assert.AreEqual (512, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);

            Assert.IsTrue (output[0,0,0] > -10.0f);
        }

        [Test]
        public void NoBias ()
        {
            var image = Tensor.ReadImageResource ("rgbywb3x2", "png");
            var output = image.Dense (32, bias: false);

            Assert.AreEqual (3, output.Shape.Length);
            Assert.AreEqual (2, output.Shape[0]);
            Assert.AreEqual (3, output.Shape[1]);
            Assert.AreEqual (32, output.Shape[2]);

            Assert.IsTrue (output[0, 0, 0] > -10.0f);
        }
    }
}
