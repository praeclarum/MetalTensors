using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class MaxPoolLayerTests
    {
        [Test]
        public void Defaults ()
        {
            var image = Tensor.ReadImageResource ("elephant", "jpg");
            var output = image.MaxPool ();

            Assert.AreEqual (3, output.Shape.Length);
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (3, output.Shape[2]);

            Assert.IsTrue (output[0,0,0] > -10.0f);
        }

        [Test]
        public void Stride2 ()
        {
            var image = Tensor.ReadImageResource ("elephant", "jpg");
            var output = image.MaxPool (2, 2);

            Assert.AreEqual (3, output.Shape.Length);
            Assert.AreEqual (256, output.Shape[0]);
            Assert.AreEqual (256, output.Shape[1]);
            Assert.AreEqual (3, output.Shape[2]);

            Assert.IsTrue (output[0,0,0] > -10.0f);
        }
    }
}
