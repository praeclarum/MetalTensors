using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class ConvLayerTests
    {
        [Test]
        public void Defaults ()
        {
            var image = Tensor.ReadImageResource ("elephant", "jpg");
            var conv = image.Conv (32, 3);

            Assert.AreEqual (3, conv.Shape.Length);
            Assert.AreEqual (512, conv.Shape[0]);
            Assert.AreEqual (512, conv.Shape[1]);
            Assert.AreEqual (32, conv.Shape[2]);

            Assert.IsTrue (conv[0,0,0] > -10.0f);
        }
    }
}
