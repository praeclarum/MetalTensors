using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class AddLayerTests
    {
        [Test]
        public void ZeroPlusOne ()
        {
            var x0 = Tensor.Zeros (1);
            var x1 = Tensor.Ones (1);
            var y = x0 + x1;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.AreEqual (1, y[0]);
        }

        [Test]
        public void OnePlusOne ()
        {
            var x0 = Tensor.Ones (1);
            var x1 = Tensor.Ones (1);
            var y = x0 + x1;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.AreEqual (2, y[0]);
        }
    }
}
