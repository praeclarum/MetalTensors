using System;
using NUnit.Framework;

using MetalTensors;

namespace Tests.iOS
{
    public class PlaceholderTensorTests
    {
        [Test]
        public void Input ()
        {
            var t = Tensor.Input (1);
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (0.0f, t[0]);
        }

        [Test]
        public void Labels ()
        {
            var t = Tensor.Labels (1);
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (0.0f, t[0]);
        }
    }
}
