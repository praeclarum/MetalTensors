using System;
using NUnit.Framework;

using MetalTensors.Tensors;

namespace Tests.iOS
{
    public class ZeroTensorTests
    {
        [Test]
        public void Single ()
        {
            var t = new ZeroTensor (1);
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (0.0f, t.Item);
        }
    }
}
