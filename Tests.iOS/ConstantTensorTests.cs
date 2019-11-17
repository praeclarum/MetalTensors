using System;
using NUnit.Framework;

using MetalTensors;

namespace Tests.iOS
{
    public class ConstantTensorTests
    {
        [Test]
        public void OnesNone ()
        {
            var t = Tensor.Ones ();
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (1.0f, t[0]);
        }

        [Test]
        public void ZerosSingle ()
        {
            var t = Tensor.Zeros (1);
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (0.0f, t[0]);
        }

        [Test]
        public void ZerosFirstOfThree ()
        {
            var t = Tensor.Zeros (3);
            Assert.AreEqual (0.0f, t[0]);
        }
    }
}
