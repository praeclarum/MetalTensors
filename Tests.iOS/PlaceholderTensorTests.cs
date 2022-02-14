using System;
using NUnit.Framework;

using MetalTensors;

namespace Tests
{
    public class PlaceholderTensorTests
    {
        [Test]
        public void DefaultInputShape ()
        {
            var t = Tensor.Input ("foo");
            Assert.AreEqual ("foo", t.Label);
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
        }

        [Test]
        public void Input ()
        {
            var t = Tensor.Input ("foo", 1);
            Assert.AreEqual ("foo", t.Label);
            Assert.AreEqual (1, t.Shape.Length);
            Assert.AreEqual (1, t.Shape[0]);
            Assert.AreEqual (0.0f, t[0]);
        }

        [Test]
        public void InputImage ()
        {
            var t = Tensor.InputImage ("foo", 7, 5);
            Assert.AreEqual ("foo", t.Label);
            Assert.AreEqual (3, t.Shape.Length);
            Assert.AreEqual (7, t.Shape[0]);
            Assert.AreEqual (5, t.Shape[1]);
            Assert.AreEqual (3, t.Shape[2]);
            Assert.AreEqual (0.0f, t[0]);
        }
    }
}
