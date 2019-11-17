using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class ConcatLayerTests
    {
        [Test]
        public void OneAndOne ()
        {
            var one = Tensor.Ones ();
            var two = Tensor.Constant (2.0f);
            var c = one.Concat (two);

            Assert.AreEqual (1, c.Shape.Length);
            Assert.AreEqual (8, c.Shape[0]);
            Assert.AreEqual (1.0, c[0], 1.0e-6f);
            Assert.AreEqual (0.0, c[1], 1.0e-6f);
            Assert.AreEqual (2.0, c[4], 1.0e-6f);
            Assert.AreEqual (0.0, c[6], 1.0e-6f);
        }

        [Test]
        public void SevenAndEleven ()
        {
            var one = Tensor.Ones (3, 5, 7);
            var two = Tensor.Constant (2.0f, 3, 5, 11);
            var c = one.Concat (two);

            Assert.AreEqual (3, c.Shape.Length);
            Assert.AreEqual (3, c.Shape[0]);
            Assert.AreEqual (5, c.Shape[1]);
            Assert.AreEqual (20, c.Shape[2]);
            Assert.AreEqual (1.0, c[0, 0, 0], 1.0e-6f);
            Assert.AreEqual (1.0, c[2, 4, 6], 1.0e-6f);
            Assert.AreEqual (0.0, c[2, 4, 7], 1.0e-6f);
            Assert.AreEqual (2.0, c[2, 4, 8], 1.0e-6f);
            Assert.AreEqual (2.0, c[2, 4, 9], 1.0e-6f);
            Assert.AreEqual (2.0, c[2, 4, 18], 1.0e-6f);
            Assert.AreEqual (0.0, c[2, 4, 19], 1.0e-6f);
        }
    }
}
