using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class BinaryArithmeticLayerTests
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
        public void ZeroMinusOne ()
        {
            var x0 = Tensor.Zeros (1);
            var x1 = Tensor.Ones (1);
            var y = x0 - x1;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.AreEqual (-1, y[0]);
        }

        [Test]
        public void ZeroTimesOne ()
        {
            var x0 = Tensor.Zeros (1);
            var x1 = Tensor.Ones (1);
            var y = x0 * x1;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.AreEqual (0, y[0]);
        }

        [Test]
        public void ZeroDividedByOne ()
        {
            var x0 = Tensor.Zeros (1);
            var x1 = Tensor.Ones (1);
            var y = x0 / x1;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.AreEqual (0, y[0]);
        }

        [Test]
        public void ZeroDividedByZero ()
        {
            var x0 = Tensor.Zeros (1);
            var x1 = Tensor.Zeros (1);
            var y = x0 / x1;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.IsTrue (float.IsNaN (y[0]));
        }

        [Test]
        public void OneDividedByZero ()
        {
            var xa = Tensor.Ones (1);
            var xb = Tensor.Zeros (1);
            var y = xa / xb;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.IsTrue (float.IsPositiveInfinity (y[0]));
        }

        [Test]
        public void NegativeOneDividedByZero ()
        {
            var xa = Tensor.Constant (-1, 1);
            var xb = Tensor.Zeros (1);
            var y = xa / xb;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (1, y.Shape[0]);
            Assert.IsTrue (float.IsNegativeInfinity (y[0]));
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

        [Test]
        public void FortyPlusTwo ()
        {
            var x0 = Tensor.Constant (40.0f, 4);
            var x1 = Tensor.Constant (2.0f, 4);
            var y = x0 + x1;
            Assert.AreEqual (1, y.Shape.Length);
            Assert.AreEqual (4, y.Shape[0]);
            Assert.AreEqual (42.0f, y[0]);
            Assert.AreEqual (42.0f, y[1]);
            Assert.AreEqual (42.0f, y[2]);
            Assert.AreEqual (42.0f, y[3]);
        }
    }
}
