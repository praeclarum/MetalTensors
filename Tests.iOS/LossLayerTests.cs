using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class LossLayerTests
    {
        [Test]
        public void MSE ()
        {
            var x = Tensor.Constant (800, 1);
            var y = Tensor.Constant (1000, 1);
            var loss = x.Loss (y, LossType.MeanSquaredError);

            Assert.AreEqual (1, loss.Shape.Length);
            Assert.AreEqual (1, loss.Shape[0]);

            Assert.AreEqual (Math.Pow(x[0] - y[0], 2.0), loss[0]);
        }

        [Test]
        public void MAE ()
        {
            var x = Tensor.Constant (800, 1);
            var y = Tensor.Constant (1000, 1);
            var loss = x.Loss (y, LossType.MeanAbsoluteError);

            Assert.AreEqual (Math.Abs (x[0] - y[0]), loss[0]);
        }

        [Test]
        public void Hinge ()
        {
            var x = Tensor.Constant (-0.2f, 1);
            var y = Tensor.Constant (0.9f, 1);
            var loss = x.Loss (y, LossType.Hinge);

            Assert.AreEqual (Math.Max (0.0, 1.0 - x[0] * y[0]), loss[0], 0.05);
        }
    }
}
