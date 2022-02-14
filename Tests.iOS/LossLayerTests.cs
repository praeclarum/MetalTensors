using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class LossLayerTests
    {
        Tensor BLoss (Tensor prediction, Tensor truth, LossType lossType, ReductionType reductionType = ReductionType.Mean, float weight = 1.0f)
        {
            return new BuiltinLoss (lossType, reductionType).Call (prediction, truth, weight);
        }

        [Test]
        public void MAESum ()
        {
            var x = Tensor.Constant (800, 3);
            var y = Tensor.Constant (1000, 3);
            var loss = BLoss (x, y, LossType.MeanAbsoluteError, ReductionType.Sum);

            Assert.AreEqual (1, loss.Shape.Length);
            Assert.AreEqual (1, loss.Shape[0]);

            Assert.AreEqual (Math.Abs (x[0] - y[0]) * 3, loss[0], 1.0e-4);
        }

        [Test]
        public void MAEMean ()
        {
            var x = Tensor.Constant (800, 3);
            var y = Tensor.Constant (1000, 3);
            var loss = BLoss (x, y, LossType.MeanAbsoluteError, ReductionType.Mean);

            Assert.AreEqual (1, loss.Shape.Length);
            Assert.AreEqual (1, loss.Shape[0]);

            Assert.AreEqual (Math.Abs (x[0] - y[0]), loss[0], 1.0e-4);
        }

        [Test]
        public void MSE ()
        {
            var x = Tensor.Constant (800, 1);
            var y = Tensor.Constant (1000, 1);
            var loss = BLoss (x, y, LossType.MeanSquaredError);

            Assert.AreEqual (1, loss.Shape.Length);
            Assert.AreEqual (1, loss.Shape[0]);

            Assert.AreEqual (Math.Pow(x[0] - y[0], 2.0), loss[0]);
        }

        [Test]
        public void MAE ()
        {
            var x = Tensor.Constant (800, 1);
            var y = Tensor.Constant (1000, 1);
            var loss = BLoss (x, y, LossType.MeanAbsoluteError);

            Assert.AreEqual (Math.Abs (x[0] - y[0]), loss[0]);
        }

        [Test]
        public void WeightedMAE ()
        {
            var x = Tensor.Constant (800, 1);
            var y = Tensor.Constant (1000, 1);
            var loss = BLoss (x, y, LossType.MeanAbsoluteError, weight: 0.5f);

            Assert.AreEqual (Math.Abs (x[0] - y[0]) * 0.5f, loss[0]);
        }

        [Test]
        public void Hinge ()
        {
            var x = Tensor.Constant (-0.2f, 1);
            var y = Tensor.Constant (0.9f, 1);
            var loss = BLoss (x, y, LossType.Hinge);

            Assert.AreEqual (Math.Max (0.0, 1.0 - x[0] * y[0]), loss[0], 0.05);
        }

        [Test]
        public void BadLabelShape ()
        {
            var output = Tensor.Zeros (2, 2, 1);
            var label = Tensor.Zeros (1, 1, 1);
            Assert.Throws<ArgumentException> (() => BLoss (output, label, LossType.MeanSquaredError));
        }
    }
}
