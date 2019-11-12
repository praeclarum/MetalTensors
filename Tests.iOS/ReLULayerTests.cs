using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class ReLULayerTests
    {
        [Test]
        public void One ()
        {
            var input = Tensor.Constant (1, 1);
            var output = input.ReLU (0.4f);

            Assert.AreEqual (1, output.Shape.Length);
            Assert.AreEqual (1, output.Shape[0]);

            Assert.AreEqual (1, output[0]);
        }

        [Test]
        public void Two ()
        {
            var input = Tensor.Constant (2.0f, 1);
            var output = input.ReLU (0.4f);
            Assert.AreEqual (2.0f, output[0]);
        }

        [Test]
        public void Half ()
        {
            var input = Tensor.Constant (0.5f, 1);
            var output = input.ReLU (0.4f);
            Assert.AreEqual (0.5f, output[0]);
        }

        [Test]
        public void Zero ()
        {
            var input = Tensor.Constant (0, 1);
            var output = input.ReLU (0.4f);
            Assert.AreEqual (0, output[0]);
        }

        [Test]
        public void NegativeHalf ()
        {
            var input = Tensor.Constant (-0.5f, 1);
            var output = input.ReLU (0.4f);
            Assert.AreEqual (-0.2f, output[0], 1.0e-6f);
        }

        [Test]
        public void NegativeHalfOtherAlpha ()
        {
            var input = Tensor.Constant (-0.5f, 1);
            var output = input.ReLU (0.1f);
            Assert.AreEqual (-0.05f, output[0], 1.0e-6f);
        }
    }
}
