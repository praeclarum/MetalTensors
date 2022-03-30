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
            var output = input.LeakyReLU (0.4f);

            Assert.AreEqual (1, output.Shape.Length);
            Assert.AreEqual (1, output.Shape[0]);

            Assert.AreEqual (1, output[0]);
        }

        [Test]
        public void Two ()
        {
            var input = Tensor.Constant (2.0f, 1);
            var output = input.LeakyReLU (0.4f);
            Assert.AreEqual (2.0f, output[0]);
        }

        [Test]
        public void Half ()
        {
            var input = Tensor.Constant (0.5f, 1);
            var output = input.LeakyReLU (0.4f);
            Assert.AreEqual (0.5f, output[0]);
        }

        [Test]
        public void Zero ()
        {
            var input = Tensor.Constant (0, 1);
            var output = input.LeakyReLU (0.4f);
            Assert.AreEqual (0, output[0]);
        }

        [Test]
        public void NegativeHalf ()
        {
            var input = Tensor.Constant (-0.5f, 1);
            var output = input.LeakyReLU (0.4f);
            Assert.AreEqual (-0.2f, output[0], 1.0e-6f);
        }

        [Test]
        public void NegativeHalfOtherAlpha ()
        {
            var input = Tensor.Constant (-0.5f, 1);
            var output = input.LeakyReLU (0.1f);
            Assert.AreEqual (-0.05f, output[0], 1.0e-6f);
        }


        [Test]
        public void ClipHigh ()
        {
            var input = Tensor.Constant (0.42f, 1);
            var output = input.Clip (-0.2f, 0.2f);
            Assert.AreEqual (0.2f, output[0], 1.0e-6f);
            var output2 = input.Clip (-0.2f, 0.5f);
            Assert.AreEqual (0.42f, output2[0], 1.0e-6f);
        }

        [Test]
        public void ClipLow ()
        {
            var input = Tensor.Constant (-0.42f, 1);
            var output = input.Clip (-0.2f, 0.2f);
            Assert.AreEqual (-0.2f, output[0], 1.0e-6f);
            var output2 = input.Clip (-0.5f, 0.2f);
            Assert.AreEqual (-0.42f, output2[0], 1.0e-6f);
        }

        [Test]
        public void ClipNone ()
        {
            var input = Tensor.Constant (0.12f, 1);
            var output = input.Clip (-0.2f, 0.2f);
            Assert.AreEqual (0.12f, output[0], 1.0e-6f);
        }

        [Test]
        public void NegateClip0 ()
        {
            var input = Tensor.Input (1);
            var output = (0.0f - input).Clip (0.0f, 0.06f);
            var model = new Model (input, output);
            Assert.AreEqual (0.06f, model.Predict (Tensor.Constant (-0.12f))[0], 1.0e-6f);
            Assert.AreEqual (0.0f, model.Predict (Tensor.Constant (0.12f))[0], 1.0e-6f);
        }

        [Test]
        public void NegateClip0Tiny ()
        {
            var input = Tensor.Input (1);
            var output = (0.0f - input).Clip (0.0f, 1e-4f);
            var model = new Model (input, output);
            Assert.AreEqual (1e-4f, model.Predict (Tensor.Constant (-0.12f))[0], 1.0e-6f);
            Assert.AreEqual (0.0f, model.Predict (Tensor.Constant (0.12f))[0], 1.0e-6f);
        }
    }
}
