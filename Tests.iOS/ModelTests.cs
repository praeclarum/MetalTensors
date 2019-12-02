using System;
using System.Collections.Generic;
using System.Diagnostics;
using MetalTensors;
using MetalTensors.Applications;
using NUnit.Framework;

namespace Tests
{
    public class ModelTests
    {
        [Test]
        public void Identity ()
        {
            var x = Tensor.Input ("x");
            var y = x;
            var m = y.Model ();

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (1, m.Inputs.Length);
            Assert.AreEqual (0, m.Labels.Length);
            Assert.AreEqual (1, m.Sources.Length);
            Assert.AreEqual (0, m.Layers.Length);
        }

        [Test]
        public void TwoInput ()
        {
            var x0 = Tensor.Input ("x0");
            var x1 = Tensor.Input ("x1");
            var y = x0 + x1;
            var m = y.Model ();

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (2, m.Inputs.Length);
            Assert.AreEqual (0, m.Labels.Length);
            Assert.AreEqual (2, m.Sources.Length);
            Assert.AreEqual (1, m.Layers.Length);
        }

        [Test]
        public void Dense ()
        {
            var x = Tensor.Input ("x");
            var y = x.Dense (16).Tanh ().Dense (1).Tanh ();
            var m = y.Model ();

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (1, m.Inputs.Length);
            Assert.AreEqual (0, m.Labels.Length);
            Assert.AreEqual (1, m.Sources.Length);
            Assert.AreEqual (4, m.Layers.Length);
        }

        //[Test]
        public void Gan ()
        {
            var height = 8;
            var width = 8;

            var z = Tensor.InputImage ("z", 2, 2, 1);

            var generator = z.Conv (16).Tanh ()
                .Upsample ().Conv (16).Tanh ()
                .Upsample ().Conv (16).Tanh ()
                .Conv (3)
                .Model ("generator");

            Assert.AreEqual (1, generator.Inputs.Length);
            Assert.AreEqual (z, generator.Input);
            Assert.AreEqual (1, generator.Outputs.Length);
            Assert.AreEqual (0, generator.Labels.Length);
            Assert.AreEqual (1, generator.Sources.Length);
            Assert.AreEqual (9, generator.Layers.Length);

            var dinput = Tensor.InputImage ("dinput", height, width);
            var discriminator = dinput.Conv (16).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (1)
                .Loss (Tensor.Labels ("realOrFake", 1, 1, 1), LossType.SigmoidCrossEntropy)
                .Model ("discriminator");

            Assert.AreEqual (dinput, discriminator.Input);
            Assert.AreEqual (1, discriminator.Output!.Shape[0]);
            Assert.AreEqual (1, discriminator.Output!.Shape[1]);
            Assert.AreEqual (1, discriminator.Output!.Shape[2]);

            var gan = discriminator.Lock ().Apply (generator);

            Assert.AreEqual (1, gan.Inputs.Length);
            Assert.AreEqual (z, gan.Input);
            Assert.AreEqual (1, gan.Output!.Shape[0]);
            Assert.AreEqual (1, gan.Output!.Shape[1]);
            Assert.AreEqual (1, gan.Output!.Shape[2]);
            Assert.AreEqual (2, gan.Submodels.Length);

            var h = gan.Train (DataSet.Generated (GetTrainingData, 35, "z", "realOrFake"), batchSize: 5, epochs: 1);

            Tensor[] GetTrainingData (int _)
            {
                return new[] { Tensor.Ones (height, width, 3), Tensor.Ones (1, 1, 1) };
            }

            Assert.AreEqual (7, h.Batches.Length);
        }

        [Test]
        public void Mnist ()
        {
            var mnist = MnistApplication.CreateModel ();

            Assert.AreEqual (1, mnist.Inputs.Length);
            Assert.AreEqual (28, mnist.Input!.Shape[0]);
            Assert.AreEqual (28, mnist.Input!.Shape[1]);
            Assert.AreEqual (1, mnist.Input.Shape[2]);
            Assert.AreEqual (1, mnist.Outputs.Length);
            Assert.AreEqual (1, mnist.Output!.Shape.Length);
            Assert.AreEqual (1, mnist.Output!.Shape[0]);
        }
    }
}
