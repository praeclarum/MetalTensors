using System;
using MetalTensors;
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
            var m = new Model (y);

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (1, m.Inputs.Length);
            Assert.AreEqual (0, m.Labels.Length);
            Assert.AreEqual (1, m.Sources.Length);
            Assert.AreEqual (0, m.Layers.Length);
            Assert.AreEqual (x, m.TrainingTensor);
        }

        [Test]
        public void TwoInput ()
        {
            var x0 = Tensor.Input ("x0");
            var x1 = Tensor.Input ("x1");
            var y = x0 + x1;
            var m = new Model (y);

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
            var m = new Model (y);

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (1, m.Inputs.Length);
            Assert.AreEqual (0, m.Labels.Length);
            Assert.AreEqual (1, m.Sources.Length);
            Assert.AreEqual (4, m.Layers.Length);
            Assert.AreEqual (y, m.TrainingTensor);
        }

        [Test]
        public void Gan ()
        {
            var z = Tensor.InputImage ("z", 4, 4);

            var generatedImage = z.Conv (16).Tanh ().Upsample ().Conv (16).Tanh ().Upsample ().Conv (16).Tanh ().Conv (3);
            var generator = new Model (generatedImage);

            Assert.AreEqual (1, generator.Outputs.Length);
            Assert.AreEqual (1, generator.Inputs.Length);
            Assert.AreEqual (0, generator.Labels.Length);
            Assert.AreEqual (1, generator.Sources.Length);
            Assert.AreEqual (9, generator.Layers.Length);

            var discriminatedImage = z.Conv (16).Tanh ().Conv (32, stride:2).Tanh ().Conv (32, stride: 2).Tanh ().Conv (1);
            var discriminator = new Model (discriminatedImage);

            Assert.AreEqual (1, discriminator.Outputs[0].Shape[0]);
            Assert.AreEqual (1, discriminator.Outputs[0].Shape[1]);
            Assert.AreEqual (1, discriminator.Outputs[0].Shape[2]);

            var gan = generatedImage.Apply(discriminator);

            Assert.AreEqual (1, gan.Shape[0]);
            Assert.AreEqual (1, gan.Shape[1]);
            Assert.AreEqual (1, gan.Shape[2]);

        }
    }
}
