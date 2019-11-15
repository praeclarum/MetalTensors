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
            var height = 8;
            var width = 8;

            var z = Tensor.InputImage ("z", 2, 2);

            var generator = z.Conv (16).Tanh ().Upsample ().Conv (16).Tanh ().Upsample ().Conv (16).Tanh ().Conv (3).Model ();

            Assert.AreEqual (1, generator.Inputs.Length);
            Assert.AreEqual (z, generator.Inputs[0]);
            Assert.AreEqual (1, generator.Outputs.Length);
            Assert.AreEqual (0, generator.Labels.Length);
            Assert.AreEqual (1, generator.Sources.Length);
            Assert.AreEqual (9, generator.Layers.Length);

            var dinput = Tensor.InputImage ("dinput", height, width);
            var discriminatedImage = dinput.Conv (16).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (1);
            var discriminator = new Model (discriminatedImage);

            Assert.AreEqual (dinput, discriminator.Inputs[0]);
            Assert.AreEqual (1, discriminator.Outputs[0].Shape[0]);
            Assert.AreEqual (1, discriminator.Outputs[0].Shape[1]);
            Assert.AreEqual (1, discriminator.Outputs[0].Shape[2]);

            var gan = discriminator.Apply (generator);

            Assert.AreEqual (z, gan.Inputs[0]);
            Assert.AreEqual (1, gan.Outputs[0].Shape[0]);
            Assert.AreEqual (1, gan.Outputs[0].Shape[1]);
            Assert.AreEqual (1, gan.Outputs[0].Shape[2]);

        }
    }
}
