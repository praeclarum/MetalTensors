using System;
using System.Collections.Generic;
using System.Diagnostics;
using MetalTensors;
using MetalTensors.Applications;
using MetalTensors.Layers;
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
            var m = y.Model (x);

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (1, m.Inputs.Length);
            Assert.AreEqual (1, m.Sources.Length);
            Assert.AreEqual (0, m.Layers.Length);
        }

        [Test]
        public void TwoInput ()
        {
            var x0 = Tensor.Input ("x0");
            var x1 = Tensor.Input ("x1");
            var y = x0 + x1;
            var m = y.Model (x0, x1);

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (2, m.Inputs.Length);
            Assert.AreEqual (2, m.Sources.Length);
            Assert.AreEqual (1, m.Layers.Length);
        }

        [Test]
        public void Dense ()
        {
            var x = Tensor.Input ("x");
            var y = x.Dense (16).Tanh ().Dense (1).Tanh ();
            var m = y.Model (x);

            Assert.AreEqual (1, m.Outputs.Length);
            Assert.AreEqual (1, m.Inputs.Length);
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
                .Model (z, "generator");

            Assert.AreEqual (1, generator.Inputs.Length);
            Assert.AreEqual (z, generator.Input);
            Assert.AreEqual (1, generator.Outputs.Length);
            Assert.AreEqual (1, generator.Sources.Length);
            Assert.AreEqual (9, generator.Layers.Length);

            var dinput = Tensor.InputImage ("dinput", height, width);
            var discriminator = dinput.Conv (16).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (32, stride: 2).Tanh ()
                .Conv (1)
                .Model (dinput, "discriminator");
            discriminator.Compile (Loss.SigmoidCrossEntropy, new AdamOptimizer ());

            Assert.AreEqual (dinput, discriminator.Input);
            Assert.AreEqual (1, discriminator.Output.Shape[0]);
            Assert.AreEqual (1, discriminator.Output.Shape[1]);
            Assert.AreEqual (1, discriminator.Output.Shape[2]);

            discriminator.IsTrainable = false;
            var gan = discriminator.Call (generator);
            gan.Compile (new AdamOptimizer ());

            Assert.AreEqual (1, gan.Inputs.Length);
            Assert.AreEqual (z, gan.Input);
            Assert.AreEqual (1, gan.Output.Shape[0]);
            Assert.AreEqual (1, gan.Output.Shape[1]);
            Assert.AreEqual (1, gan.Output.Shape[2]);
            Assert.AreEqual (2, gan.Submodels.Length);

            var h = gan.Fit (DataSet.Generated (GetTrainingData, 35), batchSize: 5, epochs: 1);

            (Tensor[], Tensor[]) GetTrainingData (int _)
            {
                return (new[] { Tensor.Ones (height, width, 3) }, new[]{ Tensor.Ones (1, 1, 1) });
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
            Assert.AreEqual (3, mnist.Output.Shape.Length);
            Assert.AreEqual (10, mnist.Output.Shape[^1]);
        }

        [Test]
        public void CompileCustomLoss ()
        {
            var x = Tensor.Input ("x", 3);
            var y = x.Dense (32).ReLU ().Dense (5);
            var model = new Model (x, y);
            model.Compile (Loss.Custom (CustomLoss));
            Tensor CustomLoss (Tensor prediction, Tensor truth)
            {
                return (prediction - truth).Abs ().SpatialMean ();
            }
        }

        [Test]
        public void CompileModelAddLoss ()
        {
            var x = Tensor.Input ("x", 3);
            var y = x.Dense (32).ReLU ().Dense (5);
            var model = new Model (x, y);
            model.AddLoss ((y - 1).Abs ());
            Assert.AreEqual (x, model.Input);
            model.Compile ();
        }

        [Test]
        public void CompileInputAddLoss ()
        {
            var x = Tensor.Input ("x", 3);
            var denseLayer = new DenseLayer (3, 32);
            var y = denseLayer.Call(x).ReLU ().Dense (5);
            var model = new Model (x, y);
            denseLayer.AddLoss ((y - 1).Abs ());
            Assert.AreEqual (x, model.Input);
            var cm = model.Compile ();
            Assert.AreEqual (1, cm.Losses.Length);
        }
    }
}
