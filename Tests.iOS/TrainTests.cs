using System;
using System.Linq;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class TrainTests
    {
        const float DenseLearningRate = 1e-5f;
        const int DenseDataCount = 100_000;
        const int DenseBatchSize = 100;

        const float DenseMaxTrainedLoss = 0.2f;
        const float SphereRadius = 0.1f;
        const float DenseOutputPrecision = SphereRadius * 0.04f;

        const float DenseLossClip = 1.0e-2f;
        const float DenseSampleDistance = 1.0e-3f;

        private static Model CreateDenseModel ()
        {
            var input = Tensor.Input (3);
            var output =
                input
                .Dense (512).ReLU ()
                .Dense (512).ReLU ()
                .Dense (512).ReLU ()
                .Dense (512).ReLU ()
                .Dense (512).ReLU ().Concat(input)
                .Dense (512).ReLU ()
                .Dense (512).ReLU ()
                .Dense (512).ReLU ()
                .Dense (1)
                .Tanh ();
            var model = new Model (input, output);
            return model;
        }

        private static DataSet CreateDenseData ()
        {
            return DataSet.Generated ((index, dev) => {
                GetRow (index, out var input, out var output);
                var inputs = new[] { input };
                var outputs = new[] { output };
                return (inputs, outputs);
            }, DenseDataCount);
        }

        static DataSet CreateDenseAddLossData ()
        {
            return DataSet.Generated ((index, dev) => {
                GetRow (index, out var input, out var output);
                var inputs = new[] { input, output };
                var outputs = Array.Empty<Tensor> ();
                return (inputs, outputs);
            }, DenseDataCount);
        }

        static readonly float InsideScale = 2.0f * MathF.Sqrt (SphereRadius * SphereRadius / 3.0f);

        static void GetRow (int index, out Tensor input, out Tensor output)
        {
            var inside = (index % 2) == 0;
            var x = (float)StaticRandom.NextDouble () - 0.5f;
            var y = (float)StaticRandom.NextDouble () - 0.5f;
            var z = (float)StaticRandom.NextDouble () - 0.5f;
            var r = 1.0f / MathF.Sqrt (x * x + y * y + z * z);
            x *= r;
            y *= r;
            z *= r;
            float depth;
            if (inside) {
                depth = (float)Math.Abs (StaticRandom.NextNormal () * DenseSampleDistance);
            }
            else {
                if (StaticRandom.Next (2) == 0) {
                    depth = (float)-Math.Abs (StaticRandom.NextNormal () * DenseSampleDistance);
                }
                else {
                    depth = (float)-StaticRandom.NextDouble ();
                }
            }
            var signedDistance = -depth;
            x *= SphereRadius + signedDistance;
            y *= SphereRadius + signedDistance;
            z *= SphereRadius + signedDistance;
            //Console.WriteLine ($"X={x}, Y={y}, Z={z}, D={signedDistance}");
            input = Tensor.Array (x, y, z);
            output = Tensor.Array (signedDistance);
        }

        static void ValidateDense (Model model)
        {
            var d = DenseSampleDistance / 2.0f;
            var r = model.Predict (Tensor.Array (SphereRadius + d, 0.0f, 0.0f));
            Assert.AreEqual (d, r[0], DenseOutputPrecision);
            var r2 = model.Predict (Tensor.Array (0.0f, SphereRadius - d, 0.0f));
            Assert.AreEqual (-d, r2[0], DenseOutputPrecision);
        }

        //[Test]
        public void Dense ()
        {
            var model = CreateDenseModel ();
            model.Compile (Loss.MeanAbsoluteError, DenseLearningRate);
            var data = CreateDenseData ();
            var history = model.Fit (data, batchSize: DenseBatchSize, epochs: 1.0f);
            Assert.True (history.Batches[^1].AverageLoss < DenseMaxTrainedLoss);
            ValidateDense (model);
        }

        //[Test]
        public void DenseSubModel ()
        {
            var input = Tensor.Input (3);
            var smodel = CreateDenseModel ();
            var output = smodel.Call (input);
            var model = new Model (input, output);
            model.Compile (Loss.MeanAbsoluteError, DenseLearningRate);
            var data = CreateDenseData ();
            var history = model.Fit (data, batchSize: DenseBatchSize, epochs: 1.0f);
            Assert.True (history.Batches[^1].AverageLoss < DenseMaxTrainedLoss);
            ValidateDense (smodel);
        }

        //[Test]
        public void DenseSubModelAddLoss ()
        {
            var input = Tensor.Input (3);
            var expected = Tensor.Input (1);
            var smodel = CreateDenseModel ();
            var output = smodel.Call (input);
            var model = new Model (new[] { input, expected }, new[] { output });
            var totalLoss = output.Loss (expected, Loss.MeanAbsoluteError);
            model.AddLoss (totalLoss);
            model.Compile (new AdamOptimizer (DenseLearningRate));
            var data = CreateDenseAddLossData ();
            var history = model.Fit (data, batchSize: DenseBatchSize, epochs: 1.0f);
            Assert.True (history.Batches[^1].AverageLoss < DenseMaxTrainedLoss);
            ValidateDense (smodel);
        }

        

        [Test]
        public void DenseSubModelAddClippedLoss ()
        {
            var input = Tensor.Input (3);
            var expected = Tensor.Input (1);
            var smodel = CreateDenseModel ();
            var output = smodel.Call (input);
            var model = new Model (new[]{input,expected}, new[] { output });
            var clipOutput = output.Clip (-DenseLossClip, DenseLossClip);
            var clipExpected = expected.Clip (-DenseLossClip, DenseLossClip);
            var totalLoss = clipOutput.Loss (clipExpected, Loss.MeanAbsoluteError);
            model.AddLoss (totalLoss);
            model.Compile (new AdamOptimizer (DenseLearningRate));
            var data = CreateDenseAddLossData ();
            var history = model.Fit (data, batchSize: DenseBatchSize, epochs: 1.0f, callback: b => {
                Console.WriteLine ($"SDF {b}");
            });
            Assert.True (history.Batches[^1].AverageLoss < DenseMaxTrainedLoss);
            ValidateDense (smodel);
        }

        [Test]
        public void TrainImageClassifier ()
        {
            var input = Tensor.InputImage ("input image", 256, 256);
            var output = input.Conv (32, stride: 2).Conv (32, stride: 2).Conv (1);

            var label = Tensor.Zeros (64, 64, 1);

            var batchSize = 5;

            var getDataCount = 0;

            var model = new Model (input, output);
            model.Compile (Loss.MeanSquaredError, learningRate: Optimizer.DefaultLearningRate);

            var history = model.Fit (DataSet.Generated ((_, device) => {
                getDataCount++;
                return (new Tensor[] { input }, new[]{ label });
            }, 100), batchSize: batchSize, epochs: 1.0f);

            Assert.AreEqual (100, getDataCount);

            Assert.AreEqual (1, history.Batches[0].Losses.Count);
            Assert.AreEqual (0, history.Batches[0].IntermediateValues.Count);
        }
    }
}
