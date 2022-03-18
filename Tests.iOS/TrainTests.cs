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
        const float DenseRadius = 0.1f;
        const float DenseOutputPrecision = DenseRadius * 0.04f;

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

        static readonly float InsideScale = 2.0f * MathF.Sqrt (DenseRadius * DenseRadius / 3.0f);

        static void GetRow (int index, out Tensor input, out Tensor output)
        {
            var inside = (index % 2) == 0;
            var x = (float)StaticRandom.NextDouble () - 0.5f;
            var y = (float)StaticRandom.NextDouble () - 0.5f;
            var z = (float)StaticRandom.NextDouble () - 0.5f;
            if (inside) {
                x *= InsideScale;
                y *= InsideScale;
                z *= InsideScale;
            }
            var r = MathF.Sqrt (x * x + y * y + z * z);
            var d = r - 0.1f;
            input = Tensor.Array (x, y, z);
            output = Tensor.Array (d);
        }

        static void ValidateDense (Model model)
        {
            var r = model.Predict (Tensor.Array (0.12f, 0.0f, 0.0f));
            Assert.AreEqual (0.02f, r[0], DenseOutputPrecision);
            var r2 = model.Predict (Tensor.Array (0.0f, -0.06f, 0.0f));
            Assert.AreEqual (-0.04f, r2[0], DenseOutputPrecision);
        }

        [Test]
        public void Dense ()
        {
            var model = CreateDenseModel ();
            model.Compile (Loss.MeanAbsoluteError, DenseLearningRate);
            var data = CreateDenseData ();
            var history = model.Fit (data, batchSize: DenseBatchSize, epochs: 1.0f);
            Assert.True (history.Batches[^1].AverageLoss < DenseMaxTrainedLoss);
            ValidateDense (model);
        }

        [Test]
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

        [Test]
        public void DenseSubModelAddLoss ()
        {
            var input = Tensor.Input (3);
            var expected = Tensor.Input (1);
            var smodel = CreateDenseModel ();
            var output = smodel.Call (input);
            var model = new Model (new[]{input,expected}, new[] { output });
            var totalLoss = output.Loss (expected, Loss.MeanAbsoluteError);
            model.AddLoss (totalLoss);
            model.Compile (new AdamOptimizer (DenseLearningRate));
            var data = CreateDenseAddLossData ();
            var history = model.Fit (data, batchSize: DenseBatchSize, epochs: 1.0f);
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
