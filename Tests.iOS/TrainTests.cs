using System;
using System.Linq;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class TrainTests
    {
        [Test]
        public void TrainImageClassifier ()
        {
            var input = Tensor.InputImage ("input image", 256, 256);
            var output = input.Conv (32, stride: 2).Conv (32, stride: 2).Conv (1);

            var label = Tensor.Zeros (64, 64, 1);

            var batchSize = 5;
            var numBatches = 10;
            var valInterval = 2;

            var getDataCount = 0;

            var model = new Model (output);
            model.Compile (LossType.MeanSquaredError, learningRate: Optimizer.DefaultLearningRate);

            var history = model.Fit (DataSet.Generated (_ => {
                getDataCount++;
                return new Tensor[] { input, label };
            }, 100, "input image", "label image"), batchSize: batchSize, numBatches: numBatches, validationInterval: valInterval);

            Assert.AreEqual (batchSize * numBatches + (numBatches / valInterval) * batchSize, getDataCount);

            Assert.AreEqual (numBatches, history.Batches.Length);
            Assert.AreEqual (batchSize, history.Batches[0].Loss.Length);
            Assert.AreEqual (1, history.Batches[0].IntermediateValues.Count);
        }
    }
}
