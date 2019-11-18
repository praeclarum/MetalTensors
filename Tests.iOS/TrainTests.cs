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

            var label = Tensor.Labels ("label image", 64, 64, 1);
            var loss = output.Loss (label, LossType.MeanSquaredError);

            var batchSize = 5;
            var numBatches = 10;
            var valInterval = 2;

            var getDataCount = 0;

            var history = loss.Train (needed => {
                Assert.AreEqual (2, needed.Length);
                Assert.AreEqual (input.Label, needed[0].Label);
                Assert.AreEqual ("input image", needed[0].Label);
                Assert.AreEqual (label.Label, needed[1].Label);
                Assert.AreEqual ("label image", needed[1].Label);
                getDataCount++;

                return needed.Select (x => x.Tensor);
            }, batchSize: batchSize, numBatches: numBatches, validationInterval: valInterval);

            Assert.AreEqual (batchSize * numBatches + (numBatches / valInterval) * batchSize, getDataCount);

            Assert.AreEqual (numBatches, history.Batches.Length);
            Assert.AreEqual (batchSize, history.Batches[0].Loss.Length);
            Assert.AreEqual (1, history.Batches[0].IntermediateValues.Count);
            Assert.AreEqual (batchSize, history.Batches[0].IntermediateValues[loss.Label].Length);
        }
    }
}
