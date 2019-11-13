using System;
using System.Linq;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class TrainTests
    {
        [Test]
        public void ConvIdentity ()
        {
            var input = Tensor.InputImage (256, 256);
            var output = input.Conv (32, stride: 2).Conv (32, stride: 2).Conv (1);

            var label = Tensor.Labels (64, 64, 1);
            var loss = output.Loss (label, LossType.MeanSquaredError);

            var batchSize = 5;
            var numBatches = 3;

            var getDataCount = 0;

            var history = loss.Train (needed => {
                Assert.AreEqual (2, needed.Length);
                Assert.AreEqual (input.Label, needed[0].Label);
                Assert.AreEqual (label.Label, needed[1].Label);
                getDataCount++;

                return needed.Select (x => x.Tensor.Clone ()).ToArray ();
            }, batchSize: batchSize, numBatches: numBatches);

            Assert.AreEqual (batchSize * numBatches, getDataCount);

            Assert.AreEqual (numBatches, history.Losses.Length);
        }
    }
}
