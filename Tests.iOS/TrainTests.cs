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
            var input = Tensor.ReadImageResource ("elephant", "jpg");
            var output = input.Conv (32, stride: 2).Conv (32, stride: 2).Conv (1);

            var label = Tensor.Zeros (128, 128, 1);
            var loss = output.Loss (label, LossType.MeanSquaredError);

            var history = loss.Train (inputs => {
                Assert.AreEqual (2, inputs.Length);
                return inputs.Select (x => x.Tensor.Clone ()).ToArray ();
            }, batchSize: 5, numBatches: 3);
        }
    }
}
