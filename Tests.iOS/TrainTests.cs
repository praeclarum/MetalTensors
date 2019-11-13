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
            var image = Tensor.ReadImageResource ("elephant", "jpg");
            var output = image.Conv (3, 3);

            var loss = output.Loss (image, LossType.MeanSquaredError);

            var history = loss.Train (inputs => {
                return inputs.Select (x => x.Tensor).ToArray ();
            });
        }
    }
}
