using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class ReductionLayerTests
    {
        [Test]
        public void ReduceMeanShape ()
        {
            var v = 42.123f;
            var input = Tensor.Constant (v, 2, 3, 5);
            var output = input.ReduceMean ();

            Assert.AreEqual (3, output.Shape.Length);
            Assert.AreEqual (1, output.Shape[0]);
            Assert.AreEqual (1, output.Shape[1]);
            Assert.AreEqual (5, output.Shape[2]);
            Assert.AreEqual (v, output[0]);
        }

        [Test]
        public void MeanShape ()
        {
            var v = 42.123f;
            var input = Tensor.Constant (v, 2, 3, 1);
            var output = input.Mean ();

            Assert.AreEqual (3, output.Shape.Length);
            Assert.AreEqual (2, output.Shape[0]);
            Assert.AreEqual (3, output.Shape[1]);
            Assert.AreEqual (1, output.Shape[2]);
            Assert.AreEqual (v, output[0]);
        }
    }
}
