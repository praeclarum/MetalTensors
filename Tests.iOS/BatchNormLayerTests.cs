using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class BatchNormLayerTests
    {
        [Test]
        public void DefaultShape ()
        {
            var y = Tensor.Constant (0.9f, 3, 5, 7).BatchNorm ();

            Assert.AreEqual (3, y.Shape[0]);
            Assert.AreEqual (5, y.Shape[1]);
            Assert.AreEqual (7, y.Shape[2]);
        }

        [Test]
        public void DefaultEpsilon ()
        {
            var y = Tensor.Constant (0.9f, 3, 5, 7).BatchNorm ();

            Assert.AreEqual (0.9f, y[0], 0.001);
        }

        [Test]
        public void SmallEpsilon ()
        {
            var y = Tensor.Constant (0.9f, 3, 5, 7).BatchNorm (epsilon: 1e-5f);

            Assert.AreEqual (0.9f, y[0], 1e-5f);
        }
    }
}
