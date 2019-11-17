using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class BatchNormLayerTests
    {
        [Test]
        public void DefaultShape ()
        {
            var y = Tensor.Constant (1.0f, 3, 5, 7).BatchNorm ();

            Assert.AreEqual (3, y.Shape[0]);
            Assert.AreEqual (5, y.Shape[1]);
            Assert.AreEqual (7, y.Shape[2]);
            Assert.AreEqual (0.0f, y[0]);
        }
    }
}
