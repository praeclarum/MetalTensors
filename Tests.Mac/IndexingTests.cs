using System;
using MetalTensors;
using NUnit.Framework;

namespace Tests
{
    public class IndexingTests
    {
        [Test]
        public void OneHotMultiArrayIndexing ()
        {
            var t = Tensor.Array (new int[] { 2, 2, 3 },
                1, 0, 0, 1, 0, 0,
                1, 0, 0, 1, 0, 0);
            Assert.AreEqual (1, t[0, 0, 0]);
            Assert.AreEqual (1, t[0, 1, 0]);
            Assert.AreEqual (1, t[1, 0, 0]);
            Assert.AreEqual (1, t[1, 1, 0]);
        }

        [Test]
        public void OneHotImageIndexing ()
        {
            var t = Tensor.OneHot (0, 2, 2, 3);
            Assert.AreEqual (1, t[0, 0, 0]);
            Assert.AreEqual (1, t[0, 1, 0]);
            Assert.AreEqual (1, t[1, 0, 0]);
            Assert.AreEqual (1, t[1, 1, 0]);
        }
    }
}
