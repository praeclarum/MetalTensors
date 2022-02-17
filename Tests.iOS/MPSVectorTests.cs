using System;
using MetalTensors;
using NUnit.Framework;

using static MetalTensors.MetalHelpers;

namespace Tests
{
    public class MPSVectorTests
    {
        [Test]
        public void Uninitialized23 ()
        {
            using var v = Vector (23);
            Assert.AreEqual (23, (int)v.Length);
        }

        [Test]
        public void Init23 ()
        {
            using var v = Vector (42.123f, 23);
            Assert.AreEqual (23, (int)v.Length);
            var s = v.ToSpan ();
            foreach (var x in s) {
                Assert.AreEqual (42.123f, x);
            }
        }
    }
}
