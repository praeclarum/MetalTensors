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

        [Test]
        public void UniformInit23 ()
        {
            using var v = Vector (23);
            v.UniformInitAsync (10.0f, 20.0f, 456).Wait();
            Assert.AreEqual (23, (int)v.Length);
            var s = v.ToSpan ();
            foreach (var x in s) {
                if (!float.IsFinite (x))
                    Assert.Fail ($"Non-finite value found: {x}");
                if (x < 10.0f || x > 20.0f) {
                    Assert.Fail($"Uniform init out of range: {x}");
                }
            }
        }

        [Test]
        public void NormalInit23 ()
        {
            using var v = Vector (23);
            v.NormalInitAsync (100.0f, 2.0f, 456).Wait ();
            Assert.AreEqual (23, (int)v.Length);
            var s = v.ToSpan ();
            foreach (var x in s) {
                if (!float.IsFinite (x))
                    Assert.Fail ($"Non-finite value found: {x}");
                var d = MathF.Abs (x - 100.0f);
                if (d > 50.0f) {
                    Assert.Fail ($"Uniform init out of range: {x}");
                }
            }
        }
    }
}
