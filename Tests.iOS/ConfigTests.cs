using System;
using System.IO;
using MetalPerformanceShaders;
using MetalTensors;
using MetalTensors.Applications;
using MetalTensors.Tensors;
using NUnit.Framework;

using static Tests.Imaging;

namespace Tests
{
    public class ConfigTests
    {
        [Test]
        public void ConstTensorConfig ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var c = t.Config;
            Assert.AreEqual (24.0f, c["constant"]);
        }
    }
}
