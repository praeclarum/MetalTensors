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

        [Test]
        public void SerializeConstTensor ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var data = t.Config.Serialize ();
            Assert.IsTrue (data.Length > 10);
        }

        [Test]
        public void DeserializeConstTensor ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var data = t.Config.Serialize ();
            var t2 = Config.Deserialize<ConstantTensor> (data);
            Assert.AreEqual (24.0f, t2.ConstantValue);
        }
    }
}
