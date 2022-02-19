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

        [Test]
        public void DeserializeConstAddTensor ()
        {
            var t = new ConstantTensor (24.0f, 2, 3, 5);
            var data = (t + 11.0f).Config.Serialize ();
            var t2 = Config.Deserialize<Tensor> (data);
            Assert.AreEqual (35.0f, t2[0]);
        }

        [Test]
        public void DeserializeLeakyReLUTensor ()
        {
            var t = Tensor.Constant (-10.0f, 2, 3, 5).LeakyReLU (0.3f);
            var dataStr = t.Config.StringValue;
            var data = t.Config.Serialize ();
            var t2 = Config.Deserialize<Tensor> (data);
            Assert.AreEqual (35.0f, t2[0]);
        }
    }
}
